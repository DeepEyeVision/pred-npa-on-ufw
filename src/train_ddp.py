from logging import log
import os
import sys
import time
import numpy as np
from numpy.random import shuffle

import torch
from torch.distributed.distributed_c10d import get_world_size
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import sampler
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.tensorboard import SummaryWriter
from rich.progress import track
from scipy.ndimage.filters import uniform_filter1d
import optuna

from common.arguments import get_args
from common.logging import setup_logger, save_model, log_vcs
from common.visualization import tb_visualzie
from models import deeplabv3
from data_loader.dataset import Optos_Dataset 
from evaluate import eval
from train_misc import HyperParams, FocalLoss



def train(rank, dataset, trial=None, hp=None, args=None, logger=None): # hp stands for Hyper Parameters
    world_size = torch.cuda.device_count()
    print('world_size = ', world_size)
    setup_DDP(rank, world_size) 
    global best_iou
    model = deeplabv3.get_model(method=args.learning_method, high_res=args.high_res)
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay,
                            betas=hp.betas, eps=hp.eps)

    criterion = FocalLoss(gamma=hp.fl_gamma, alpha=torch.tensor([1 - hp.alpha, hp.alpha]))
    criterion.to(rank)
 
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=hp.gamma)
    logger.info('Using {} images for train'.format(len(dataset) - args.num_val))
    iou_history = []

    for epoch in range(args.epochs):

        # NOTE for cross validation
        dataset.trainval_split(args.num_val, seed=epoch)
        dataset.train()

        dist_sampler = DistributedSampler(dataset, rank=rank, shuffle=True) 
        dist_sampler.set_epoch(epoch)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size // world_size, shuffle=False,
                                                    pin_memory=True, num_workers=16, drop_last=True, sampler=dist_sampler)

        print('hoge')
        for img_mask_pair in train_loader:
            img, mask = img_mask_pair
            print('size', img.shape)
            img = img.to(rank, non_blocking=True)
            mask = mask.to(rank, non_blocking=True)

            out = model(img)
            pred = out['out']
            pred_aux = out['aux']
            
            loss = criterion(pred, mask)
            loss_aux = criterion(pred_aux, mask)
            loss_total = loss + hp.aux_weight * loss_aux
            loss_total.backward()

            optimizer.step()
            optimizer.zero_grad()

        if rank == 0 and (epoch + 1) % args.validate_freq == 0:

            dataset.eval()
            val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
            accuracy, recall, iou = eval(model, val_loader)
            logger.info('accuracy = {}'.format(accuracy))
            logger.info('recall   = {}'.format(recall))
            logger.info('IOU      = {}'.format(iou))

            iou_history.append(iou)            

            if iou > best_iou:
                logger.info('Best Score Updated at epoch {}'.format(epoch))
                best_iou = iou
                save_model(model, os.path.join(args.logdir, 'model-best.pth'))
                tb_visualzie(model, val_loader, writer, epoch)
            if epoch > 100 and trial is not None:
                trial.report(iou, epoch)
                if trial.should_prune():
                    logger.info('Trial Pruned\n')
                    raise optuna.exceptions.TrialPruned()

        #lr_scheduler.step()

    if rank == 0:
        smoothed_history = uniform_filter1d(iou_history, 3)
        score = max(smoothed_history)
        logger.info('Succenssfully Train Finished')
        cleanup_DDP()

        return score
    else:
        cleanup_DDP()
    


class Objective:
    def __init__(self, dataset):
        self.dataset = dataset
    def __call__(self, trial):

        # learning parameters
        lr = trial.suggest_loguniform('lr', 5.0e-5, 2.0e-4)
        weight_decay = trial.suggest_loguniform('weight_decay', 5e-7, 5e-6)
        #weight_decay = 1.0e-6
        #beta1_aux = trial.suggest_loguniform('beta1_aux', 0.05, 0.2)
        #beta2_aux = trial.suggest_loguniform('beta2_aux', 0.0001, 0.001)
        #beta1 = 1.0 - beta1_aux
        #beta2 = 1.0 - beta2_aux
        beta1 = 0.88
        beta2 = 0.9999
        #eps = trial.suggest_loguniform('eps', 5e-8, 5e-7)
        eps = 2e-7
        gamma = trial.suggest_loguniform('gamma', 0.1, 0.6)
        #gamma = 0.48
        #aux_weight = trial.suggest_loguniform('aux_weight', 3.0, 8)
        aux_weight = 4.5 
        alpha = trial.suggest_uniform('alpha', 0.65, 0.8)
        #class_wise_weight = 1.0
        fl_gamma = trial.suggest_loguniform('fl_gamma', 1.0, 2.5)
        hp = HyperParams(
            lr=lr, weight_decay=weight_decay,
            betas=[beta1, beta2], eps=eps,
            gamma=gamma, aux_weight=aux_weight,
            alpha=alpha,
            fl_gamma = fl_gamma
        )

        logger.info('#=========== Starting with params of ... ===========#')
        logger.info('aux weight   = {}'.format(aux_weight))
        logger.info('alpha        = {}'.format(alpha))
        logger.info('fl gamma     = {}'.format(fl_gamma))
        logger.info('lr           = {}'.format(lr))
        logger.info('betas        = [{}, {}]'.format(beta1, beta2))
        logger.info('weight decay = {}'.format(weight_decay))
        logger.info('gamma        = {}'.format(gamma))
        logger.info('eps          = {}'.format(eps))
        logger.info('num nonNPA 1e= {}'.format(args.num_nonNPA_1epoch))


        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(self.dataset, trial, hp, args, logger), nprocs=world_size)
        score = train(self.dataset, trial=trial, hp=hp, threshold=0.25)

        logger.info('Best IOU     = {}'.format(score))
        logger.info('One trial finished\n')

        return score


def setup_DDP(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '38861'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup_DDP():
    dist.destroy_process_group()


def main():
    dataset = Optos_Dataset(args.dataset_root, 896, num_nonNPA_1epoch=args.num_nonNPA_1epoch)
    optuna_objective = Objective(dataset)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
        )
    study.optimize(optuna_objective, n_trials=args.trials)
    
    logger.info(study.best_params)
    logger.info(study.best_value)

    logger.info('Hyper Parameter Search Finished')
    print('hyper parameter search finished')



if __name__ == '__main__':
    args = get_args()

    # prepare logger
    args.logdir = os.path.join(args.logdir, '{}'.format(time.strftime('%Y_%m%d_%H%M')))
    logger = setup_logger(args.logdir)
    logger.info(args)
    log_vcs(logger, args.logdir)
    writer = SummaryWriter(log_dir=args.logdir)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    best_iou = 0.0

    main()

