import os
import sys
import time

import nncf  # NOTE 直接呼び出されることはないが，必須．torchのすぐ後にimportする必要がある．
import optuna
import torch
import torch.optim as optim
from nncf import NNCFConfig, create_compressed_model, register_default_init_args
from PIL import ImageFile
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True

import models
from common.arguments import get_args
from common.logging import ExperimetManager, log_vcs, setup_logger
from common.optuna_loader import OptunaConfigConverter
from common.visualization import tb_visualize
from data_loader.dataset import NNCFExampler, Optos_Dataset
from evaluate import eval, log_format
from losses import StructureFocal, clip_gradient


class Trainer(object):
    def __init__(
        self,
        cfg,
        model,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        logdir,
        lr_scheduler=None,
        epochs=1,
        trial=None,
    ):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.loss_weights = cfg.loss_weights
        self.validate_freq = args.validate_freq
        self.trial = trial

        self.logdir = logdir
        self.best_score = 0.0
        self.cur_epoch = 0

    def fit(self):
        for epoch in range(self.epochs):
            logger.info("begin epoch: {}".format(epoch))
            self.cur_epoch = epoch
            self.train_epoch()
            self.train_epoch_end()
            if (epoch + 1) % self.validate_freq == 0:
                self.validate()
        logger.info("Successfully Trial Finished")
        return self.best_score

    def train_epoch(self):
        for it, batch in track(
            enumerate(self.train_loader),
            description="epoch: {} ".format(self.cur_epoch),
            total=len(self.train_loader),
        ):
            batch = self.train_step(batch)

            loss = self.loss_fn(batch)
            loss.backward()
            clip_gradient(self.optimizer, self.cfg.grad_clip)  # FIXME

            self.optimizer.step()
            self.optimizer.zero_grad()

    def train_step(self, batch):
        img = batch["img"].cuda()
        out = self.model(img)
        batch["out"] = out
        return batch

    def loss_fn(self, batch):
        # PraNet
        out = batch["out"]
        mask = batch["mask"].cuda()

        loss_total = 0.0
        for key, out_value in out.items():
            loss = self.criterion(out_value, mask.to(float))
            loss_total += loss * self.loss_weights[key]

        loss_total = loss_total.mean()

        return loss_total

    def train_epoch_end(self):
        if self.lr_scheduler != None:
            self.lr_scheduler.step()

    @torch.no_grad()
    def validate(self):
        metrics = eval(self.model, self.val_loader)
        logger.info("\n{}".format(log_format(metrics)))
        score = metrics[self.cfg.metric]

        self.optuna_prunning(score)

        self.best_score = max(score, self.best_score)
        global global_best_score
        if score > global_best_score:
            logger.info("Best Score Updated !")
            global_best_score = score
            self.save_model()
            self.tb_visualize()

    def tb_visualize(self):
        # tb_visualize(model, val_loader, writer, epoch)
        pass

    def optuna_prunning(self, score):
        self.trial.report(score, self.cur_epoch)
        if self.trial.should_prune():
            logger.info("Trial Pruned\n")
            raise optuna.exceptions.TrialPruned()

    def save_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            state = self.model.module.state_dict()
        else:
            state = self.model.state_dict()
        torch.save(state, os.path.join(self.logdir, "model-best.pth"))


class VinoTrainer(Trainer):
    def __init__(
        self,
        cfg,
        model,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        logdir,
        lr_scheduler=None,
        epochs=1,
        nncf_config=None,
        example_loader=None,
        trial=None,
    ):
        super().__init__(
            cfg,
            model,
            criterion,
            train_loader,
            val_loader,
            optimizer,
            logdir,
            lr_scheduler=lr_scheduler,
            epochs=epochs,
            trial=trial,
        )
        nncf_config = register_default_init_args(
            nncf_config,
            example_loader,
            criterion=self.criterion,
            criterion_fn=self.criterion_fn,
        )
        self.cmp_ctrl, self.model = create_compressed_model(self.model, nncf_config)
        self.cmp_ctrl.scheduler.epoch_step()
        self.model = torch.nn.DataParallel(self.model.cuda())
        self.criterion = torch.nn.DataParallel(self.criterion.cuda())

    def fit(self):
        best_score = super().fit()
        if self.best_score >= global_best_score:
            self.save_model(ignore_this_call=False)
        return best_score

    def train_step(self, batch):
        self.cmp_ctrl.scheduler.step()
        return super().train_step(batch)

    def train_epoch_end(self):
        self.cmp_ctrl.scheduler.epoch_step()

    def loss_fn(self, batch):
        compression_loss = self.cmp_ctrl.loss()
        loss = compression_loss + super().loss_fn(batch)
        return loss

    def criterion_fn(self, outputs, target, criterion):
        """[
            This function is just for generation of nncf_config.
            nncf doesn't accept variadic manner.
            https://github.com/openvinotoolkit/nncf/blob/a8e2ee1f88d12212a195fb37f09dfe64d820885e/nncf/torch/initialization.py#L256
        ]
        Args:
        """
        total_loss = 0.0
        for i, (_, weight) in self.loss_weights.items():
            print("outputs[i] = ", outputs[i].shape)
            loss = criterion(outputs[i], target)
            total_loss += weight * loss

        return total_loss

    def save_model(self, ignore_this_call=True):
        if ignore_this_call:
            pass
        else:
            self.cmp_ctrl.export_model(
                os.path.join(self.logdir, "model-best.onnx"),
                input_names=["input"],
                output_names=["output"],
            )
        # print('after compressed', self.model)
        # print(temp)
        # self.model = torch.nn.DataParallel(temp)
        # self.model = torch.nn.DataParallel(self.model.cuda())
        # self.model = self.model.cuda() # NOTE model compression inplicitly sends the model to cpu
        # print('type(model)', type(self.model))

    def exampler_serialize(self):
        pass


def train(cfg, trial=None):
    # return torch.rand(1).item()
    global global_best_score
    best_score = 0.0

    model = models.get_model(cfg.arch, args.resume)
    model.train()

    criterion = StructureFocal(**cfg.criterion)
    # criterion = StructureLoss()

    train_set = Optos_Dataset(
        cfg,
        phase="train",
        img_types=args.img_types,
        logger=logger,
        num_non_npa_epoch=cfg.num_non_npa_epoch,
    )
    val_set = Optos_Dataset(cfg, phase="val")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_set.cal_batch_size(model),
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    optimizer = optim.Adam(model.parameters(), **cfg.optimizer)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **cfg.lr_scheduler)

    if args.nncf_config != None:
        nncf_config = NNCFConfig.from_json(args.nncf_config)
        nncf_config["input_info"][0]["sample_size"] = [
            1,
            3,
            cfg.base_size,
            cfg.base_size,
        ]

        example_set = NNCFExampler(cfg, "train", img_types=args.img_types)
        batch_per_gpu = train_set.cal_batch_size(model, is_per_gpu=True)
        example_loader = torch.utils.data.DataLoader(example_set, batch_per_gpu)

        trainer = VinoTrainer(
            cfg,
            model,
            criterion,
            train_loader,
            val_loader,
            optimizer,
            args.logdir,
            lr_scheduler,
            nncf_config=nncf_config,
            example_loader=example_loader,
            epochs=args.epochs,
            trial=trial,
        )
    else:
        model = torch.nn.DataParallel(model.cuda())
        criterion = torch.nn.DataParallel(criterion.cuda())
        trainer = Trainer(
            cfg,
            model,
            criterion,
            train_loader,
            val_loader,
            optimizer,
            args.logdir,
            lr_scheduler,
            epochs=args.epochs,
            trial=trial,
        )

    best_score = trainer.fit()

    return best_score


class Objective:
    def __init__(self, cfg_path):
        self.optuna_manager = OptunaConfigConverter(cfg_path, logdir=args.logdir)

    def __call__(self, trial):
        cfg = self.optuna_manager.suggest_params(trial)

        text_params = self.optuna_manager.format_cfg()
        logger.info(text_params)

        score = train(cfg, trial)

        logger.info("Best {}  = {}\nOne trial finished\n".format(cfg.metric, score))
        return score


def main():
    objective = Objective(args.cfg_path)

    study = optuna.create_study(
        direction="maximize",
        study_name="optos_distributed",
        storage=None
        if args.storage == None
        else "mysql://root:username@localhost/{}".format(args.storage),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    objective.optuna_manager.hook_study(study)
    experiment_manager = ExperimetManager(
        args.master_log,
        args.logdir,
        study,
        sys.argv,
        args.message,
        write_md=(not args.debug),
    )

    study.optimize(objective, n_trials=args.trials)

    logger.info(study.best_params)
    logger.info(study.best_value)
    logger.info("Hyper Parameter Search Finished")


if __name__ == "__main__":
    args = get_args()

    logdir = "debug" if args.debug else "{}".format(time.strftime("%Y_%m%d_%H%M"))
    args.logdir = os.path.join(args.logdir, logdir)

    logger = setup_logger(args.logdir, master_log=args.master_log)
    logger.info(args)
    logger.info("This experiment is about: {}".format(args.message))
    log_vcs(logger, args.logdir)

    writer = SummaryWriter(log_dir=args.logdir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    global_best_score = 0.0

    main()
    writer.close()
