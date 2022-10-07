import csv
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import models  # NQDA
from common.arguments import get_args
from common.logging import ExperimetManager, setup_logger
from common.visualization import embed_dot, embed_text, visualize
from data_loader.dataset import Optos_Dataset
from evaluate import AverageMeter, Timer, eval, log_format, metrics
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from rich.progress import track


def demo(
    model,
    loader,
    show: bool = False,
    statistics: bool = True,
    save_pics: bool = False,
    gif: bool = False,
    save_folder: str = "../res",
):
    model.eval()
    avg_meter = AverageMeter()
    timer = Timer()

    with torch.no_grad():
        for it, img_mask_tuple in track(
            enumerate(loader), total=len(loader), description="evaluating"
        ):
            img = img_mask_tuple["img"].to(device)
            mask = img_mask_tuple["mask"]  # .to(device)
            if "sub_img" in img_mask_tuple:
                sub_img = img_mask_tuple["sub_img"]  # .to(device)
            else:
                sub_img = None
            raw = img_mask_tuple["raw"]  # .to(device)

            with timer("model-inference"):
                pred = model(img)  # ["out"]
            pred = pred.to("cpu")

            metrics_dict = metrics(pred, mask)
            metrics_dict["file_name"] = img_mask_tuple["file_name"][0]
            avg_meter.update(metrics_dict)

            fig = visualize(pred, raw, mask, sub_img, flatten=save_pics, gif=gif)
            if show and not gif:
                plt.imshow(fig.transpose(1, 2, 0))
                plt.pause(0.01)
                plt.close("all")

            if save_pics:
                metrics_text = log_format(metrics_dict)

                if gif:
                    pil_imgs: list(Image) = [
                        Image.fromarray(img.transpose(1, 2, 0)) for img in fig
                    ]
                    [
                        embed_text(
                            pil_img, metrics_text, font_size=int(args.base_size * 0.02)
                        )
                        for pil_img in pil_imgs
                    ]
                    # pil_imgs = [sub_img, sub_mask, raw, mask_raw, pred_raw, mask_pred]
                    embed_dot(pil_imgs[1], metrics_dict["centroid_gt"], fill="red")
                    embed_dot(pil_imgs[3], metrics_dict["centroid_gt"], fill="red")
                    embed_dot(pil_imgs[4], metrics_dict["centroid_pred"], fill="red")
                    embed_dot(pil_imgs[5], metrics_dict["centroid_gt"], fill="red")
                    embed_dot(pil_imgs[5], metrics_dict["centroid_pred"], fill="red")
                    pil_imgs[0].save(
                        "{}/{}.gif".format(save_folder, it),
                        save_all=True,
                        append_images=pil_imgs[1:],
                        duration=1500,
                        loop=0,
                    )
                else:
                    pil_img = Image.fromarray(fig.transpose(1, 2, 0))
                    # FIXME: add font
                    # embed_text(
                    #     pil_img, metrics_text, font_size=int(cfg.base_size * 0.03 * 0.2)
                    # )
                    pil_img.save(
                        "{}/{}".format(save_folder, img_mask_tuple["file_name"][0])
                    )

    metrics_dict = avg_meter.compute_average()
    print(log_format(metrics_dict))
    logger.info(log_format(metrics_dict))

    with open(os.path.join(save_folder, "res.csv"), "w") as f:
        writer = csv.writer(f)
        rows = avg_meter.export_csv(
            [
                "file_name",
                "dice",
                "iou",
                "accuracy",
                "precision",
                "recall",
                "area_gt",
                "area_pred",
                "centroid_gt",
                "centroid_pred",
                "centroid_l1",
            ]
        )
        writer.writerows(rows)

    plt.scatter(avg_meter.get_column("area_gt"), avg_meter.get_column("dice"))
    plt.savefig(os.path.join(save_folder, "dice_area.png"), dpi=300)
    print("Inferrence Finished. ")
    return metrics_dict[cfg.metric]


def main(args):
    model = models.get_model(cfg.arch, resume=args.resume)
    model.to(device)

    dataset = Optos_Dataset(
        cfg,
        phase="test",
        img_types=args.img_types,
        read_sub_img=args.read_sub_img,
        num_non_npa_epoch=args.num_non_npa_epoch,
        annotators=args.annotators,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.test_batch_size, shuffle=False
    )
    test_score = demo(
        model,
        test_loader,
        show=False,
        statistics=True,
        save_pics=args.save_pics,
        save_folder=save_folder,
        gif=args.gif,
    )
    experiment_manager = ExperimetManager(args.master_log, orig_logdir)
    experiment_manager.insert_item("test score", test_score)


def load_cfg(args, logdir, test_cfg="test.yaml", train_cfg="config/base.yaml"):
    if os.path.exists(os.path.join(logdir, test_cfg)):
        cfg_path = os.path.join(logdir, test_cfg)
        cfg = OmegaConf.load(cfg_path)
        print("Load config from {}".format(cfg_path))
    else:
        cfg = OmegaConf.load(train_cfg)
        cfg.base_size = args.base_size
        print("Load config from {}".format(train_cfg))

    cfg.num_non_npa_epoch = (
        args.num_non_npa_epoch
    )  # we should specify this at test time

    return cfg


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(1)
    random.seed(1)

    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = "cpu" if args.gpus == "-1" else "cuda:0"
    # device = torch.cuda.device(device)
    print("Using {} for inference".format(device))

    orig_logdir = os.path.dirname(args.resume)
    save_folder = os.path.join(orig_logdir, "res" + "-".join(args.annotators))
    cfg = load_cfg(args, orig_logdir)

    logger = setup_logger(save_folder, file_name="test.log")
    logger.info(args)

    main(args)
