from typing import List
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchinfo import summary
from torchvision import transforms
import SimpleITK as sitk

from . import sync_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Optos_Dataset(Dataset):
    def __init__(
        self,
        cfg,
        phase,
        img_types=["npa"],
        read_sub_img=False,
        logger=None,
        num_non_npa_epoch=0,
        annotators: List[str]=None,
    ) -> None:
        """The custom dataset class for the npa inference task of Optos California.

        Args:
            cfg (Omegaconf): config file object.
            phase (str): "train", "val" or "test".
            img_type (str, optional): "mix", "npa" or "non_npa". Defaults to "npa".
            read_sub_img (bool, optional): whether read sub_img (Fluorescein Angiography: FA) or not. Defaults to False.
            mixup (bool, optional): Defaults to False.
        """
        self.cfg = cfg
        self.phase = phase
        self.img_types = img_types  # choices = ['mix', 'npa', 'non_npa']
        self.read_sub_img = read_sub_img
        self.annotators = annotators

        self._pre_transform = transforms.CenterCrop(cfg.center_size)
        self.img_transform = get_img_transform(cfg, phase=phase)
        self.sub_img_transform = get_img_transform(cfg, phase=phase, img_type="sub_img")
        self.sync_transform = get_sync_transform(cfg, phase=phase)

        # path_pattern = os.path.join(cfg.root, phase, cfg.path_pattern[img_type])
        # img_types = ["non_npa"]
        self.img_paths = {"npa": [], "non_npa": []}
        for img_type in self.img_types:
            print("path_pattern", cfg.path_pattern)
            print("img_type", img_type)
            self.img_paths[img_type] = glob.glob(
                os.path.join(cfg.root, phase, cfg.path_pattern[img_type])
            )
        self.num = {
            "npa": len(self.img_paths["npa"]),
            "non_npa": len(self.img_paths["non_npa"]),
            "non_npa_epoch": num_non_npa_epoch,
        }
        print("self.num = ", self.num)

        if logger != None:
            logger.info("NPA    : {}".format(self.num["npa"]))
            logger.info(
                "Non NPA: {} / {}".format(
                    self.num["non_npa_epoch"], self.num["non_npa"]
                )
            )

    def __len__(self) -> int:
        return self.num["npa"] + self.num["non_npa"]

    def __getitem__(self, idx):
        print("getitem")
        if idx >= self.num["npa"]:
            if self.phase == "train":
                idx = np.random.randint(0, self.num["non_npa"])
            else:
                idx = idx - self.num["npa"]
            npa_class = "non_npa"
        else:
            npa_class = "npa"

        img_path = self.img_paths[npa_class][idx]
        mask_dir_names = [
            "/Segmentation_0_" + annotator + "/" for annotator in self.annotators
        ]
        mask_paths = [
            img_path.replace("/Images/", mask_dir_name)
            for mask_dir_name in mask_dir_names
        ]
        sub_img_path = img_path.replace("/Images/", "/SubImages/")

        img_mask_tuple = {}

        print(img_path)
        raw = Image.open(img_path)
        raw = self._pre_transform(raw)
        img = self.img_transform(raw)
        img_mask_tuple["raw"] = raw
        img_mask_tuple["img"] = img

        if npa_class == "npa":
            if len(self.annotators) == 1:
                mask_path = mask_paths[0]
                print("mask_path", mask_path)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path)
                else:
                    # アノテーション領域が一切ない場合は画像が存在しない
                    mask = Image.new(
                        "L", (self.cfg.center_size, self.cfg.center_size), (0)
                    )
                img_mask_tuple["mask"] = self._pre_transform(mask)
            else:
                # annotators が 2 名以上の場合は、 STAPLE により共通 GT を定義する
                masks = []
                for mask_path in mask_paths:
                    print(mask_path)
                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path)
                    else:
                        mask = Image.new(
                            "L", (self.cfg.center_size, self.cfg.center_size), (0)
                        )
                    mask = self._pre_transform(mask)
                    mask = (np.array(mask) / 255).astype(np.int16)
                    masks.append(sitk.GetImageFromArray(mask))

                staple_mask = sitk.STAPLE(masks)
                staple_mask = staple_mask >= 0.5
                img_mask_tuple["mask"] = Image.fromarray(
                    sitk.GetArrayFromImage(staple_mask) * 255
                )
        elif npa_class == "non_npa":
            # non_npa 画像の場合は指摘領域なしのアノテーション画像を作成する
            mask = Image.new("L", (self.cfg.center_size, self.cfg.center_size), (0))
            img_mask_tuple["mask"] = self._pre_transform(mask)

        else:
            raise NotImplementedError(
                f"npa_class should be npa or non_npa, but {npa_class}"
            )

        if self.read_sub_img and npa_class == "npa":
            sub_img = Image.open(sub_img_path).convert("RGB")
            sub_img = self._pre_transform(sub_img)
            sub_img = self.sub_img_transform(sub_img)[0:1, :, :]
            img_mask_tuple["sub_img"] = sub_img

        img_mask_tuple = self.sync_transform(img_mask_tuple)

        # img_mask_tuple['raw'] = np.array(raw).transpose(2, 0, 1)
        img_mask_tuple["raw"] = np.array(img_mask_tuple["raw"]).transpose(2, 0, 1)
        img_mask_tuple["file_name"] = os.path.basename(img_path)

        return img_mask_tuple

    def cal_batch_size(self, model, is_per_gpu=False):
        num_gpus = torch.cuda.device_count()
        memory = torch.cuda.get_device_properties(0).total_memory
        memory = memory / (1 << 30) * 0.8  # for marginize

        img_size = [1, 3, self.cfg.base_size, self.cfg.base_size]
        model_stat = summary(model, input_size=img_size, verbose=0)
        model_size = _flaot_to_gigabytes(model_stat.total_output)

        batch_par_gpu = int(memory // model_size)
        if batch_par_gpu < 2:
            print("You may encounter OOM")
        batch_par_gpu = min(batch_par_gpu, 128 // num_gpus)
        batch_par_gpu = max(batch_par_gpu, 2)

        if is_per_gpu:
            return batch_par_gpu
        else:
            return batch_par_gpu * num_gpus

    def __len__(self):
        return self.num["npa"] + self.num["non_npa_epoch"]


class NNCFExampler(Optos_Dataset):
    def __init__(self, cfg, phase, img_types=["npa"], read_sub_img=False, logger=None):
        super().__init__(
            cfg, phase, img_types=img_types, read_sub_img=read_sub_img, logger=logger
        )

    def __getitem__(self, idx):
        img_mask_tuple = super().__getitem__(idx)
        return img_mask_tuple["img"], img_mask_tuple["mask"]


def _flaot_to_gigabytes(num: int) -> float:
    return num * 4 / (1 << 30)


def mixup(tuple1, tuple2):
    ratio = np.random.uniform(0, 1)
    out = []
    for i in range(len(tuple1)):
        img = tuple1[i] * ratio + tuple2[i] * (1 - ratio)
        out.append(img)
    return out


class ToGrayTenor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0.46
        self.std = 0.18
        self.to_tensor = transforms.ToTensor()

    def forward(self, img):
        img = self.to_tensor(img)
        # print('img.size = ', img.shape)
        _, h, w = img.shape
        img = img[0] * 0.299 + img[1] * 0.587 + img[2] + 0.144
        img = (img - self.mean) / self.std
        img = img.reshape(1, h, w)
        return img


class RandomGamma(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self._gain = gain

    def forward(self, img):
        gain = np.random.uniform(1 - self._gain, 1 + self._gain)
        img = torchvision.transforms.functional.adjust_gamma(img, gain)
        return img


def get_img_transform(cfg, phase="train", img_type="img"):
    if "train" in phase:
        transform = transforms.Compose(
            [
                transforms.ColorJitter(**cfg.colorjitter),
                transforms.ToTensor(),
                transforms.Normalize(**cfg.stat[img_type]),
            ]
        )
    elif phase in ["val", "test"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(**cfg.stat[img_type]),
            ]
        )
    else:
        raise NotImplementedError
    return transform


def get_sync_transform(cfg, phase="train"):
    if "train" in phase:
        transform = transforms.Compose(
            [
                sync_transforms.SyncRandomAffine(**cfg.affine),
                sync_transforms.SyncRandomResizedCrop(
                    cfg.base_size, scale=1, ratio=cfg.ratio
                ),
                sync_transforms.SyncRandomCrop(cfg.crop_size),  # for OOM
                sync_transforms.SyncHorizontalFlip(),
                sync_transforms.MaskToTensor(),
            ]
        )
    elif phase == "val":
        transform = transforms.Compose(
            [
                sync_transforms.SyncRandomResizedCrop(
                    cfg.base_size, scale=1.0, ratio=1.0
                ),
                sync_transforms.SyncRandomCrop(cfg.crop_size),  # for OOM
                sync_transforms.MaskToTensor(),
            ]
        )
    elif phase == "test":
        transform = transforms.Compose(
            [
                sync_transforms.SyncRandomResizedCrop(
                    cfg.base_size, scale=1.0, ratio=1.0
                ),
                sync_transforms.MaskToTensor(),
            ]
        )
    else:
        raise NotImplementedError
    return transform
