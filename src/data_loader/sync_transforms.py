import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as tf


class SyncRandomCrop(transforms.RandomCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img_mask_pair):
        img = img_mask_pair["img"]
        i, j, h, w = self.get_params(img, output_size=self.size)

        for key, value in img_mask_pair.items():
            img_mask_pair[key] = tf.crop(value, i, j, h, w)

        return img_mask_pair


class SyncRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale, ratio):
        super().__init__(size)
        self.scale = [scale, 2.0 - scale]
        self.ratio = [ratio, 2.0 - ratio]

    def forward(self, img_mask_pair):
        img = img_mask_pair["img"]
        i, j, h, w = super().get_params(img, self.scale, self.ratio)

        for key, value in img_mask_pair.items():
            img_mask_pair[key] = tf.resized_crop(
                value, i, j, h, w, self.size, self.interpolation
            )
        return img_mask_pair


class SyncRandomAffine(transforms.RandomAffine):
    def __init__(self, degrees, translate, scale):
        super().__init__(degrees)
        self.degrees = [0, degrees]
        self.translate = [translate, translate]
        self.scale_ranges = [scale, 2.0 - scale]

    def forward(self, img_mask_pair):
        img = img_mask_pair["img"]
        params = super().get_params(
            degrees=self.degrees,
            translate=self.translate,
            scale_ranges=self.scale_ranges,
            img_size=img.shape[-2:],
            shears=None,
        )

        for key, value in img_mask_pair.items():
            img_mask_pair[key] = tf.affine(value, *params)

        return img_mask_pair


class SyncRandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img_mask_pair):
        degree = np.random.uniform(-self.degrees, self.degrees)

        for key, value in img_mask_pair.items():
            img_mask_pair[key] = tf.rotate(value, degree)
        return img_mask_pair


class MaskToTensor:
    def __init__(self):
        self.trans = transforms.ToTensor()

    def __call__(self, img_mask_pair):
        mask = img_mask_pair["mask"]

        mask_t = self.trans(mask).squeeze()
        mask_t = torch.round(mask_t)
        mask_t = mask_t.to(torch.long)

        img_mask_pair["mask"] = mask_t
        return img_mask_pair


class SyncHorizontalFlip:
    def __call__(self, img_mask_pair):
        is_flip = np.random.uniform(0, 1) < 0.5
        if is_flip:
            for key, value in img_mask_pair.items():
                img_mask_pair[key] = tf.hflip(value)
        return img_mask_pair
