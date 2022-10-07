import glob
import os

import torch
from PIL import Image
from rich.progress import track
from torchvision import transforms


def is_img_file(file_path):
    """str (file path) -> bool (is image file?)"""
    try:
        ext = os.path.splitext(file_path)[-1]
    except:
        return False
    img_exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    if ext in img_exts:
        return True
    else:
        return False


def calc_mean_std(dataset_root):
    mean_over_all = 0
    std_over_all = 0
    cnt = 0
    id_transform = transforms.Compose([transforms.ToTensor()])

    paths = glob.glob("{}/train/NPA/Images/*.png".format(dataset_root))[:50]

    for path in track(paths):
        img = Image.open(path)
        img = img.crop((700, 700, 3300, 3300))
        img = id_transform(img)
        mean = torch.mean(img.view(3, -1), axis=1)
        std = torch.std(
            img.view(3, -1), axis=1
        ) 

        mean_over_all += mean
        std_over_all += std

        cnt += 1

    return mean_over_all / cnt, std_over_all / cnt


def main():
    dataset_root = "../dataset"
    mean, std = calc_mean_std(dataset_root)

    print("mean = ", mean)
    print("std  = ", std)


if __name__ == "__main__":
    main()
