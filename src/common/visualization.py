import os
import sys
from typing import List

import numpy as np
import torch
import torchvision
from numpy.core.records import array
from PIL import Image, ImageDraw, ImageFont

sys.path.append(".")


@torch.no_grad()
def tb_visualize(model, test_loader, writer, epoch):
    model.eval()
    for it, img_mask_tuple in enumerate(test_loader):
        img = img_mask_tuple["img"]  # .cuda()
        mask = img_mask_tuple["mask"]  # .cuda()
        if "sub_img" in img_mask_tuple:
            sub_img = img_mask_tuple["sub_img"]  # .cuda()
        else:
            sub_img = None
        raw = img_mask_tuple["raw"]  # .cuda()

        pred = model(img)  # ["out"]
        pred = pred.to("cpu")

        fig = visualize(pred, raw, mask, sub_img)
        writer.add_image("{}_{}".format(epoch, it), fig)
    model.train()


def visualize(pred, raw, mask, sub_img, flatten: bool = False, gif: bool = False):
    # to unnnormalized form
    if pred.shape[1] == 1:
        B, _, H, W = pred.shape
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > 0.5, 1, 0).reshape(B, H, W)
    else:
        pred = torch.argmax(pred, dim=-3)

    fig = _get_segres(raw, mask, pred, sub_img, flatten=flatten, gif=gif)
    return fig


def _tensor_to_8bit_img(img, mean, std):
    """tensor[any](float[-1.0, 1.0]) -> tensor[any](long[0, 256])"""
    inv_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=-np.asarray(mean) / np.asarray(std), std=1.0 / np.asarray(std)
            ),
        ]
    )
    img = inv_transform(img)
    img = (img * 255).to(torch.long)
    if len(mean) == 1:
        img = img.expand(-1, 3, -1, -1)
    return img


def _indices_to_timg(mask, color="green"):
    """tensor[B, H, W](long[0, class]) -> tensor[B, H, W, 3](long[0, 256])"""
    if color == "green":
        pallete_opt = [[0, 0, 0], [0, 255, 0]]
    elif color == "blue":
        pallete_opt = [[0, 0, 0], [0, 0, 255]]
    elif color == "white":
        pallete_opt = [[0, 0, 0], [255, 255, 255]]
    else:
        raise NotImplementedError("color must be green or blue")
    palette = torch.Tensor(pallete_opt)  # .cuda()
    mask_np = mask.to(torch.long)
    mask_c = palette[mask_np]
    mask_c = mask_c.permute(0, 3, 1, 2)
    return mask_c


def _blend_tensor(img, mask, alpha):
    """tensor[B, 3, H, W](any) -> tensor[B, 3, H, W](any)"""
    blended_img = (1 - alpha) * img + alpha * mask
    return blended_img


def _resize_arr(arr: np.array, ratio: float = 0.2):
    # arr = arr[0]
    assert arr.ndim == 3
    arr = np.uint8(arr).transpose(1, 2, 0)
    pil = Image.fromarray(arr)
    H, W = pil.size
    downsized_pil = pil.resize((int(H * ratio), int(W * ratio)))
    downsized_img = np.asarray(downsized_pil).transpose(2, 0, 1)
    return downsized_img


def _stack_plot(imgs: List[torch.Tensor], ratio: float = 1.0) -> List[np.array]:
    """
    Args:
        imgs : tensor[N, 3, H, W]

    Return:
        [np.array([3, H, W]), ...]
    """
    imgs: list[np.array] = [img.cpu().numpy() for img in imgs]
    downsized_imgs: list[np.array] = [_resize_arr(img, ratio=ratio) for img in imgs]
    return downsized_imgs


def _grid_plot(img, ratio=0.2, flatten=False):
    """tensor[N, 3, H, W] -> numpy[3, H, W]"""
    if flatten:
        B = img[0].shape[0] * len(img)
    else:
        B = img[0].shape[0]
    img = torch.cat(img)
    """
    if flatten:
        B = img.shape[0] 
    else:
        B = img.shape[0] // 2
    """
    if B in [2, 3]:
        ratio = 1.0
    joint_imgs = torchvision.utils.make_grid(img, nrow=B, padding=10)
    joint_imgs = joint_imgs.cpu().numpy()
    downsized_imgs = _resize_arr(joint_imgs, ratio=ratio)
    return downsized_imgs


def _get_segres(
    raw, mask, pred, sub_img=None, flatten: bool = False, gif: bool = False
):
    """tensor[B, 3, H, W](uint8[0, 256], long[0, class], long[0, class]) -> numpy.uint8[3, H, W](uint8[0, 256])"""
    mask = _indices_to_timg(mask, color="blue")
    pred = _indices_to_timg(pred, color="green")
    mask_raw = _blend_tensor(raw, mask, 0.6)
    pred_raw = _blend_tensor(raw, pred, 0.6)
    mask_pred = _blend_tensor(mask, pred, 0.6)

    if sub_img == None:
        # FA画像を結果に含めない場合
        fig = _grid_plot([mask_raw, pred_raw], flatten=flatten)
    else:
        # sub_imgの血管部分を白、背景を黒とする
        sub_img = 1.0 - sub_img.repeat(1, 3, 1, 1)
        # TODO: sub_img と mask を _blend_tensorすると sub_img がほぼ消えてしまうので修正
        sub_mask = _blend_tensor(sub_img, mask, 0.6)
        imgs = [sub_img, sub_mask, raw, mask_raw, pred_raw, mask_pred]
        if gif:
            fig = _stack_plot(imgs)
        else:
            fig = _grid_plot(imgs, flatten=flatten)

    fig = np.uint8(fig)
    return fig


def embed_text(img, text, font_size=25):
    size = img.size
    num_linesep = text.count(os.linesep)
    pos = (font_size, size[1] - font_size * 1.25 * num_linesep)

    draw = ImageDraw.Draw(img)
    draw.font = ImageFont.truetype(
        "/work/.fonts/Menlo-Regular.ttf",
        font_size,
    )
    draw.text(pos, text, fill="#FFF")


def embed_dot(img, pos: tuple, r: int = 10, fill: str = "red"):
    draw = ImageDraw.Draw(img)
    ellipse_pos = (pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r)
    draw.ellipse(ellipse_pos, fill=fill, outline=fill)
