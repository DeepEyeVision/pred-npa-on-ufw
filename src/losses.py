import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-7


def _flaot_to_gigabytes(num: int) -> float:
    return num * 4 / (1 << 30)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, pred, target):
        if pred.shape[1] == 2:
            # B, 2, H, W
            pred = torch.softmax(pred, dim=-3)
            logpt = torch.log(pred + _EPS)
            # pt = pred.detach()
            loss = -self.alpha * pred[:, 0] ** self.gamma * logpt[:, 1] * target
            loss -= (
                (1.0 - self.alpha)
                * pred[:, 1] ** self.gamma
                * logpt[:, 0]
                * (1 - target)
            )
            return loss.mean()
        else:
            target = target.squeeze()
            pred = torch.sigmoid(pred)
            pred = pred.squeeze()
            loss = (
                -self._alpha
                * (1.0 - pred.detach()) ** self._gamma
                * torch.log(pred + _EPS)
                * target
            )
            loss += (
                -(1.0 - self._alpha)
                * pred.detach() ** self._gamma
                * torch.log(1.0 - pred + _EPS)
                * (1 - target)
            )
            return loss.mean()


class DiceFocal(nn.Module):
    def __init__(self, gamma=0, alpha=None, mu=1.0):
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha
        self._mu = mu

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            target = target.squeeze()
            pred = torch.sigmoid(pred)
            pred = pred.squeeze()

            loss = (
                -self._alpha
                * (1.0 - pred.detach()) ** self._gamma
                * torch.log(pred + _EPS)
                * target
            )
            loss += (
                -(1.0 - self._alpha)
                * pred.detach() ** self._gamma
                * torch.log(1.0 - pred + _EPS)
                * (1 - target)
            )

            inter = torch.sum(pred * target, dim=(1, 2))
            union = torch.sum(pred + target, dim=(1, 2))
            loss_dice = 1.0 - (2 * inter + 1.0) / (union + 1.0)  # NOTE \in [0, 1]

            loss = loss.mean() + self._mu * loss_dice.mean()
            return loss
        else:
            raise NotImplementedError


class StructureFocal(FocalLoss):
    def __init__(self, gamma=0, alpha=0.5, mu=5, kernel_size=31):
        super().__init__(gamma=gamma, alpha=alpha)
        self._mu = mu
        self._kernel_size = kernel_size

    def forward(self, pred, mask):
        B, H, W = mask.shape
        mask = mask.reshape(B, 1, H, W)

        if pred.shape[1] == 2:
            pred = pred[:, 1].reshape(
                B, 1, H, W
            )
        weit = 1 + self._mu * torch.abs(
            F.avg_pool2d(
                mask,
                kernel_size=self._kernel_size,
                stride=1,
                padding=self._kernel_size // 2,
            )
            - mask
        )
        wbce = super().forward(pred, mask) * 2.0
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(
            dim=(2, 3)
        )

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class StructureLoss(nn.Module):
    def __init__(self, mu=5):
        super().__init__()
        warnings.warn("Deprecated, integrated to StructureFocal")
        self._mu = mu

    def forward(self, pred, mask):
        B, H, W = mask.shape
        mask = mask.reshape(B, 1, H, W)
        if pred.shape[1] == 2:
            pred = pred[:, 1].reshape(
                B, 1, H, W
            )
        weit = 1 + self._mu * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(
            dim=(2, 3)
        )

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


def l2_loss(pred, sub_img):
    return torch.mean((pred - sub_img) ** 2)


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
