import atexit
import copy
import math
import time

import numpy as np
import torch
from scipy.ndimage import center_of_mass

_EPS = 1e-6


def metrics(pred: torch.Tensor, mask: torch.Tensor, threshold=5):
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
        pred = torch.cat([1.0 - pred, pred], dim=1)
    pred_dx = torch.argmax(pred, dim=-3)
    assert mask.shape == pred_dx.shape

    centroid_gt = _calc_centroid(mask)
    centroid_pred = _calc_centroid(pred_dx)
    centroid_l1: float = math.sqrt(
        (centroid_gt[0] - centroid_pred[0]) ** 2
        + (centroid_gt[1] - centroid_pred[1]) ** 2
    ) / max(centroid_pred)

    tp = pred_dx * mask
    tp = torch.sum(tp, dim=[-1, -2])
    tp = tp.to(torch.float32)

    tn = (1.0 - pred_dx) * (1.0 - mask)
    tn = torch.sum(tn, dim=[-1, -2])
    tn = tn.to(torch.float32)

    tp_fn = torch.sum(mask, dim=[-1, -2])
    tp_fp = torch.sum(pred_dx, dim=[-1, -2])

    accuracy = (tp + tn) / (tp_fn + tp_fp + tn - tp)
    precision = tp / (tp_fp + _EPS)
    recall = tp / (tp_fn + _EPS)
    iou = tp / (tp_fn + tp_fp - tp + _EPS)
    dice = 2.0 * tp / (tp_fn + tp_fp + _EPS)

    accuracy = torch.mean(accuracy).item()
    precision = torch.mean(precision).item()
    recall = torch.mean(recall).item()
    iou = torch.mean(iou).item()
    dice = torch.mean(dice).item()
    area_gt = torch.sum(
        mask.to(float) / torch.numel(mask)
    ).item()
    area_pred = torch.sum(pred_dx.to(float) / torch.numel(pred_dx)).item()

    # num_FP_pix = torch.sum(pred_dx, dim=[1, 2])
    # specificity = torch.mean((num_FP_pix < threshold) * 1.0).item()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "area_gt": area_gt,
        "area_pred": area_pred,
        "centroid_gt": centroid_gt,
        "centroid_pred": centroid_pred,
        "centroid_l1": centroid_l1,
    }


def _calc_centroid(mask: torch.Tensor):
    """Calculate centroid of binary mask tensor.
    mask.shape == (1, H, W)"""
    mask = mask.cpu().numpy()
    _, h, w = center_of_mass(mask)
    try:
        centroid: tuple[int] = (int(w), int(h))
        return centroid
    except ValueError:
        return (1, 1)


@torch.no_grad()
def eval(model, data_loader):
    model.eval()
    avg_meter = AverageMeter()

    for img_mask_tuple in data_loader:
        img = img_mask_tuple["img"].cuda()
        mask = img_mask_tuple["mask"].cuda()
        # sub_img = img_mask_tuple["sub_img"].cuda()

        # pred = model(img)["out"]
        pred = model(img)

        metrics_batch = metrics(pred, mask)
        avg_meter.update(metrics_batch)

    model.train()
    return avg_meter.compute_average()


def add_values(dict_a, dict_b):
    if len(dict_a) == 0:
        return copy.deepcopy(dict_b)
    else:
        res = {}
        for key in dict_a:
            res[key] = dict_a[key] + dict_b[key]
        return res


def log_format(metrics_dict):
    text_for_log = ""
    for (
        key,
        value,
    ) in metrics_dict.items():
        if type(value) == str:
            text_for_log += "{}\n".format(value)
        elif type(value) == tuple:
            text_for_log += "{:<10} = {}\n".format(key, str(value))
        else:
            text_for_log += "{:<10} = {:.3f}\n".format(key, value)
    return text_for_log


class AverageMeter:
    def __init__(self):
        self._sum = None
        self._all = {}
        self._cnt = 0

    def reset(self):
        self._all = {}
        self._cnt = 0

    def update(self, dict_batch):
        if self._all == {}:
            for key in dict_batch:
                self._all[key] = []

        for key, value in dict_batch.items():
            self._all[key].append(value)

        self._cnt += 1

    def export_csv(self, keys):  # e.g. keys = ['file_name', 'dice', 'iou']
        csv_list = []
        csv_list.append(keys)
        for i in range(len(self._all[keys[0]])):
            batch = [self._all[key][i] for key in keys]
            csv_list.append(batch)
        return csv_list

    def get_column(self, key):
        return self._all[key]

    def compute_average(self):
        total = {}

        for key, values in self._all.items():
            if type(values[0]) == str:
                continue
            total[key] = np.mean(values)

        return total


class Timer(object):
    def __init__(self):
        self.cnts = {}
        self.starts = {}
        self.totals = {}
        self.fn_stack = []
        atexit.register(self.show_result)

    def __call__(self, fn_name):
        self.fn_stack.append(fn_name)
        if fn_name not in self.cnts:
            self.cnts[fn_name] = 0
            self.totals[fn_name] = 0
        return self

    def __enter__(self):
        fn_name = self.fn_stack[-1]
        self.cnts[fn_name] += 1
        self.starts[fn_name] = time.time()

    def __exit__(self, type, value, traceback):
        fn_name = self.fn_stack.pop()
        end = time.time()

        self.totals[fn_name] += end - self.starts[fn_name]

    def show_result(self):
        for fn_name in self.cnts.keys():
            print(f"# {fn_name}")
            print("Calls     : ", self.cnts[fn_name])
            print("Total Time: ", self.totals[fn_name])
            print("Per    Hit: ", self.totals[fn_name] / self.cnts[fn_name])
