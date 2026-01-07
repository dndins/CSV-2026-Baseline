import numpy as np
from torch import nn
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

def compute_nsd(pred, gt, tolerance=1.0):
    """
    pred, gt: (H, W) binary numpy array
    tolerance: pixel distance
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    # surface extraction
    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)

    # distance maps
    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt = distance_transform_edt(~gt_surface)

    # surface points within tolerance
    pred_match = (pred_surface & (dt_gt <= tolerance)).sum()
    gt_match = (gt_surface & (dt_pred <= tolerance)).sum()

    nsd = (pred_match + gt_match) / (pred_surface.sum() + gt_surface.sum() + 1e-8)
    return nsd

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        # Accept input of shape [B,H,W] or [B,1,H,W] and return [B,C,H,W]
        if input_tensor.ndim == 3:
            # [B,H,W] -> [B,1,H,W]
            input_tensor = input_tensor.unsqueeze(1)
        tensor_list = []
        for i in range(self.n_classes):
            # comparison yields [B,1,H,W]; cast to float
            temp_prob = (input_tensor == i).float()
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)  # -> [B,C,H,W]
        return output_tensor

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
