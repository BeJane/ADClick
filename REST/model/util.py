import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def random_crop_feature(feature,scale=[32,64],out_size=(64,64),other_features=[]):
    b,c,h,w = feature.shape
    crop_h = random.randint(scale[0],scale[1])
    crop_w = round(crop_h/h*w)

    y_min = random.randint(0,h-crop_h)
    x_min = random.randint(0, w - crop_w)

    y_max = y_min + crop_h
    x_max = x_min + crop_w

    feature = feature[:,:,y_min:y_max,x_min:x_max]
    feature = torch.nn.functional.interpolate(feature,out_size,mode='bilinear',align_corners=False)
    # print('crop h,w',crop_h,crop_h, f'y_min={y_min},y_max={y_max}, x_min={x_min}, x_max={x_max}')
    crop_other_features = []
    for f in other_features:
        tb,tc,th,tw = f.shape
        tscale = th / h
        f = f[:, :, round(y_min * tscale):round(y_max * tscale), round(x_min * tscale):round(x_max * tscale)]

        f = torch.nn.functional.interpolate(f,(th,tw),mode='bilinear',align_corners=False)
        crop_other_features.append(f)
    return feature,crop_other_features

def prepocess_residual(x,residual_method):
    if residual_method == 'square':
        x = x ** 2
    if residual_method == 'abs':
        x = torch.abs(x)
    if residual_method == 'concat_abs_square':
        x = torch.cat((torch.abs(x), x ** 2), dim=1)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N * H * W, 1, C)
        x = torch.nn.functional.adaptive_avg_pool1d(x, C // 2).reshape(N, H, W, C // 2).permute(0, 3, 1, 2)
    return x

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

def random_masking( x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L = x.shape[0], x.shape[2] * x.shape[3]
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    # ids_keep = ids_shuffle[:, :len_keep]
    # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 1 is keep, 0 is remove
    mask = torch.zeros([N, L], device=x.device)
    mask[:, :len_keep] = 1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask
def get_iou(gt_mask, pred_mask, ignore_label=-1):

    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = torch.logical_and(torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = torch.logical_and(torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    # plt.subplot(1,2,1)
    # plt.imshow(gt_mask[0,0])
    # plt.subplot(1,2,2)
    # plt.imshow(pred_mask[0,0])
    #
    # plt.title(intersection / union)
    # plt.show()
    return intersection / union
def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int32)

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list
