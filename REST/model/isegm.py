import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=True):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:

                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            # print(self.norm_radius)
            coords[:,0] = (coords[:,0] <= (self.norm_radius*2 * self.spatial_scale) ** 2).float()
            coords[:, 1] = (coords[:, 1] <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        # plt.subplot(1,2,1)
        # plt.imshow(coords[-1,0].cpu())
        # plt.subplot(1, 2, 2)
        # plt.imshow(coords[-1, 1].cpu())
        # plt.show()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])

def get_object_roi(pred_mask, clicks_list, expansion_ratio, min_crop_size):
    pred_mask = pred_mask.copy()
    # plt.imshow(pred_mask)
    # plt.show()
    assert clicks_list.shape[0] == 1
    for click in clicks_list[0]:
        if click[-1] == 1:
            pred_mask[int(click[0]), int(click[1])] = 1
    # plt.imshow(pred_mask)
    # plt.show()
    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox
def check_object_roi(object_roi, clicks_list):
    for click in clicks_list[0]:
        if click[-1] == 1:
            if click[0] < object_roi[0] or click[0] >= object_roi[1]:
                return False
            if click[1] < object_roi[2] or click[1] >= object_roi[3]:
                return False

    return True
def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union
def get_ori_pred(pred,points,pred_thres,_object_roi,recompute_thresh_iou=0.5):
    # plt.subplot(1,2,1)
    # plt.imshow(pred.cpu()[0,0])

    current_pred_mask = (pred > pred_thres)[0, 0].cpu().numpy()
    if current_pred_mask.sum() > 0:
        current_object_roi = get_object_roi(current_pred_mask, points,
                                            1.4, 200)

    else:
        current_object_roi = 0, pred.shape[2] - 1, 0, pred.shape[3] - 1

    update_object_roi = False
    if _object_roi is None:
        update_object_roi = True
    elif not check_object_roi(_object_roi, points):
        update_object_roi = True
    elif get_bbox_iou(current_object_roi, _object_roi) < recompute_thresh_iou:
        update_object_roi = True

    if update_object_roi:
        _object_roi = current_object_roi
    if _object_roi is not None:
        rmin, rmax, cmin, cmax = _object_roi
        pred_ori = torch.zeros_like(pred)
        pred_ori[:, :, rmin:rmax + 1, cmin:cmax + 1] = pred[:, :, rmin:rmax + 1, cmin:cmax + 1]
        pred = pred_ori
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(pred.cpu()[0, 0])
    # plt.show()
    return pred,_object_roi
def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))
