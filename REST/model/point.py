import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def get_points(pred, gt, points, click_indx, args,pred_thresh=0.49):
    if args.point_strategy == 'max_entropy':
        return get_max_entropy_points(pred,gt,points,click_indx,pred_thresh)
    if args.point_strategy == 'wrong':
        return get_next_points(pred,gt,points,click_indx,pred_thresh)
def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :]
    # plt.subplot(1,2,1)
    # plt.imshow(gt[-1]>pred_thresh)
    # plt.subplot(1,2,2)
    # plt.imshow(pred[-1])
    # # plt.imshow(pred[-1],cmap='gray')
    # plt.show()

    fn_mask = np.logical_and(gt>pred_thresh, pred < pred_thresh,gt != -1)
    fp_mask = np.logical_and(gt<pred_thresh, pred > pred_thresh,gt != -1)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)

        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(1)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(0)

    return points
def get_max_entropy_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :]
    # plt.subplot(1,2,1)
    # plt.imshow(gt[-1])
    # plt.subplot(1,2,2)
    # plt.imshow(pred[-1])
    # plt.show()
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(pred.shape[0]):
        if np.max(pred) == 0:
            h = gt[bindx]
        else:
            h = -pred[bindx]*np.log(pred[bindx]) - (1 - pred[bindx]) * np.log((1 - pred[bindx]))
        # plt.imshow(h)
        # plt.show()
        indices = np.argwhere(h == np.max(h))
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            # print(indices)
            if gt[bindx,coords[0],coords[1]] > pred_thresh:

                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(1)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(0)

    return points


def get_first_points(pred, gt, points, click_indx=1, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :]
    # plt.subplot(1,2,1)
    # plt.imshow(gt[-1])
    # plt.subplot(1,2,2)
    # plt.imshow(pred[-1])
    # plt.show()

    fn_mask =(gt==1)# np.logical_and(gt==1, pred < pred_thresh)
    fp_mask = (gt==0)#np.logical_and(gt==0, pred > pred_thresh)

    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):

        indices = np.argwhere(fn_mask[bindx])
        if len(indices) > 0:
            num = np.random.randint(1,min(4,len(indices)))
            # print(num)
            ids = np.random.choice(np.arange(len(indices)),num)
            for i,id in enumerate(ids):
                coords = indices[id]
                # if is_positive:
                points[bindx,i, 0] = float(coords[0])
                points[bindx, i, 1] = float(coords[1])
                points[bindx, i, 2] = float(1)

        indices = np.argwhere(fp_mask[bindx])
        if len(indices) > 0:
            num = np.random.randint(1, min(4, len(indices)))
            # print(num)
            ids = np.random.choice(np.arange(len(indices)), num)
            for i, id in enumerate(ids):
                coords = indices[id]
                # if is_positive:
                points[bindx,num_points + i, 0] = float(coords[0])
                points[bindx,num_points + i, 1] = float(coords[1])
                points[bindx,num_points + i, 2] = float(0)
                # else:
                #     points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                #     points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                #     points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points
