import math

import cv2
import numpy as np
def alpha_score(bin_thres,segmentations,threshold=5):
    scale_list, max_area_list, area_list, max_rectangle_list = [], [], [], []
    for i in range(segmentations.shape[0]):
        img = segmentations[i].copy()
        img[img <= bin_thres] = 0
        img[img > bin_thres] = 255
        img = np.uint8(img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, ltype=None)
        # 1.ng占比
        scale = (img == 255).sum() / img.size
        scale_list.append(scale)
        if scale == 1.0:
            max_area_list.append(scale)
            area_list.append(scale)
            max_rectangle_list.append(int(math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))
            continue

        if img.sum() == 0:
            area_list.append(0)
            max_area_list.append(0)
            max_rectangle_list.append(0)
        else:
            tmp = sorted(stats[:, 4], reverse=True)
            # 2.最大连通占比
            max_area_list.append(tmp[1] / img.size)
            tmp = np.array(tmp[1:])
            # 3.阈值连通之和占比
            area_list.append(tmp[tmp >= threshold].sum() / img.size)
            # 4.阈值对角线
            w_h = [math.sqrt(w ** 2 + h ** 2) for w, h in stats[1:, 2:4]]
            max_rectangle_list.append(max(w_h))
    return scale_list, max_area_list, area_list, max_rectangle_list
