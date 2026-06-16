import cv2 as cv
import numpy as np
import os
import glob

from matplotlib import pyplot as plt


def get_oriented_bboxs(gt):

    max_length = max(gt.shape[0:2])
    cn, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(gt, connectivity=8)
    mask = np.zeros_like(gt, dtype=np.uint8)
    for i in range(1, cn):
        y, x = np.where(cc_labels == i)
        # TODO
        if y.shape[0] <= 10:
            continue

        rect_out = cv.minAreaRect(np.stack([x, y], axis=1))
        box_out = cv.boxPoints(rect_out)
        for i in range(4):
            for j in range(2):
                box_out[i, j] = box_out[i, j] + np.random.uniform(-max_length*0.01, max_length*0.01)

        mask = cv.fillPoly(mask, box_out[None].astype(int), 100)


    mask = mask.astype(int)
    # print(np.unique(mask))
    print(np.unique(mask))
    mask[mask == 0] = -1
    mask[mask == 100] = 0
    # mask[mask == 255] = 1
    # print('pause')
    # cv.imshow('test1', mask)
    # # # cv.imshow('test2', gt)
    # cv.waitKey(0)
    return mask
def get_horizontal_bboxs(gt):
    max_length = max(gt.shape[0:2])
    cn, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(gt, connectivity=8)
    mask = -np.ones(gt.shape[:2])
    for i in range(1, cn):
        b = cc_stats[i]
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]

        x0 =round( x0  + np.random.uniform(-max_length*0.01, max_length*0.01))
        y0 =round( y0 + np.random.uniform(-max_length*0.01, max_length*0.01))
        y1 = round( y1 + np.random.uniform(-max_length*0.01, max_length*0.01))
        x1 = round( x1 + np.random.uniform(-max_length*0.01, max_length*0.01))
        mask[y0:y1, x0:x1] = 0


    # print(np.unique(mask))
    print(np.unique(mask))
    # plt.subplot(1, 2, 1)
    # plt.imshow(gt)
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask)
    # plt.show()
    return mask

if __name__ == '__main__':
    CLASS_NAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    for bbox_kind in [ 'rbbox' , 'hbbox']:
        # CLASS_NAMES = ['01','02','03']
        data_root = '../data/defect_512/mvtec'
        # out_dir = 'out_data/mvtec_anomaly_detection'
        for classname in CLASS_NAMES:
            print(classname)
            # cur_classname_output_dir = os.path.join(out_dir, classname)
            for fold in ['train','test']:
                images = glob.glob(os.path.join(data_root,classname,fold,'ng','binary','*.png'))
                out_dir = os.path.join(data_root,classname,fold,'ng',f'noise_{bbox_kind}')
                os.makedirs(out_dir,exist_ok=True)
                for i in images:
                    # print(i)
                    image = cv.imread(i, cv.CV_8U)
                    if bbox_kind == 'hbbox':
                        out_image = get_horizontal_bboxs(image)
                    elif bbox_kind == 'rbbox':
                        out_image = get_oriented_bboxs(image)
                    elif bbox_kind == 'image_level':
                        out_image =  np.zeros(image.shape[:2])


                    # print('pause')
                    # cv.imwrite(images[i], out_image)
                    np.save(os.path.join(out_dir,os.path.basename(i).replace('_pha.png','')),out_image)

