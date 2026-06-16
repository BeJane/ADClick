# 参数设定
import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import torch

from torchvision.transforms import transforms


from matplotlib import pyplot as plt
mvtec_dir = '/media/wjq/F/Dataset/MVTEC/mvtec_anomaly_detection'
our_dir = '/media/wjq/F/Projects/vit-defect-detection/results/weight0.1_aug532_residual_slide32_8_swin_ws8_head32_depths4_alpha0.25_440_exp1'
patchcore_dir = '/media/wjq/F/Dataset/MVTEC/patchcore_preds'

draem_dir = '/media/wjq/F/Dataset/MVTEC/DRAEM_preds'


for category in os.listdir(our_dir):
# for category in ['bottle']:
    for anomaly in os.listdir(os.path.join(mvtec_dir ,category,'ground_truth')):
        paths = [*glob(os.path.join(our_dir,category,'test',anomaly,'*.npy'),recursive=True)]

        for p in paths:
            our_pred = np.load(p)
            bsn = os.path.basename(p).replace('.npy','')
            gt = cv2.imread(os.path.join(mvtec_dir,category,'ground_truth',anomaly,bsn+'_mask.png'),cv2.IMREAD_GRAYSCALE)
            # print(os.path.join(mvtec_dir,category,'ground_truth',anomaly,bsn+'.png'))
            img = cv2.imread(os.path.join(mvtec_dir,category,'test',anomaly,bsn+'.png'))

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            patchcore_pred = np.load(os.path.join(patchcore_dir,
                                                  'mvtec_'+category,f'{category}_test_{anomaly}_{bsn}.png.npy'))

            draem_pred = np.load(p.replace(our_dir,draem_dir))

            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(16,12),
                                    subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0)
            # for ax in axs:
            #     for loc_axis in ['top', 'right', 'bottom', 'left']:ax.spines[loc_axis].set_visible(False)

            axs[0].imshow(img)
            axs[1].imshow(gt,cmap='gray')
            axs[2].imshow(cv2.resize(img,(draem_pred.shape[0],draem_pred.shape[1])))
            axs[2].imshow(draem_pred,alpha=0.4,cmap='jet')

            patchcore_img = cv2.resize(img,(256,256))[16:240,16:240]

            axs[3].imshow(patchcore_img)
            axs[3].imshow(patchcore_pred, alpha=0.4,cmap='jet')

            axs[4].imshow(cv2.resize(img,(our_pred.shape[0],our_pred.shape[1])))
            axs[4].imshow(our_pred, alpha=0.4,cmap='jet')

            # plt.tight_layout()
            # plt.show()
            save_path = os.path.join(our_dir,category,f'draem_patchcore_{anomaly}_{bsn}.png')
            plt.savefig(save_path,bbox_inches='tight', pad_inches=0)
            plt.close()