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
mvtec_dir = '/home/szcycy/qi_data/mvtec'
out_dir = '../work_dirs/weakrest_result'
patchcore_dir = '../work_dirs/ad_sota_results/patchcore_preds'

draem_dir = '../work_dirs/ad_sota_results/DRAEM_preds'
bgad_dir = '/media/szcycy/E/adclick/work_dirs/BGAD/BGAD_mvtec_supervised_results'
reslnet_dir = '/media/szcycy/E/RealNet/experiments/MVTec-AD/realnet_vis'
sets = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]

for category in sets:
    os.makedirs(os.path.join(out_dir, category), exist_ok=True)
# for category in ['bottle']:
    for anomaly in os.listdir(os.path.join(mvtec_dir ,category,'ground_truth')):


        for idx in os.listdir(os.path.join(mvtec_dir,category,'ground_truth',anomaly)):
            idx = idx.split('_')[0]
            un_path = (f'../work_dirs/concat_abs_square_pca0.95_glo64s4pca16w4_pos0.*_l123_lr5e-5_ema_gt_50_10_swin_ws8_head32_depths4_alpha0.25_420_exp1_'
                       f'{category}/prediction/*{anomaly}_{idx}.npy')
            un_path = [*glob(un_path)][:1]

            rbbox_path = (
                f'../work_dirs/ablaRandomNN_concat_abs_square_lr5e-5_rbbox_prenoaug_pca0.95_glo64s4pca16w4_pos0.*_l123_lambdau5_gt110_50_50_10_fl_alpha0.25_ualpha0.75_221_exp1_'
                f'{category}/prediction/*{anomaly}_{idx}.npy')
            rbbox_path = [*glob(rbbox_path)][:1]
            if len(rbbox_path) == 0:continue

            un_pred = np.load(''.join(un_path))
            rbbox_pred = np.load(''.join(rbbox_path))

            gt = cv2.imread(os.path.join(mvtec_dir,category,'ground_truth',anomaly,idx+'_mask.png'),cv2.IMREAD_GRAYSCALE)
            # print(os.path.join(mvtec_dir,category,'ground_truth',anomaly,bsn+'.png'))
            img = cv2.imread(os.path.join(mvtec_dir,category,'test',anomaly,idx+'.png'))

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            patchcore_pred = np.load(os.path.join(patchcore_dir,
                                                  'mvtec_'+category,f'{category}_test_{anomaly}_{idx}.png.npy'))
            print(np.max(patchcore_pred),np.min(patchcore_pred))
            draem_pred = np.load(os.path.join(draem_dir,category,'test',anomaly,idx+'.npy'))
            realnet_pred= np.load(os.path.join(reslnet_dir,category,f'{anomaly}_{idx}.npy'))
            bgad_path =os.path.join(bgad_dir,category,anomaly,idx+'.npy')
            if not os.path.exists(bgad_path):continue

            bgad_pred = np.load(bgad_path)
            # print(np.max(bgad_pred),np.min(bgad_pred))
            fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(12,24),
                                    subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0)
            # for ax in axs:
            #     for loc_axis in ['top', 'right', 'bottom', 'left']:ax.spines[loc_axis].set_visible(False)

            axs[0].imshow(img)
            axs[1].imshow(gt,cmap='gray')
            axs[2].imshow(cv2.resize(img,(draem_pred.shape[0],draem_pred.shape[1])))
            axs[2].imshow(draem_pred,alpha=0.4,cmap='jet',vmin=0,vmax=1)

            patchcore_img = cv2.resize(img,(256,256))[16:240,16:240]

            axs[3].imshow(patchcore_img)
            axs[3].imshow(patchcore_pred, alpha=0.4,cmap='jet',vmin=0,vmax=1)

            axs[4].imshow(cv2.resize(img,(realnet_pred.shape[0],realnet_pred.shape[1])))
            axs[4].imshow(realnet_pred,alpha=0.4,cmap='jet',vmin=0,vmax=1)

            # axs[5].imshow(cv2.resize(img,(bgad_pred.shape[0],bgad_pred.shape[1])))
            # axs[5].imshow(bgad_pred,alpha=0.4,cmap='jet',vmin=0,vmax=3)

            axs[5].imshow(cv2.resize(img,(un_pred.shape[0],un_pred.shape[1])))
            axs[5].imshow(un_pred, alpha=0.4,cmap='jet',vmin=0,vmax=1)

            axs[6].imshow(cv2.resize(img,(rbbox_pred.shape[0],rbbox_pred.shape[1])))
            axs[6].imshow(rbbox_pred,alpha=0.4,cmap='jet',vmin=0,vmax=1)

               # plt.tight_layout()
            # plt.show()
            save_path = os.path.join(out_dir,category,f'draem_patchcore_realnet_un_rbbox_{anomaly}_{idx}.png')
            plt.savefig(save_path,bbox_inches='tight', pad_inches=0)
            plt.close()