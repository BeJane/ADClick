import os

import numpy as np
import pandas as pd
dir = '/home/Jingqi/AD/SimpleClickResLang/logs/iter_mask/mvtec_zero_conv_clsprompt/others/'
sets = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut',
        'metal_nut','pill', 'screw',   'toothbrush','transistor', 'zipper' ]
# sets = ['carpet','grid','leather','tile','wood' ]
# sets = ['bottle', 'cable', 'capsule','hazelnut',
        # 'metal_nut','pill', 'screw',   'toothbrush','transistor', 'zipper' ]
best_epoch=0

best_iou=0
for i in range(0,10):
    path = os.path.join(dir,f'00{i}','result_20.csv')

    result = pd.read_csv(path)
    iou_list=[]
    for set in sets:

        iou_list.append(result[result['dataset']==set]['iou'].values[0])
    mean_iou = np.mean(iou_list)
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_epoch = i
    print(f'epoch:{i},mean_iou:{mean_iou}')

print(f'best epoch:{best_epoch},best_iou:{best_iou}')
path = os.path.join(dir,f'00{best_epoch}','result_5.csv')
result = pd.read_csv(path)
f=open('result.csv','w')
f.write(f'dataset,ap,pro,pixel_auc,iou,noc80,noc85,noc90,epoch\n')

ap_list,pro_list,pixel_auc_list=[],[],[]
iou_list,noc80_list,noc85_list,noc90_list=[],[],[],[]
for set in sets:
    row = result[result['dataset']==set]
    ap = row['ap'].values[0]
    pro = row['pro'].values[0]
    pixel_auc = row['pixel_auc'].values[0]
    iou = row['iou'].values[0]
    noc80 = row['noc80'].values[0]
    noc85 = row['noc85'].values[0]
    noc90 = row['noc90'].values[0]

    ap_list.append(ap)
    pro_list.append(pro)
    pixel_auc_list.append(pixel_auc)
    iou_list.append(iou)
    noc80_list.append(noc80)
    noc85_list.append(noc85)
    noc90_list.append(noc90)

    f.write(f"{set},{ap},{pro},{pixel_auc},{iou},{noc80},{noc85},{noc90},{best_epoch+1}\n")

mean_ap = np.mean(ap_list)
mean_pro = np.mean(pro_list)
mean_pixel_auc = np.mean(pixel_auc_list)
mean_iou = np.mean(iou_list)
mean_noc80 = np.mean(noc80_list)
mean_noc85 = np.mean(noc85_list)
mean_noc90 = np.mean(noc90_list)

f.write(f"Average,{mean_ap},{mean_pro},{mean_pixel_auc},{mean_iou},{mean_noc80},{mean_noc85},{mean_noc90},{best_epoch+1}")

