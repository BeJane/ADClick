import json
import os
import numpy as np
import pandas as pd
data_root = '/media/wjq/F/Dataset/MVTEC/mvtec_anomaly_detection'
def get_path(single_dataset,thres=0.43):

    return  f'../work_dirs/finetune_pwma_head_click_lr3e-5_ema_swin_ws8_head32_depths4_240_exp1_{single_dataset}' \
            f'/finetune_pwma_head_click_lr3e-5_ema_swin_ws8_head32_depths4_240_exp1_{single_dataset}' \
            f'_100_bilinear_512_512_c20_{thres}_metric.csv'
    # return  f'work_dirs/exp/semi_gt110_50_fl_alpha0.25_ualpha0.75_332_exp1_{single_dataset}_bilinear_metric.csv'
    # if single_dataset in ['carpet','grid','leather','tile','wood']:
    #     return  f'work_dirs/exp/gt_80_0_weight0.1_aug532_residual_slide32_8_swin_ws8_head32_depths4_alpha0.25_440_exp1_{single_dataset}_800_bilinear_metric.csv'
    # else:
    #     return f'work_dirs/exp/fg_augdefect0.01_gt_80_0_weight0.1_aug532_residual_slide32_8_swin_ws8_head32_depths4_alpha0.25_440_exp1_{single_dataset}_800_bilinear_metric.csv'
        # return f'work_dirs/exp/512fg_slide32_8_swin_ws8_head32_depths4_sample_alpha0.25_332_exp1_{single_dataset}_1000_metric.csv'
sets = ['carpet','grid','leather','tile','wood', 'bottle', 'cable','capsule', 'hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
# sets = ['grid', 'wood', 'capsule','hazelnut','pill', 'screw','transistor', 'zipper' ]
# sets = ['01','02','03' ]
# sets = ['carpet','grid','leather','tile','wood']
# sets = ['bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',   'toothbrush','transistor', 'zipper']
# sets =  ['BS','KolektorSDD','KolektorSDD2',' magnetic_tile','cam0','cam2','cam3','LAO','YITENG','USI']
# thres_list = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48]
thres_list = [0.41]
best_thres = 0
best_iou = 0
best_index = 0
for thres in thres_list:
    total_pro = []


    for single_dataset in sets:
        # if single_dataset  in ['wood','zipper','transistor']:continue
        # print(single_dataset)
        result = pd.read_csv(get_path(single_dataset,thres))

        # print(single_dataset,result['iou'])
        # print(result['pro'][0])
        total_pro.append(result['iou'])
    # print(np.array(total_pro)[:,50])
    total_pro = np.array(total_pro).mean(0)
    if np.max(total_pro) > best_iou:
        best_iou = np.max(total_pro)
        best_thres = thres
        best_index =  np.argmax(total_pro)
        print(thres,np.max(total_pro),np.argmax(total_pro))
# index = np.argmax(total_pro)
# # index = 8
f = open('../results.csv','w',encoding='utf-8')
# f.write('Dataset,AP,PRO,pixel_AUROC,image_AUROC,iteration\n')
# f.write('Dataset,AP,PRO,pixel_AUROC,image_AUROC,iteration,test_underkill,test_overkill,train_underkill,train_overkill,image_AUROC1\n')
f.write('Dataset,AP,PRO,pixel_AUROC,iou,NoC80,NoC85,NoC90,thres,iteration\n')
all_ap,all_pro,all_p_auc, all_iou = [],[],[],[]
all_noc80,all_noc85,all_noc90=[],[],[]
for single_dataset in sets:
    # if single_dataset in ['zipper', 'wood', 'pill', 'transistor', 'hazelnut', 'tile']: continue
    # if single_dataset  in ['hazelnut']:continue0.7962413219430938
    # if single_dataset not in ['carpet','grid','leather','tile','wood']:continue

    result = pd.read_csv(get_path(single_dataset,best_thres))
    row = result.loc[best_index]
    all_ap.append(row["ap"])
    all_pro.append(row["pro"])
    all_p_auc.append(row["pixel_auroc"])
    all_iou.append(row["iou"])

    all_noc80.append(row["noc80"])
    all_noc85.append(row["noc85"])
    all_noc90.append(row["noc90"])
    # print(single_dataset,result.loc[index])
    f.write(f'{single_dataset},{row["ap"]*100},{row["pro"]*100},{row["pixel_auroc"]*100},{row["iou"]*100},{row["noc80"]},{row["noc85"]},'
            f'{row["noc90"]},{best_thres},'
            f'{row["iteration"]}\n')
avg_ap = np.mean(all_ap)*100
avg_pro = np.mean(all_pro)*100
avg_p_auc = np.mean(all_p_auc) * 100
avg_iou = np.mean(all_iou) * 100

avg_noc80 = np.mean(all_noc80)
avg_noc85 = np.mean(all_noc85)
avg_noc90 = np.mean(all_noc90)
f.write(f'Average,{avg_ap},{avg_pro},{avg_p_auc},{avg_iou},{avg_noc80},{avg_noc85},{avg_noc90},{best_thres},{row["iteration"]}')