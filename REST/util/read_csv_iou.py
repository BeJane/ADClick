import json
import os
import numpy as np
import pandas as pd
total_pro = []
data_root = '/media/wjq/F/Dataset/MVTEC/mvtec_anomaly_detection'

def get_path(single_dataset):

    return  f'../work_dirs/finetune_pwma_head_click_lr3e-5_ema_p532_swin_ws8_head32_depths4_240_exp1_{single_dataset}' \
            f'/finetune_pwma_head_click_lr3e-5_ema_p532_swin_ws8_head32_depths4_240_exp1_{single_dataset}' \
            f'_700_bilinear_512_512_c20_metric.csv'
sets = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
# sets = ['01','02','03']
sets = ['carpet','grid','leather','tile','wood' ]
for single_dataset in sets:
    # if single_dataset  in ['wood','zipper','transistor']:continue
    print(single_dataset)
    result = pd.read_csv(get_path(single_dataset))

    print(single_dataset,result['iou'])
    # print(result['pro'][0])
    total_pro.append(result['iou'])
# print(np.array(total_pro)[:,50])
total_pro = np.array(total_pro).mean(0)
print(np.max(total_pro),np.argmax(total_pro))
index = np.argmax(total_pro)
# index = 8
f = open('results.csv','w',encoding='utf-8')
# f.write('Dataset,AP,PRO,pixel_AUROC,image_AUROC,iteration\n')
f.write('Dataset,AP,PRO,pixel_AUROC,image_AUROC,iou,iteration\n')
# f.write('Dataset,AP,PRO,pixel_AUROC,iou,image_AUROC,iteration,test_underkill,test_overkill,train_underkill,train_overkill,image_AUROC1\n')
all_ap,all_pro,all_p_auc, all_iou = [],[],[],[]
for single_dataset in sets:
    # if single_dataset in ['zipper', 'wood', 'pill', 'transistor', 'hazelnut', 'tile']: continue
    # if single_dataset  in ['hazelnut']:continue0.7962413219430938
    # if single_dataset not in ['carpet','grid','leather','tile','wood']:continue

    result = pd.read_csv(get_path(single_dataset))
    row = result.loc[index]
    all_ap.append(row["ap"])
    all_pro.append(row["pro"])
    all_p_auc.append(row["pixel_auroc"])
    all_iou.append(row["iou"])
    # print(single_dataset,result.loc[index])
    # f.write(f'{single_dataset},{row["ap"]},{row["pro"]},{row["pixel_auroc"]},{row["image_auroc"]},{row["iteration"]}\n')
    # f.write(f'{single_dataset},{row["ap"]*100},{row["pro"]*100},{row["pixel_auroc"]*100},{row["iou"]*100},{row["image_auroc"]*100},{row["iteration"]},'
    #         f'{row["test_underkill"]*100},{row["test_overkill"]*100},{row["train_underkill"]*100},'
    #         f'{row["train_overkill"]*100},{row["image_auroc1"]*100}\n')
    #
    f.write(f'{single_dataset},{row["ap"]*100:.2f},{row["pro"]*100:.2f},{row["pixel_auroc"]*100:.2f},{row["iou"]*100:.2f},{int(row["iteration"])}\n')
#
avg_ap = np.mean(all_ap)*100
avg_pro = np.mean(all_pro)*100
avg_p_auc = np.mean(all_p_auc) * 100
avg_iou = np.mean(all_iou) * 100
f.write(f'Average,{avg_ap:.2f},{avg_pro:.2f},{avg_p_auc:.2f},{avg_iou:.2f},{int(row["iteration"])}')