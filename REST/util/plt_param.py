import json
import os
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt, cm
import seaborn as sns
data_root = '/media/wjq/F/Dataset/MVTEC/mvtec_anomaly_detection'

def get_path(single_dataset,global_nn,fg_e):

    path =f'../work_dirs/concat_abs_square_pca0.95_glo64s4pca16w4_pos0.15_l23_lr5e-5_ema_gt_50_10_swin_ws8_head32_depths4_alpha0.25_420_exp1_{single_dataset}' \
            f'/concat_abs_square_pca0.95_glo64s4pca16w4_pos0.15_l23_lr5e-5_ema_gt_50_10_swin_ws8_head32_depths4_alpha0.25_420_exp1_{single_dataset}' \
            f'_4500_bilinear_512_512_fg_{global_nn}_{fg_e}_metric.csv'
    if single_dataset == '03':
        return path.replace('512_512','384_512')
    return path
def get_rbbox_path(single_dataset,global_nn,fg_e):
    path = f'../work_dirs/concat_abs_square_pca0.95_glo64s4pca16w4_pos0.15_l23_lr5e-5_ema_gt_50_10_swin_ws8_head32_depths4_alpha0.25_420_exp1_{single_dataset}' \
           f'/concat_abs_square_pca0.95_glo64s4pca16w4_pos0.15_l23_lr5e-5_ema_gt_50_10_swin_ws8_head32_depths4_alpha0.25_420_exp1_{single_dataset}' \
           f'_4500_bilinear_512_512_fg_{global_nn}_{fg_e}_metric.csv'
    if single_dataset == '03':
        return path.replace('512_512', '384_512')
    return path
sets = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
# sets = object_sets
sets =[ 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper']

f = open('inference_un.csv','w',encoding='utf-8')
f.write('global nn,fg_e,Dataset,AP,PRO,pixel_AUROC,image_AUROC,iteration\n')

rf = open('inference_rbbox.csv','w',encoding='utf-8')
rf.write('global nn,fg_e,Dataset,AP,PRO,pixel_AUROC,image_AUROC,iteration\n')

global_nn_list = [16,32,48,64]
fg_e_list = [0.05,0.1,0.15,0.2,0.25]
avg_ap_list = []
rbbox_avg_ap_list = []
for global_nn in global_nn_list:
    for fg_e in fg_e_list:
        total_pro = []
        rbbox_total_pro = []
        for single_dataset in sets:

            result = pd.read_csv(get_path(single_dataset,global_nn,fg_e))
            rbbox_result = pd.read_csv(get_rbbox_path(single_dataset,global_nn,fg_e))

            total_pro.append(result['pro']+result['ap'])
            rbbox_total_pro.append(rbbox_result['pro']+rbbox_result['ap'])
        # print(np.array(total_pro)[:,50])
        total_pro = np.array(total_pro).mean(0)
        rbbox_total_pro = np.array(rbbox_total_pro).mean(0)
        index = np.argmax(total_pro)
        rrbox_index = np.argmax(rbbox_total_pro)


        all_ap,all_pro,all_p_auc, all_i_auc = [],[],[],[]
        rbbox_all_ap,rbbox_all_pro,rbbox_all_p_auc,rbbox_all_i_auc = [],[],[],[]
        for single_dataset in sets:

            result = pd.read_csv(get_path(single_dataset,global_nn,fg_e))
            rbbox_result = pd.read_csv(get_rbbox_path(single_dataset,global_nn,fg_e))
            row = result.loc[index]
            all_ap.append(row["ap"])
            all_pro.append(row["pro"])
            all_p_auc.append(row["pixel_auroc"])
            all_i_auc.append(row["image_auroc"])

            f.write(f'{global_nn},{fg_e},{single_dataset},{row["ap"]*100:.2f},{row["pro"]*100:.2f},{row["pixel_auroc"]*100:.2f},{row["image_auroc"]*100:.2f},{int(row["iteration"])}\n')
        #
            rbbox_row = rbbox_result.loc[index]
            rbbox_all_ap.append(rbbox_row['ap'])
            rbbox_all_pro.append(rbbox_row['pro'])
            rbbox_all_p_auc.append(rbbox_row['pixel_auroc'])
            rbbox_all_i_auc.append(rbbox_row['image_auroc'])

            rf.write(f'{global_nn},{fg_e},{single_dataset},{rbbox_row["ap"]*100:.2f},{rbbox_row["pro"]*100:.2f},{rbbox_row["pixel_auroc"]*100:.2f},{rbbox_row["image_auroc"]*100:.2f},{int(rbbox_row["iteration"])}\n')
        avg_ap = np.mean(all_ap)*100
        avg_pro = np.mean(all_pro)*100
        avg_p_auc = np.mean(all_p_auc) * 100
        avg_i_auc = np.mean(all_i_auc) * 100

        rbbox_avg_ap = np.mean(rbbox_all_ap)*100
        rbbox_avg_pro = np.mean(rbbox_all_pro)*100
        rbbox_avg_p_auc = np.mean(rbbox_all_p_auc) * 100
        rbbox_avg_i_auc = np.mean(rbbox_all_i_auc) * 100


        f.write(f'{global_nn},{fg_e},Average,{avg_ap:.2f},{avg_pro:.2f},{avg_p_auc:.2f},{avg_i_auc:.2f},{int(row["iteration"])}\n\n')
        rf.write(f'{global_nn},{fg_e},Average,{rbbox_avg_ap:.2f},{rbbox_avg_pro:>2f},{rbbox_avg_p_auc:.2f},{rbbox_avg_i_auc:.2f},{int(rbbox_row["iteration"])}\n\n')
        avg_ap_list.append(avg_ap)
        rbbox_avg_ap_list.append(rbbox_avg_ap)

# Create a DataFrame for ap_matrix and rbbox_matrix
ap_matrix = pd.DataFrame(np.reshape(np.array(avg_ap_list), (5,5)), index=global_nn_list, columns=fg_e_list)
rbbox_ap_matrix = pd.DataFrame(np.reshape(np.array(rbbox_avg_ap_list), (5,5)), index=global_nn_list, columns=fg_e_list)

# Set up the subplot grid
fig, axes = plt.subplots(1, 2, figsize=(8,4))

cmap =  sns.diverging_palette(230,0,90,60, as_cmap=True)
# Create the first heatmap for AP
sns.heatmap(ap_matrix, annot=True, fmt=".1f", annot_kws={"size": 18,"color":"white"}, ax=axes[0],cbar=False,cmap=cmap)
axes[0].set_title('Unsupervised (AP)', fontsize=18)
axes[0].set_ylabel('K', fontsize=18)
axes[0].set_xlabel('Foreground Confidence', fontsize=18)
axes[0].tick_params(axis='both', labelsize=18)  # Adjust tick label size
# Create the second heatmap for RBBox
sns.heatmap(rbbox_ap_matrix, annot=True, fmt=".1f", annot_kws={"size": 18,"color":"white"}, ax=axes[1],cbar=False,cmap=cmap)
axes[1].set_title('RBBox (AP)', fontsize=18)
axes[1].set_ylabel('K', fontsize=18)
axes[1].set_xlabel('Foreground Confidence', fontsize=18)
axes[1].tick_params(axis='both', labelsize=18)  # Adjust tick label size
# Adjust layout
plt.tight_layout()
# Show plot
plt.savefig('inference_params.pdf',bbox_inches='tight')