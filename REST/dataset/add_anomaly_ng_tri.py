
import cv2

import os

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

import json
import sys

sys.path.append('..')
from dataset.config import exp_path, mvtec_root_dir, out_mvtec_root_dir, categories

from dataset.generate_anomaly import generate_anomaly
from dataset.utils import get_new_img_size
from util.util import fix_seed

"""
10个缺陷样本作为训练集。其他缺陷样本作为测试集
图片resize到512*512 pixel.
用Memseg方式在2倍训练集ok图片的全图范围加缺陷（连通域面积大于50）
"""

print("Set dataset/config.py before running, please!")

SEED = 0
n_anomaly = 10#训练样本真缺陷
ref_size = 512
root_dir = mvtec_root_dir
out_root_dir = out_mvtec_root_dir

with open(exp_path,'r',encoding='utf-8') as f:
    path_info = json.load(f)

for single_dataset in sorted(categories):
# for single_dataset in ['BS','KolektorSDD','KolektorSDD2','magnetic_tile','cam0','cam2','cam3','LAO','YITENG']:
    # if single_dataset in ['bottle','leather','capsule']:continue
    if '.' in  single_dataset:continue
    out_dir = f'{out_root_dir}/{single_dataset}'
    with open(os.path.join(out_dir, 'origin_path.json'), 'r', encoding="utf-8") as f:
        origin_paths = json.load(f)
    fix_seed(SEED)
    input_size = None

    # if os.path.exists(out_dir):continue
    print(single_dataset)
    ok_path_list = path_info[single_dataset]['train']['ok']

    ng_path_list = path_info[single_dataset]['test']['ng'] #+ path_info[single_dataset]['train']['ng']
    ng_alpha_list = path_info[single_dataset]['test']['ng_binary'] #+ path_info[single_dataset]['train']['ng_binary']
    np.random.RandomState(SEED).shuffle(ng_path_list)
    np.random.RandomState(SEED).shuffle(ng_alpha_list)

    path_list = ng_path_list[:n_anomaly]
    dset = 'train'
    category = 'ng'

    for i, image_path in tqdm(enumerate(path_list)):
        ori_image_name = image_path.split('/')[-2] + '_' + os.path.basename(image_path).split('.', 1)[0]

        # ref_path = ref_list[i]

        origin_img = cv2.imread(os.path.join(root_dir,image_path))
        if input_size is None:
            H, W = origin_img.shape[:2]

            input_size = get_new_img_size(H, W, ref_size)
            with open(f'{out_dir}/input_size.txt', 'w') as f:
                f.write(f'{input_size[1]},{input_size[0]}')
        crop_image = cv2.resize(origin_img, (input_size[0], input_size[-1]))

        ori_pha_image = np.load(os.path.join(out_dir,dset,category,'c5',f'{dset}_{category}_{i}_{ori_image_name}.npy'))

        ori_bin_pha_image = np.zeros_like(ori_pha_image)
        ori_bin_pha_image[ori_pha_image > 0.5] = 255

        sub_folder = os.path.join(out_dir,dset,category,f'{dset}_{category}_{i}_{ori_image_name}')
        os.makedirs(os.path.join(sub_folder,'tri_label'),exist_ok=True)
        for image_id in range(20):
            # origin_ref = cv2.resize(origin_ref,(W,H),cv2.INTER_NEAREST)
            image_name = f'{dset}_{category}_{i}{image_id}_' + ori_image_name

            save_image_name = f'{image_name}.png'
            p = np.random.uniform()
            if p < 0.5:
                while True:
                    crop_image_,m = generate_anomaly(img=crop_image,target_foreground_mask=np.ones_like(ori_pha_image))
                    tmp_pha_image = np.uint8(m * 255)
                    if np.max(tmp_pha_image) < 255:continue
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp_pha_image, connectivity=8)
                    if  np.min(np.array(stats)[:,-1]) > 50:#input_size[0]*input_size[1]*0.005:
                        crop_image__ =crop_image_
                        break
                crop_pha_image = ori_pha_image.copy()
                crop_pha_image[tmp_pha_image == 255] = 1

                origin_paths[save_image_name] = image_path

            else:


                ok_index = np.random.choice(len(ok_path_list))
                origin_paths[save_image_name] = ok_path_list[ok_index]

                # image_name = f'{dset}_{category}_{i}_{image_id}_good_' + os.path.basename(ok_path_list[ok_index]).split('.', 1)[0]
                # print(ok_index)
                ok_img = cv2.imread(os.path.join(root_dir, ok_path_list[ok_index]))

                ok_image = cv2.resize(ok_img, (input_size[0], input_size[-1]))

                height, width = crop_image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
                # center = (width / 2, height / 2)  # 绕图片中心进行旋转
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(ori_bin_pha_image), connectivity=8)
                angle = np.random.uniform(-45, 45)
                scale = np.random.uniform(0.8,1.2)  # 将图像缩放为80%
                # 获得旋转矩阵
                M = cv2.getRotationMatrix2D(centroids[1], angle, scale)

                anomaly_source_img = cv2.warpAffine(src=crop_image, M=M, dsize=(width,height))
                crop_pha_image = cv2.warpAffine(src=ori_bin_pha_image, M=M, dsize=(width,height))

                mask_expanded = np.expand_dims(crop_pha_image.copy(), axis=2)
                mask_expanded[mask_expanded <= 127] = 0
                mask_expanded[mask_expanded > 127] = 1
                crop_image__ = mask_expanded * anomaly_source_img + (np.ones_like(mask_expanded) - mask_expanded) * ok_image

                # ymin,ymax,xmin,xmax = 16,240,16,240
                # crop_image = crop_image[ymin:ymax, xmin:xmax]  # 裁剪图片

                # crop_pha_image = crop_pha_image[ymin:ymax, xmin:xmax]
                # crop_fg_image = crop_fg_image[ymin:ymax, xmin:xmax]
                crop_pha_image[crop_pha_image < 127] = 0

                crop_pha_image[crop_pha_image >= 127] = 1  # 去除噪声干扰
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_pha_image, connectivity=8)
            # if dset == 'test' and category == 'ng' and np.min(np.array(stats)[:,-1]) < input_size[0]*input_size[1]*0.005 :
            #     print(image_path)
            #     continue
            # plt.subplot(1,2,1)
            # plt.imshow(crop_image)
            # plt.subplot(1,2,2)
            # plt.imshow(crop_pha_image)
            # plt.show()
            cv2.imwrite(os.path.join(
                sub_folder, save_image_name), crop_image__)
            # cv2.imwrite(os.path.join(
            #     out_dir, dset,save_median_image_name), crop_median_image)
            save_pha_image_name = f'{image_name}_pha'
            np.save(os.path.join(sub_folder,'tri_label',save_pha_image_name),crop_pha_image)

        image_name = f'{dset}_{category}_{i}_' + ori_image_name
        save_image_name = f'{image_name}.png'
        origin_paths[save_image_name] = image_path
        cv2.imwrite(os.path.join(
            sub_folder, save_image_name), crop_image)
        # cv2.imwrite(os.path.join(
        #     out_dir, dset,save_median_image_name), crop_median_image)
        save_pha_image_name = f'{image_name}_pha.npy'
        save_pha_image_name = f'{image_name}_pha'
        np.save(os.path.join(sub_folder, 'tri_label', save_pha_image_name), ori_pha_image)
    with open(os.path.join(out_dir, 'origin_path.json'), 'w', encoding="utf-8") as f:
        json.dump(origin_paths, f, indent=4, ensure_ascii=False)