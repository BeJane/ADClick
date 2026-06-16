import sys
sys.path.append('../')
from glob import glob
from loguru import logger
from tqdm import tqdm
import cv2 as cv
import json
import numpy as np
import os
from models import DensenetPM, EfficientnetPM
import pickle
import pickle
import torch
from futils import DATASET_INFOS, read_image, transform_img, \
        ForegroundPredictor, RetrievalPredictor, COLORS


dataset_name = 'btad'
resize = 640
device = 'cuda'
model_class = DensenetPM
layer = 'features.denseblock2'
retrieval_n_clusters = 12
knn = 10

# commom
CLASS_NAMES, object_classnames, texture_classnames = DATASET_INFOS[dataset_name]
data_root = f'/home/szcycy/qi_data/btad/BTech_Dataset_transformed'
output_dir = f'/media/szcycy/E/adclick/work_dirs/{dataset_name}_retreival_foreground_{retrieval_n_clusters}_{resize}_{layer}_{layer}_{model_class.__name__}'

model = model_class(layers=[layer])
model.to(device)
model.eval()
for classname in CLASS_NAMES:
    logger.info(classname)
    cur_classname_output_dir = os.path.join(output_dir, classname)
    os.makedirs(cur_classname_output_dir, exist_ok=True)
    train_image = {}
    train_features = None
    for i, fn in enumerate(tqdm(sorted(glob(os.path.join(data_root, classname, 'train/*/*'))), desc='extract train features', leave=False)):
        k = os.path.relpath(fn, os.path.join(data_root, classname))
        image = read_image(fn, (resize, resize))
        image_t = transform_img(image)
        features_d = model.get_features(image_t[None].to(device))
        if train_features is None:
            train_features = torch.zeros(len(glob(os.path.join(data_root, classname, 'train/*/*'))), *features_d[layer].shape[1:], device=device)
        train_features[i:i+1] = features_d[layer].detach()
        train_image[k] = image
    if classname in object_classnames:
        logger.info('foreground')
        foreground_predictor = ForegroundPredictor(device)
        foreground_predictor.fit(train_features)
        # foreground_conv = torch.nn.Conv2d(foreground_predictor.lda_predictor.coef.shape[0], foreground_predictor.lda_predictor.coef.shape[1], 1).to(device)
        # foreground_conv.weight.data = foreground_predictor.lda_predictor.coef[None, :, :, None] / (foreground_predictor.normalizer.max - foreground_predictor.normalizer.min)
        # foreground_conv.bias.data = (foreground_predictor.lda_predictor.intercept - foreground_predictor.normalizer.min) / (foreground_predictor.normalizer.max - foreground_predictor.normalizer.min)
        # foreground_conv.requires_grad_(False)
        for fn in tqdm(sorted(glob(os.path.join(data_root, classname, 'train/*/*'))) + sorted(glob(os.path.join(data_root, classname, 'test/*/*'))), desc='predict data', leave=False):
            k = os.path.relpath(fn, os.path.join(data_root, classname))
            image = read_image(fn, (resize, resize))
            image_t = transform_img(image)
            features_d = model.get_features(image_t[None].to(device))
            features = features_d[layer]  # b x 512 x h x w
            lda_predict_norm = foreground_predictor.transform(features)
            lda_predict_norm = lda_predict_norm.cpu().numpy()
            # lda_predict_norm_t = foreground_conv(features)[0, 0].cpu().numpy()
            # 保存
            cur_save_dir = os.path.dirname(os.path.join(cur_classname_output_dir, k))
            os.makedirs(cur_save_dir, exist_ok=True)
            cur_image_name = os.path.basename(k).split('.', 1)[0]
            cv.imwrite(os.path.join(cur_save_dir, f'f_{cur_image_name}.png'), lda_predict_norm*255.)
            np.save(os.path.join(cur_save_dir, f'f_{cur_image_name}.npy'), lda_predict_norm)
        with open(os.path.join(cur_classname_output_dir, f'foreground_predictor.pkl'), 'wb') as f:
            pickle.dump(foreground_predictor, f)
    logger.info('retrieval')
    train_ks = list(train_image.keys())
    retrieval_predictor = RetrievalPredictor(device, n_clusters=retrieval_n_clusters)
    retrieval_predictor.fit(train_features)
    retrieval_result = {}
    logger.info('predict')
    train_cluster_l = {}
    for k, features in tqdm(zip(train_ks, train_features), total=len(train_ks), desc='retreival train data', leave=False):
        pred_l, pred, retrieval_idxs = retrieval_predictor.transform(features[None])
        retrieval_ks = [train_ks[retrieval_idx] for retrieval_idx in retrieval_idxs[1:]]  # 排除自身
        image = np.concatenate([train_image[k]] + [train_image[rk] for rk in retrieval_ks[:knn]], 1)
        image = cv.resize(image, (resize*2, resize*2//knn))
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        retrieval_result[k] = retrieval_ks
        train_cluster_l[k] = pred_l.cpu().numpy()
        # 保存
        cur_save_dir = os.path.dirname(os.path.join(cur_classname_output_dir, k))
        os.makedirs(cur_save_dir, exist_ok=True)
        cur_image_name = os.path.basename(k).split('.', 1)[0]
        cv.imwrite(os.path.join(cur_save_dir, f'r_{cur_image_name}.png'), image)
        np.save(os.path.join(cur_save_dir, f'r_{cur_image_name}.npy'), pred.cpu().numpy())
    for fn in tqdm(sorted(glob(os.path.join(data_root, classname, 'test/*/*'))), desc='retreival test data', leave=False):
        k = os.path.relpath(fn, os.path.join(data_root, classname))
        image = read_image(fn, (resize, resize))
        image_t = transform_img(image)
        features_d = model.get_features(image_t[None].to(device))
        features = features_d[layer]  # b x 512 x h x w
        pred_l, pred, retrieval_idxs = retrieval_predictor.transform(features)
        retrieval_ks = [train_ks[retrieval_idx] for retrieval_idx in retrieval_idxs]
        image = np.concatenate([image] + [train_image[rk] for rk in retrieval_ks[:knn]], 1)
        cluster_image = (COLORS[np.concatenate([cv.resize(pred_l.cpu().numpy(), (resize, resize), interpolation=cv.INTER_NEAREST)] + [cv.resize(train_cluster_l[rk], (resize, resize), interpolation=cv.INTER_NEAREST) for rk in retrieval_ks[:knn]], -1)] * 255).astype(np.uint8)
        image = np.concatenate([image, cluster_image], axis=0)
        image = cv.resize(image, (resize*2, resize*4//knn))
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        retrieval_result[k] = retrieval_ks
        # 保存
        cur_save_dir = os.path.dirname(os.path.join(cur_classname_output_dir, k))
        os.makedirs(cur_save_dir, exist_ok=True)
        cur_image_name = os.path.basename(k).split('.', 1)[0]
        cv.imwrite(os.path.join(cur_save_dir, f'r_{cur_image_name}.png'), image)
        np.save(os.path.join(cur_save_dir, f'r_{cur_image_name}.npy'), pred.cpu().numpy())
    with open(os.path.join(cur_classname_output_dir, f'retrieval_predictor.pkl'), 'wb') as f:
        pickle.dump(retrieval_predictor, f)
    with open(os.path.join(cur_classname_output_dir, r'r_result.json'), 'w') as f:
        json.dump(retrieval_result, f, indent=4)
        