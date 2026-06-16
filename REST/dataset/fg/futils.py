from collections import defaultdict
from copy import copy
from einops import rearrange
from functools import partial
from glob import glob
# from libs.perlin import rand_perlin_2d_np
from loguru import logger
# from metrics import compute_pixelwise_retrieval_metrics, compute_pro, compute_ap, compute_imagewise_retrieval_metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
# from transforms import RandomSPNoise, RandomLightness
from typing import List, Tuple, Dict, Sequence
import cv2 as cv
import imgaug.augmenters as iaa
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
# import patchcore.backbones
# import patchcore.patchcore
# import prettytable
import psutil
import random
import requests
import scipy.ndimage as ndimage
import shutil
import string
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


DATASET_INFOS = {
    'mvtec': [
        ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile','toothbrush', 'transistor', 'wood', 'zipper'], 
        ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'], 
        ['carpet', 'grid', 'leather', 'tile', 'wood']
    ],  # all, obj, texture
    'mvtec_3d': [
        ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"], 
        ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"], 
        []
    ],
    'btad': [
        ["01", "02", "03"], 
        ["01", "03"], 
        ["02"]
    ]
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
isDebug = True if sys.gettrace() else False
COLORS = np.array(
    [
        0, 0, 0,
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# https://zhuanlan.zhihu.com/p/487446562
# cv.ocl.setUseOpenCL(False)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
# cv.setNumThreads(0)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
# # 设置全局浮点精度类型为float32
# np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def save_file(log_dir):
    files = [sys.argv[0], 'futils.py', 'transforms.py']
    for f in files:
        dst_dir = os.path.join(log_dir, os.path.dirname(os.path.relpath(f)))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(os.getcwd(), f),
            os.path.join(dst_dir, os.path.basename(f)),
        )

def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # TODO: 导致网络前向变慢
    
    def seed_worker(seed, worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        if with_torch:
            torch.manual_seed(seed + worker_id)
        if with_cuda:
            torch.cuda.manual_seed(seed + worker_id)
            torch.cuda.manual_seed_all(seed + worker_id)
            torch.backends.cudnn.deterministic = True
    return partial(seed_worker, seed)

def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def get_shape(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Size]:
    """获取网络每层的输出shape

    Args:
        model (nn.Module): model
        x (torch.Tensor): 输入

    Returns:
        Dict[str, torch.Size]: _description_
    """    
    result = {}
    hook_handles = [module.register_forward_hook(partial(lambda name, module, input, output: result.__setitem__(name, output.shape), name)) for name, module in model.named_modules()]
    model(x), [hook_handle.remove() for hook_handle in hook_handles]
    return result

def modify_padding(model: nn.Module, padding_mode: str='circular'):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.padding_mode = padding_mode
    return model

def safe_zip(*args):
    assert len(set([len(i) for i in args])) == 1, 'zip check'
    return zip(*args)

def read_image(path, resize = None):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if resize:
        img = cv.resize(img, dsize=resize)
    return img

def entropy_pytorch(pk, qk, dim=0):
    return (pk * torch.log(pk / qk)).sum(dim)

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_cpu_temp():
    cpu_temp = psutil.sensors_temperatures()['coretemp'][0][1]
    return cpu_temp

def get_gpu_temp():
    # 使用所有卡最大温度值
    gpu_temp = max(list(map(float, os.popen('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader').read().splitlines())))
    return gpu_temp

def imshow_3d(ax: Axes3D, points, images, *args, iws: float=1., axis: int=1, marker='o', **kwargs):
    """
    Plot images to the 3D axes.

    Args:
        points (np.ndarray): 坐标值, nx3
        images (np.ndarray): 图片数组
        iws (np.ndarray): 间隔宽度
        axis (int, optional): 图片的面向轴. Defaults to 1. x(0), y(1), z(2)
    """
    assert isinstance(ax, Axes3D), "ax must be an instance of mpl_toolkits.mplot3d.Axes3D"
    if isinstance(images, (List, Tuple)):
        images = np.stack(images)
    if images.dtype == np.uint8:
        images = images / 255.
    if len(images.shape) == 2:
        images = images[..., None]
    if len(images.shape) == 3:
        if images.shape[-1] not in [1, 3]:
            images = images[..., None]
        else:
            images = np.repeat(images[None], len(points), axis=0)
    if not isinstance(iws, np.ndarray):
        iws = np.array([[iws]])
    if len(iws.shape) == 1:
        iws = iws[None]
    B, H, W, C = images.shape
    p0, p1 = np.meshgrid(np.arange(W), np.arange(H))
    p0, p1 = np.repeat(p0.flatten()[None], B, axis=0), np.repeat(p1.flatten()[None], B, axis=0)
    # c = images.reshape(-1, C)
    p0, p1 = (p0[None] - W/2) * iws,  -(p1[None] - H/2) * iws
    p2 = np.zeros_like(p0)
    p0, p1, p2 = p0 + points[:, 0:1], p1 + points[:, 1:2], p2 + points[:, 2:3]
    if axis == 0:
        p0, p1, p2 = p2, p0, p1
    if axis == 1:
        p0, p1, p2 = p0, p2, p1
    ax.scatter(p0.flatten(), p1.flatten(), p2.flatten(), *args, c=images.reshape(-1, C), marker=marker, **kwargs)
    
transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        if not len(self.data_source):
            return
        while True:
            yield from torch.randperm(len(self.data_source)).tolist()


class Recorder:
    def __init__(self, save_path: str, rt_save: bool = True):
        """记录结果

        Args:
            save_path (str): 保存路径
            rt_save (bool, optional): 是否实时保存. Defaults to True.
        """        
        self._records = defaultdict(dict)
        self.save_path = save_path
        self.rt_save = rt_save
    
    def add_record(self, record: dict, key):
        self._records[key].update(record)
        if self.rt_save:
            self.save()
    
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self._records, f, indent=4)
            






class SuperPointLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, exponent: int = 1) -> None:
        super().__init__()
        self.alpha = alpha
        self.exponent = exponent
    
    def forward(self, raw_pred_features, pred_features, raw_positive_points, positive_points, positive_weight, raw_negative_points, negative_points, negative_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = raw_pred_features.shape
        pos_loss, neg_loss = torch.tensor(0.), torch.tensor(0.)
        idx = torch.stack(torch.where(positive_weight > 0), 1)
        if idx.shape[0] > 0:
            raw_positive_descriptors = raw_pred_features[idx[:, 0], :, raw_positive_points[idx[:, 0], idx[:, 1], 1], raw_positive_points[idx[:, 0], idx[:, 1], 0]]  # m x D
            positive_descriptors = pred_features[idx[:, 0], :, positive_points[idx[:, 0], idx[:, 1], 1], positive_points[idx[:, 0], idx[:, 1], 0]]  # m x D
            pos_loss = pos_loss + torch.pow(torch.clamp(1.0 - (raw_positive_descriptors * positive_descriptors).sum(-1), min=0) * positive_weight[idx[:, 0], idx[:, 1]], self.exponent).sum() / max(1, len(idx))
        idx = torch.stack(torch.where(negative_weight > 0), 1)
        if idx.shape[0] > 0:
            raw_negative_descriptors = raw_pred_features[idx[:, 0], :, raw_negative_points[idx[:, 0], idx[:, 1], 1], raw_negative_points[idx[:, 0], idx[:, 1], 0]]  # m x D
            negative_descriptors = pred_features[idx[:, 0], :, negative_points[idx[:, 0], idx[:, 1], 1], negative_points[idx[:, 0], idx[:, 1], 0]]  # m x D
            neg_loss = neg_loss + torch.pow(torch.clamp((raw_negative_descriptors * negative_descriptors).sum(-1) - self.alpha, min=0) * negative_weight[idx[:, 0], idx[:, 1]], self.exponent).sum() / max(1, len(idx))
        return pos_loss, neg_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-loss)
        focal_loss = (1 - pt)**self.gamma * loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class Mahalanobis:
    def __init__(self, device):
        self.mean = None
        self.cov_inv = None
        self.device = device
        
    @torch.no_grad()
    def fit(self, data):
        B, N, C = data.shape
        self.mean = data.mean(0)  # N, C
        self.cov_inv = torch.zeros(N, C, C).to(self.device)
        for i in range(N):
            self.cov_inv[i, :, :] = torch.linalg.inv(torch.cov(data[:, i, :].T) + 1e-6 * torch.eye(C).to(self.device))

    @torch.no_grad()
    def predict(self, data):
        B, N, C = data.shape
        delta = data - self.mean[None]
        return torch.sqrt(torch.clamp(delta[:, :, None] @ self.cov_inv[None] @ delta[..., None], 0))

class FMinMaxScaler:
    def __init__(self, ratio=0.01):
        self.ratio = ratio
        self.min = None
        self.max = None
        
    def fit(self, data):
        m0 = np.partition(data, int(data.shape[0] * self.ratio), axis=0)[int(data.shape[0] * self.ratio)-1]
        m1 = np.partition(data, -int(data.shape[0] * self.ratio), axis=0)[-int(data.shape[0] * self.ratio)]
        data = data[(data>=m0) & (data<=m1)]
        self.min = data.min(0).item()
        self.max = data.max(0).item()
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    @torch.no_grad()
    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = np.clip(data, self.min, self.max)
        elif isinstance(data, torch.Tensor):
            data = torch.clamp(data, self.min, self.max)
        data = (data - self.min) / (self.max - self.min)
        return data


'''> https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/common.py#L277'''
class LastLayerToExtractReachedException(Exception):
    pass


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, stop_length: int = -1):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.stop_length = stop_length

    @torch.no_grad()
    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output.detach()
        if self.stop_length > 0 and len(self.hook_dict) >= self.stop_length:
            raise LastLayerToExtractReachedException()

  
class FeaturesCollector:
    def __init__(self, backbone: nn.Module, layers: List[str] = ['layer2'], interrupt: bool = True) -> None:
        self.backbone = backbone
        self.interrupt = interrupt
        self._features_d = {}
        self.removable_handles = []
        for layer in layers:
            forward_hook = ForwardHook(self._features_d, layer, -1 if not self.interrupt else len(layers))
            network_layer = backbone
            while "." in layer:
                extract_block, layer = layer.split(".", 1)
                network_layer = network_layer.__dict__["_modules"][extract_block]
            network_layer = network_layer.__dict__["_modules"][layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.removable_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            elif isinstance(network_layer, torch.nn.Module):
                self.removable_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
    
    @torch.no_grad()
    def __call__(self, x):
        try:
            self.backbone(x)
        except LastLayerToExtractReachedException:
            pass
        try:
            return self._features_d.copy()
        finally:
            self._features_d.clear()
    
    def __del__(self):
        for handle in self.removable_handles:
            handle.remove()


class LinearDiscriminantAnalysis_Pytorch:
    """使用pytorch推断
    """    
    def __init__(self, lda: LinearDiscriminantAnalysis, device='cuda') -> None:
        self.coef = torch.from_numpy(lda.coef_.T).to(device)
        self.intercept = torch.from_numpy(lda.intercept_).to(device)
    
    @torch.no_grad()
    def transform(self, x):
        return x @ self.coef + self.intercept


class KMeans_Pytorch:
    """使用pytorch推断
    """    
    def __init__(self, kmeans: KMeans, device='cuda') -> None:
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(device)  # n x c
    
    @torch.no_grad()
    def transform(self, x):
        # return torch.cdist(x, self.cluster_centers_)
        return torch.square((x[:, None] - self.cluster_centers[None])).sum(-1)


class ForegroundPredictor:
    gaussian_filter_sigma_ratio = 1/40
    def __init__(self, device='cuda', seed=66, n_clusters=2) -> None:
        self.device                      = device
        self.kmeans_f_num                = 50000
        self.lda_f_num                   = 15000
        self.foreground_ratio            = 0.2
        self.background_ratio            = -3
        self.n_clusters                  = n_clusters
        self.random_state                = np.random.RandomState(seed)
        self.lda                         = LinearDiscriminantAnalysis()
        self.kmeans                      = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.lda_predictor               = None
        self.normalizer                  = FMinMaxScaler()
    
    @classmethod
    def g(cls, lda_predict_norm):
        # TODO: 平滑， 之前测试是不用平滑指标更高
        # lda_predict_norm = ndimage.gaussian_filter(lda_predict_norm, sigma=lda_predict_norm.shape[0]*cls.gaussian_filter_sigma_ratio)
        # lda_predict_norm = PointMatchDataset.sharpen(lda_predict_norm)
        return lda_predict_norm

    def fit(self, foreground_features):
        # TODO: 可能存在边缘一圈为一个类别，如果类别是2，导致中心前景不在背景类别的点为空，进而导致lda有问题
        # foreground_features: b x c x h x w
        B, C, H, W = foreground_features.shape
        image_foreground_features = foreground_features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
        self.kmeans.fit(image_foreground_features.reshape(-1, foreground_features.shape[1])[self.random_state.permutation(B*H*W)[:self.kmeans_f_num]])
        labels_imgs = self.kmeans.predict(image_foreground_features.reshape(-1, foreground_features.shape[1])).reshape(B, H, W)
        if self.background_ratio < 0:
            self.background_ratio = -self.background_ratio / labels_imgs.shape[1]
        # 以ratio作为边界框，选取边界框到边界的值统计hist
        background_mask = np.zeros((B, H, W), dtype=bool)
        background_mask[:, :int(self.background_ratio * labels_imgs.shape[1]), :] = True
        background_mask[:, -int(self.background_ratio * labels_imgs.shape[1]):, :] = True
        background_mask[:, int(self.background_ratio * labels_imgs.shape[1]):-int(self.background_ratio * labels_imgs.shape[1]), :int(self.background_ratio * labels_imgs.shape[2])] = True
        background_mask[:, int(self.background_ratio * labels_imgs.shape[1]):-int(self.background_ratio * labels_imgs.shape[1]), -int(self.background_ratio * labels_imgs.shape[2]):] = True
        bidx, hidx, widx = np.where(background_mask)
        background_features = image_foreground_features[bidx, hidx, widx, :]
        background_labels = labels_imgs[bidx, hidx, widx]
        one_hot = np.zeros((background_labels.shape[0], self.kmeans.n_clusters))
        one_hot[np.arange(one_hot.shape[0]), background_labels] = 1
        hist = one_hot.sum(0)
        # background_label_u = np.arange(kmeans.n_clusters)[hist > one_hot.shape[0] / kmeans.n_clusters]
        background_label_u = [hist.argmax()]
        background_p_mask = (np.stack([background_labels == l for l in background_label_u], 1).sum(1) > 0)
        background_features = background_features[background_p_mask]
        background_label = np.zeros((background_features.shape[0]), dtype=int)
        
        # 前景
        foreground_mask = np.zeros((B, H, W), dtype=bool)
        foreground_mask[:, int(labels_imgs.shape[1] / 2 - labels_imgs.shape[1] * self.foreground_ratio):int(labels_imgs.shape[1] / 2 + labels_imgs.shape[1] * self.foreground_ratio),
                        int(labels_imgs.shape[2] / 2 - labels_imgs.shape[2] * self.foreground_ratio):int(labels_imgs.shape[2] / 2 + labels_imgs.shape[2] * self.foreground_ratio)] = True
        
        bidx, hidx, widx = np.where(foreground_mask)
        foreground_features = image_foreground_features[bidx, hidx, widx, :]
        foreground_labels = labels_imgs[bidx, hidx, widx]
        foreground_p_mask = (np.stack([foreground_labels != l for l in background_label_u], 1).sum(1) >= len(background_label_u))
        foreground_features = foreground_features[foreground_p_mask]
        foreground_label = np.ones((foreground_features.shape[0]), dtype=int)
        background_idx = self.random_state.permutation(len(background_features))[:self.lda_f_num]
        foreground_idx = self.random_state.permutation(len(foreground_features))[:self.lda_f_num]
        background_features = background_features[background_idx]
        foreground_features = foreground_features[foreground_idx]
        background_label = background_label[background_idx]
        foreground_label = foreground_label[foreground_idx]
        self.lda.fit_transform(np.concatenate([background_features, foreground_features]), np.concatenate([background_label, foreground_label]))
        self.normalizer.fit(self.lda.decision_function(image_foreground_features.reshape(-1, image_foreground_features.shape[-1])))
        self.lda_predictor = LinearDiscriminantAnalysis_Pytorch(self.lda, self.device)
    
    def transform(self, features):
        # features: 1 x c x h x w
        return self.normalizer.transform(self.lda_predictor.transform(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]))).reshape(features.shape[2], features.shape[3])
        # 使用卷积替代，cuda下会有细微值差距
        # foreground_conv = torch.nn.Conv2d(foreground_predictor.lda_predictor.coef.shape[0], foreground_predictor.lda_predictor.coef.shape[1], 1).to(device)
        # foreground_conv.weight.data = foreground_predictor.lda_predictor.coef[None, :, :, None] / (foreground_predictor.normalizer.max - foreground_predictor.normalizer.min)
        # foreground_conv.bias.data = (foreground_predictor.lda_predictor.intercept - foreground_predictor.normalizer.min) / (foreground_predictor.normalizer.max - foreground_predictor.normalizer.min)
        

class RetrievalPredictor:
    def __init__(self, device='cuda', seed=66, n_clusters=12) -> None:
        self.device           = device
        self.kmeans_f_num     = 50000
        self.row              = 5
        self.col              = 5
        self.d_method         = 'kl'
        self.l_ratio          = 4/5
        self.n_clusters       = n_clusters
        self.random_state     = np.random.RandomState(seed)
        self.lda              = LinearDiscriminantAnalysis()
        self.kmeans           = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_predictor = None
        self.train_hs         = None
        self.idx              = None
        self.s_l              = None
        self.e_l              = None
        self.eye_m            = torch.eye(self.n_clusters, device=self.device)
    
    def _transform(self, features):
        H, W = features.shape[-2:]
        pred = self.kmeans_predictor.transform(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])).reshape(H, W, -1)
        pred_l = pred.argmin(-1)
        # h = torch.stack([F.one_hot(pred_l[s[1]:e[1], s[0]:e[0]].reshape(-1), self.kmeans.n_clusters).float().mean(0) for s, e in zip(self.s_l, self.e_l)])  # row*col x k, 0.4ms
        h = self.eye_m[pred_l[self.idx[..., 0], self.idx[..., 1]]].mean(1)
        return pred_l, pred, h

    def fit(self, retrieval_features):
        B, C, H, W = retrieval_features.shape
        self.s_l = np.mgrid[:W:W//self.col, :H:H//self.row].transpose(1, 2, 0).reshape(-1, 2)  # xy
        self.e_l = np.mgrid[W//self.col:W+1:W//self.col, H//self.row:H+1:H//self.row].transpose(1, 2, 0).reshape(-1, 2)  # xy
        p_idx = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), -1)
        self.idx = torch.stack([p_idx[s[1]:e[1], s[0]:e[0]].reshape(-1, p_idx.shape[-1]) for s, e in zip(self.s_l, self.e_l)]).to(self.device)  # (row*col) x n x 2
        image_retrieval_features = retrieval_features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
        self.kmeans.fit(image_retrieval_features.reshape(-1, C)[self.random_state.permutation(B*H*W)[:self.kmeans_f_num]])
        self.kmeans_predictor = KMeans_Pytorch(self.kmeans, self.device)
        train_hs = []
        for features in retrieval_features:
            train_hs.append(
                self._transform(features[None])[-1]
            )
        self.train_hs = torch.stack(train_hs)
    
    def transform(self, features):
        # features: 1 x c x h x w
        pred_l, pred, h = self._transform(features)
        if self.d_method == 'kl':
            idx = torch.argsort(torch.sort(entropy_pytorch(
                        h[None] + 1e-8, self.train_hs + 1e-8, -1
                    ), -1)[0][:, :int(self.row*self.col*self.l_ratio)].sum(-1))
        else:
            idx = torch.argsort(torch.sort(torch.norm(
                        h[None] - self.train_hs, dim=-1
                    ), -1)[0][:, :int(self.row*self.col*self.l_ratio)].sum(-1))
        return pred_l, pred, idx

    def transform_fast(self, features, knn: int = 10):
        # features: 1 x c x h x w
        pred_l, pred, h = self._transform(features)
        if self.d_method == 'kl':
            # idx = torch.argsort(torch.topk(entropy_pytorch(h[None] + 1e-8, self.train_hs + 1e-8, -1), 
            #            k=int(self.row*self.col*self.l_ratio), dim=-1, largest=False)[0].sum(-1))
            idx = torch.topk(torch.sort(entropy_pytorch(
                        h[None] + 1e-8, self.train_hs + 1e-8, -1
                    ), -1)[0][:, :int(self.row*self.col*self.l_ratio)].sum(-1), k=knn, largest=False)[1]
        else:
            idx = torch.topk(torch.sort(torch.norm(
                        h[None] - self.train_hs, dim=-1
                    ), -1)[0][:, :int(self.row*self.col*self.l_ratio)].sum(-1), k=knn, largest=False)[1]
        return pred_l, pred, idx


