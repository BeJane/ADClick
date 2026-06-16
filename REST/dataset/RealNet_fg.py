import glob
import os

import numpy as np

import cv2
def generate_target_foreground_mask(img: np.ndarray,subclass) -> np.ndarray:
    # convert RGB into GRAY scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    target_foreground_mask = img_gray / 255.0
    if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
        return np.ones_like(img_gray)
    if subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
        target_foreground_mask = 1 - target_foreground_mask

    return target_foreground_mask

outdir = '/media/szcycy/E/RealNet_foreground'
datadir = '../data/defect_512/mvtec'
for c in os.listdir(datadir):
    os.makedirs(os.path.join(outdir,c), exist_ok=True)

    pathlist = sorted([*glob.glob(os.path.join(datadir, c,'train','ok', '*.png'))])

    for path in pathlist:
        img = cv2.imread(path)
        mask = generate_target_foreground_mask(img,c)

        np.save(os.path.join(outdir,c,os.path.basename(path).split('.')[0]),mask)