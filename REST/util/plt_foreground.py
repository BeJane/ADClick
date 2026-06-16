import glob
import os

import matplotlib.pyplot as plt
import numpy as np

realnet_dir = '/media/szcycy/E/RealNet_foreground'
found_dir = '/media/szcycy/E/Found_foreground'
our_dir = '../work_dirs/coreset_fg_mvtec'
out = '../work_dirs/vi_fg'
data_dir = '../data/defect_512/mvtec'
for c in [ 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper']:
    os.makedirs(os.path.join(out, c), exist_ok=True)
    pathlist = [*glob.glob(data_dir+'/'+c+'/train/ok/*.png')]
    for path in pathlist:
        bsn = os.path.basename(path).split('.')[0]
        plt.figure(figsize=(6,25))
        plt.subplot(4,1,1)
        plt.imshow(plt.imread(path))
        plt.axis('off')

        plt.subplot(4,1,2)
        fg = np.load(os.path.join(found_dir,c,bsn+'.npy'))
        plt.imshow(fg[0])
        plt.axis('off')

        plt.subplot(4,1,3)
        fg = np.load(os.path.join(realnet_dir,c,bsn+'.npy'))
        plt.imshow(fg)

        plt.axis('off')

        plt.subplot(4,1,4)
        fg = np.load(os.path.join(our_dir,c,bsn+'.npy'))
        plt.imshow(fg)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out,c,bsn+'.png'))
        plt.close()