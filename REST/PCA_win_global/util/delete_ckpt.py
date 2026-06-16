import os.path
from glob import glob

paths = [*glob('../work_dirs/vit_lr1e-4_ema_gt_50_8_p532_sample10_slide32_8_swin_ws8_head32_depths4_alpha0.25_420_*/*.pth')]
for p in paths:
    if '-3100.pth' in os.path.basename(p):
        continue
    print(p)
    os.remove(p)