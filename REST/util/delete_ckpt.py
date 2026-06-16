import os.path
from glob import glob

paths = [*glob('/home/szcyxy/SSD/SimpleClickResLang/model_posfar/no_ok/image/mvtec_zero_conv_clsprompt_plainvit/*/checkpoints/*.pth')]
for p in paths:
    if '003' in os.path.basename(p) :
        continue
    print(p)
    os.remove(p)