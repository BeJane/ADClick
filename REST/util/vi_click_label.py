import os
from glob import glob

root_dir = '../data/defect_512/mvtec'

for category in sorted(os.listdir(root_dir)):
    adclick_path_list = [*glob( os.path.join(root_dir,category,'train','ng','c5','*.npy'))]
    print(adclick_path_list)