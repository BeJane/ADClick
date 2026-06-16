import os

mvtec_root_dir = '/home/Jingqi/AD/Data/mvtec'
texture_source_dir = '/home/Jingqi/AD/Data/dtd/images'
out_mvtec_root_dir = '/home/Jingqi/AD/Data/defect_1024/mvtec'
exp_path = os.path.join(os.path.dirname(__file__), "mvtec_index.json")
all_sets = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
texture_sets = ['carpet','grid','leather','tile','wood']
object_sets = ['bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]

categories = object_sets