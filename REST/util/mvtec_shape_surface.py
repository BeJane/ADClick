
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

categories = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
surface = ['false_ng',  'color','faulty_imprint',   'glue',   'glue_strip','metal_contamination',
           'gray_stroke', 'oil', 'rough', 'combined', 'liquid', 'contamination', 'print', 'pill_type',
         ]
shape = ['bent','cut','broken','hole',  'thread','fold','poke', 'crack', 'scratch', 'broken_large', 'broken_small',
         'scratch_head', 'scratch_neck', 'thread_side', 'thread_top', 'defective','broken_teeth', 'fabric_border',
         'fabric_interior', 'split_teeth', 'squeezed_teeth',
           'squeeze','bent_wire',  'cut_inner_insulation', 'cut_outer_insulation',  'poke_insulation',
             'manipulated_front', 'bent_lead', 'cut_lead', 'damaged_case']
ignore = ['cable_swap','missing_cable','missing_wire','flip', 'misplaced'  ]
class MVTecSurfaceShapeTextDataset(Dataset):
    def __init__(self,root='data/mvtec_text'):
        self.sens = []
        self.labels = []
        for c in categories:
            with open(os.path.join(root,f'{c}_anomaly.pkl'),'rb') as f:
                prompt_info = pickle.load(f)
                for k, v in prompt_info.items():
                    if 'embedding'  not in k:continue
                    anomaly = k.replace('_embedding','')
                    if anomaly == 'good':continue
                    if anomaly in surface:
                        self.sens.append(v)
                        self.labels.append([0]*v.shape[0])
                    elif anomaly in shape:
                        self.sens.append(v)
                        self.labels.append([1]*v.shape[0])

        self.sens = np.vstack(self.sens)
        self.labels = np.hstack(self.labels)
        print(self.sens.shape,self.labels.shape)
    def __len__(self):
        return len(self.sens)

    def __getitem__(self, idx):

        return self.sens[idx],self.labels[idx]