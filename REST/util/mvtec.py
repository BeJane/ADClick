import os
import pickle

import numpy as np

from torch.utils.data import Dataset
# texture = ['false_ng', 'cut', 'color', 'hole', 'metal_contamination', 'thread', 'glue', 'fold', 'poke', 'crack', 'glue_strip',
#            'gray_stroke', 'oil', 'rough', 'combined', 'liquid', 'scratch', 'broken_large', 'broken_small', 'contamination',
#            'bent_wire',  'cut_inner_insulation', 'cut_outer_insulation',  'poke_insulation', 'print',
#            'scratch_head', 'scratch_neck', 'thread_side', 'thread_top', 'defective','broken_teeth',
#            'fabric_border', 'fabric_interior', 'split_teeth', 'squeezed_teeth','open']
# structure = ['bent','broken','cable_swap','missing_cable','missing_wire', 'faulty_imprint',  'squeeze', 'flip', 'pill_type',
#              'manipulated_front', 'bent_lead', 'cut_lead', 'misplaced',  'damaged_case']

categories = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
structure = ['false_ng', 'cut', 'color', 'hole', 'metal_contamination', 'thread', 'glue', 'fold', 'poke', 'crack', 'glue_strip',
           'gray_stroke', 'oil', 'rough', 'combined', 'liquid', 'scratch', 'broken_large', 'broken_small', 'contamination',
           'bent_wire',  'cut_inner_insulation', 'cut_outer_insulation',  'poke_insulation', 'print',  'damaged_case',
           'scratch_head', 'scratch_neck', 'thread_side', 'thread_top', 'defective','broken_teeth', 'manipulated_front', 'bent_lead', 'cut_lead',
           'fabric_border', 'fabric_interior', 'split_teeth', 'squeezed_teeth','open''bent','broken', 'faulty_imprint',  'squeeze']
logical = ['cable_swap','missing_cable','missing_wire', 'flip', 'pill_type',
             'misplaced']
class MVTecTextDataset(Dataset):
    def __init__(self,root='data/mvtec_text',categories=categories):
        self.sens = []
        self.labels = []
        for c in categories:
            with open(os.path.join(root,f'{c}_anomaly.pkl'),'rb') as f:
                prompt_info = pickle.load(f)
                for k, v in prompt_info.items():
                    if 'embedding'  not in k:continue
                    anomaly = k.replace('_embedding','')
                    if anomaly == 'good':continue
                    if anomaly in structure:
                        self.sens.append(v)
                        self.labels.append([0]*v.shape[0])
                    if anomaly in logical:
                        self.sens.append(v)
                        self.labels.append([1]*v.shape[0])

        self.sens = np.vstack(self.sens)
        self.labels = np.hstack(self.labels)
        print(self.sens.shape,self.labels.shape)
    def __len__(self):
        return len(self.sens)

    def __getitem__(self, idx):

        return self.sens[idx],self.labels[idx]