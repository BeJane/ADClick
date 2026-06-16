import os.path
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
noise_scale = 1e-3
res_dir = '../vi_residual'
category = 'toothbrush'
res = np.load(os.path.join(res_dir,f'{category}.npy'))
res =res.reshape(-1,1024)
# noise_res = np.load(os.path.join(res_dir,f'{category}_noise.npy'))
# noise_res = noise_res.reshape(8,64,64,1024)
# norm = np.mean(np.abs(res.reshape(-1,1024)),axis=1)
# print(norm.shape)
# hist,d = np.histogram(norm,100)
# print(d)
# plt.hist(norm,d)
# plt.title(f'Residual average L1 norm of {category}')
# plt.show()
# plt.savefig(f'Residual average L1 norm of {category}.png')
# res = torch.Tensor(res)
# index = np.random.choice(np.arange(res.shape[0]),512)
# n_r = res+torch.randn_like(res).mul_(1e-5 * random.random())
# plt.figure(figsize=(100,100))
# plt.imshow(torch.cat([res[index],n_r[index]]))
# plt.savefig(f'{res_dir}/{category}.png')
# plt.imshow(torch.cat([res[0].reshape(32,32),n_r[0].reshape(32,32)]))
# print(np.max(res[:512]))
# print(np.min(res[:512]))
# plt.show()
fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12,4),
                        subplot_kw={'xticks': [], 'yticks': []})

for i in range(6):
    r = res[i].reshape(32,32)
    r = torch.Tensor(r)
    noise_scale = torch.mean(torch.abs(r))
    n_r = r*torch.exp(torch.randn_like(r))
    axs[i].imshow(torch.cat([r,n_r]))
    # axs[i,0].set_title(f'L1 norm = {torch.mean(torch.abs(r)) }')
    # print(i*10,torch.mean(torch.abs(r)))
    # axs[i,1].imshow(r+torch.randn_like(r).mul_(noise_scale * random.random()))
    # axs[i].set_title(f'({i*10}, {i*10})')
plt.show()
# plt.savefig(f'{res_dir}/{category}.png')