import numpy as np
import torch


def aug(feature_residual, args):
    # augment
    feature_residual = feature_residual.permute(0, 2, 3, 1)  # b,h,w,3*c

    aug_residuals = []
    b, h, w = feature_residual.shape[:3]
    n = b * h * w
    feature_residual = feature_residual.reshape(n, -1, args.feature_channel)
    assert feature_residual.shape[1] >= len(args.p), feature_residual.shape  # 近邻数量
    # 独立k次数据增强
    for i in range(0, args.k):
        # 按概率随机选择近邻
        index = np.random.choice(np.arange(feature_residual.shape[1]), n, p=args.p)
        aug_residual0 = feature_residual[np.arange(n), index]
        if args.normal_var is not None:
            aug_residual0 = aug_residual0 * shake(aug_residual0.shape[0], aug_residual0.shape[1],
                                                       args.normal_var).to('cuda', non_blocking=True)
        # np.save(os.path.join(save_dir, f'{args.dataset}_noise'), aug_residual0.numpy())
        aug_residuals.append(aug_residual0)

    feature_residual = torch.cat(aug_residuals, dim=-1).view(b, h, w, -1).permute(0, 3, 1, 2)
    del aug_residual0, aug_residuals
    return feature_residual
def shake(sample_num,feature_channel,normal_var):
    patch_index = np.random.choice(np.arange(2), sample_num, (0.4, 0.6))
    dim_p = np.random.choice(np.arange(0.2, 0.6, 0.1))
    dim_index = np.random.choice(np.arange(2), (sample_num,feature_channel), (1 - dim_p, dim_p))
    dim_index[patch_index == 0] = 0
    dim_index = torch.Tensor(dim_index)
    # print(dim_index.shape,dim_index)
    s =  torch.pow(
        torch.exp(torch.normal(0., math.sqrt(normal_var), (sample_num,feature_channel)).clamp(-0.223, 0.223)),
        dim_index)
    # print(torch.unique(s))
    return s