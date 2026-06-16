import argparse
import random

import numpy as np
import torch

from model.detector_two_head import DetectorTwoHead

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', default=(3,512,512))
    parser.add_argument('--feature-channel', type=int, default=1024)
    parser.add_argument('--slide_window', type=int, default=32)
    parser.add_argument('--slide_stride', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=32)
    parser.add_argument('--depths', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=8)
    parser.add_argument('--norm-radius', type=int, default=5)

    parser.add_argument('--max-num-next-clicks', type=int, default=5)
    parser.add_argument('--gt-thres1', type=float, default=0.5)
    parser.add_argument('--gt-thres2', type=float, default=0.08)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--focal-loss-alpha', type=float, default=0.25)
    parser.add_argument('--focal-loss-gamma', type=float, default=4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=None)
    
    args = parser.parse_args()
    return args

def get_click_pred(model,feature_residual,pre_pred,points):
    coord_features = model.get_coord_features(pre_pred, points)
    with torch.no_grad():
        pred = model.vit((feature_residual, coord_features), only_ann=True)
    pred = torch.sigmoid(pred)
    # pred = torch.softmax(pred, dim=2)[:, :, 1]  # 100,256,1
    if args.slide_window is not None:
        pred = pred.reshape(pre_pred.shape[0], -1, args.slide_window, args.slide_window)
        out = torch.zeros((pre_pred.shape[0], *model.feature_size), device='cuda')
        t = torch.zeros(model.feature_size, device='cuda')
        index = 0
        for i in range(0, model.feature_size[0] - args.slide_window + 1, args.slide_stride):
            for j in range(0, model.feature_size[1] - args.slide_window + 1, args.slide_stride):
                out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                t[i:i + args.slide_window, j:j + args.slide_window] += 1
                index += 1
        pred = out / t
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(pre_pred.shape[2], pre_pred.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

    return pred,(pred > 0.49).float()
if __name__ == '__main__':
    # example
    args = get_args()
    model = DetectorTwoHead(args.image_size, stride=8, patch_size=(1, 1),
                            slide_window=args.slide_window, slide_stride=args.slide_stride,
                            in_chans=args.feature_channel, norm_radius=args.norm_radius,
                            num_classes=1, embed_dim=1024, window_size=args.window_size, depths=[args.depths],
                            num_heads=[args.num_heads])
    model.vit.load_state_dict(torch.load(f'../work_dirs/zs_ann_slide_sample4_lr3e-5_ema_gt_50_8_p532_swin_ws8_head32_depths4_alpha0.25_112_exp1_bottle/iter-6600.pth', map_location='cpu'))
    model.cuda()
    # 模拟生成residual
    residual = np.load('../data/defect_512/mvtec/bottle/train/ng/global50_residual_l123/train_ng_0_broken_small_002.npy')
    residual = torch.tensor(residual).unsqueeze(0).cuda()
    slide_residual = model.slide(residual, args)
    num_points = 24
    points = -torch.ones((1, num_points*2, 3)).cuda()
    pred = torch.zeros((1,1,*args.image_size[1:])).cuda()
    pred,bin_pred = get_click_pred(model,slide_residual,pred,points)

    for click_indx in range(1,args.max_num_next_clicks+1):
        # 模拟click
        x = random.randint(0,511)
        y = random.randint(0,511)
        label = random.randint(0,1)
        
        if label == 1:#缺陷
            points[0, num_points - click_indx, 0] = float(y)
            points[0, num_points - click_indx, 1] = float(x)
            points[0, num_points - click_indx, 2] = float(click_indx)
        else:
            points[0, 2 * num_points - click_indx, 0] = float(y)
            points[0, 2 * num_points - click_indx, 1] = float(x)
            points[0, 2 * num_points - click_indx, 2] = float(click_indx)
        print(click_indx)
        pred,bin_pred = get_click_pred(model, slide_residual, pred, points)