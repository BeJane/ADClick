import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import Sampler
from tqdm import tqdm

from model.patchcore.common import RescaleSegmentor

from util.alpha_score import alpha_score


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def predict(loader,model,args,l=None):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image']
            features = batch['feature']
            masks = batch['mask']
            label = batch['label']

            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            pred = model(features)
            #
            pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
            if args.slide_window is not None:
                pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)

                out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                t = torch.zeros(outputsize,device='cuda')
                index = 0
                for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                        # plt.imshow(pred[0,index].cpu())
                        # plt.show()
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
            # plt.subplot(1,2,1)
            # plt.imshow(pred[0].cpu().reshape(64,64))
            # plt.subplot(1,2,2)
            # plt.imshow(masks[0,0])
            # plt.show()
            if l is not None:
                features = features.numpy()
                features = np.sum(features,axis=1)
                # print(features.shape,pred.shape)
                pred = features*l+pred*(1-l)
            preds.append(pred)
            gts.append(masks)
            image_gts.append(label)
    preds = torch.cat(preds)  # 132 1024
    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)
    anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=gts.shape[-2:],gaussian=args.gaussian
    )
    preds = torch.reshape(preds, (-1, outputsize[0], outputsize[1]))
    image_scores = torch.max(preds.view(preds.shape[0],-1), dim=-1)[0]
    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds))

    # image_gts = np.max(masks, axis=(1, 2))
    return preds, gts, image_scores, image_gts

class WeightEMA(object):
    """
    https://github.com/YU1ut/MixMatch-pytorch

    @article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
    """
    def __init__(self, model, ema_model,lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset,idx_list,batch_size_list,steps_per_epoch=100):
        super(BalancedBatchSampler, self).__init__(dataset)

        self.steps_per_epoch = steps_per_epoch
        self.generator_list = []
        for idx in idx_list:
            self.generator_list.append(self.randomGenerator(idx))
        self.batch_size_list = batch_size_list

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            for i,generator in enumerate(self.generator_list):
                # if i == 0:
                #     print(next(generator))
                for _ in range(self.batch_size_list[i]):
                    batch.append(next(generator))
            yield batch
def fix_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def cal_kill(val_scores,test_scores,val_segs,test_segs,val_anomaly_label,test_anomaly_label,strategy='img'):
    if strategy == 'img':
        bin_max = np.max(val_scores)
        bin_min = np.min(val_scores)
        bin_thresholds = np.linspace(bin_min,bin_max,100)
        undertkill_weight =3
        best_val_result = [[0,1,1]]# bin_thres,underkill,overkill
        for r in bin_thresholds:

            p1 = np.ones((val_scores.shape[0]))
            p1[val_scores < r] = 0

            ls = ((val_anomaly_label == True) * (p1 == 0)).sum() / val_scores[np.array(val_anomaly_label) == True].shape[0]
            gs = ((val_anomaly_label == False) * (p1 == 1)).sum() / val_scores[np.array(val_anomaly_label) == False].shape[0]
            if undertkill_weight * best_val_result[0][1] + best_val_result[0][2] == undertkill_weight * ls + gs:
                best_val_result.append([r,ls,gs])
            elif undertkill_weight * best_val_result[0][1] + best_val_result[0][2] > undertkill_weight * ls + gs:
                best_val_result = [[r,ls,gs]]
        best_val_result = np.array(best_val_result)
        train_underkill, train_overkill = best_val_result[0,1:]
        # thres_index = np.argmin(np.power(best_val_result[:,0]-np.median(best_val_result[:,0]),2))
        bin_thres= np.median(best_val_result[:,0])

        p1 = np.ones((test_scores.shape[0]))
        p1[test_scores < bin_thres] = 0

    else:

        bin_max = np.max(val_segs)
        bin_min = np.min(val_segs)
        bin_thresholds = np.linspace(bin_min, bin_max, 30)
        undertkill_weight = 3
        best_val_result = [[0, 0, 1, 1]]  # bin_thres,lda_thres,underkill,overkill
        for bin_thres in bin_thresholds:
            scale_list, max_area_list, area_list, max_rectangle_list = alpha_score(bin_thres, val_segs)

            confidence = np.array(locals()[f'{strategy}_list'])
            # area_list = np.array(area_list+train_area_list)
            # max_rectangle_list = np.array(max_rectangle_list+train_max_rectangle_list)

            lda_step = np.linspace(np.min(confidence), np.max(confidence), 100)
            confidence = confidence[:val_segs.shape[0]]
            for r in lda_step:
                p1 = np.ones((val_segs.shape[0]))
                p1[confidence < r] = 0

                ls = ((val_anomaly_label == True) * (p1 == 0)).sum() / \
                     val_segs[np.array(val_anomaly_label) == True].shape[0]
                gs = ((val_anomaly_label == False) * (p1 == 1)).sum() / \
                     val_segs[np.array(val_anomaly_label) == False].shape[0]
                if undertkill_weight * best_val_result[0][2] + best_val_result[0][3] == undertkill_weight * ls + gs:
                    best_val_result.append([bin_thres, r, ls, gs])
                elif undertkill_weight * best_val_result[0][2] + best_val_result[0][3] > undertkill_weight * ls + gs:
                    best_val_result = [[bin_thres, r, ls, gs]]
        best_val_result = np.array(best_val_result)
        thres_index = np.argmin(np.power(best_val_result[:, 0] - np.median(best_val_result[:, 0]), 2) +
                                np.power(best_val_result[:, 1] - np.median(best_val_result[:, 1]), 2))
        bin_thres, lda_thres,train_underkill,train_overkill = best_val_result[thres_index]
        # Test

        scale_list, max_area_list, area_list, max_rectangle_list = alpha_score(bin_thres, test_segs)

        test_scores = np.array(locals()[f'{strategy}_list'])
        # max_area_list = np.array(max_area_list)
        # area_list = np.array(area_list)
        # max_rectangle_list = np.array(max_rectangle_list)

        p1 = np.ones((test_segs.shape[0]))
        p1[test_scores < lda_thres] = 0



    ls = ((test_anomaly_label == True) * (p1 == 0)).sum() / \
         test_scores[test_anomaly_label == True].shape[0]
    gs = ((test_anomaly_label == False) * (p1 == 1)).sum() / \
         test_scores[test_anomaly_label == False].shape[0]
    # print(test_scores[test_anomaly_label == True].shape[0],test_scores[test_anomaly_label == False].shape[0])
    image_auc1 = compute_imagewise_retrieval_metrics(test_scores, test_anomaly_label)['auroc']
    image_ap1 = metrics.average_precision_score( test_anomaly_label,test_scores)

    return ls,gs,train_underkill,train_overkill,image_auc1,image_ap1