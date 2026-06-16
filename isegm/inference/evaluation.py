import os
import shutil
from time import time

import cv2
import matplotlib.pyplot as plt
from sklearn  import  metrics
import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker, Click
from isegm.inference.metrics import compute_pixelwise_retrieval_metrics, compute_imagewise_retrieval_metrics, \
    compute_pro_original_mvtec
from isegm.inference.utils import compute_noc_metric
from isegm.utils.vis import draw_probmap

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    eval_times = 1
    avg_ap,avg_pro,avg_pixel_auc,avg_iou=0,0,0,0
    avg_noc80,avg_noc85,avg_noc90 = 0,0,0
    for t in range(eval_times):
        all_ious = []
        best_ious = []
        start_time = time()
        preds,gts = [],[]
        for index in tqdm(range(len(dataset)), leave=False):
            sample,info  = dataset.get_sample(index)
            path=info['path']
            prompt = info['prompt']
            for object_id in sample.objects_ids:
                _, sample_ious, pred,best_iou = evaluate_sample(sample.image,sample.residual, prompt,
                                                                sample.gt_mask(object_id), predictor,
                                                    sample_id=index, **kwargs)
                all_ious.append(sample_ious)
                preds.append(pred)
                gts.append(sample.gt_mask(object_id))
                best_ious.append(best_iou)

                dir_list = path.split('/')
                save_dir = os.path.join(f'simpleclick_vi_results',*dir_list[-5:-1])
                os.makedirs(save_dir,exist_ok=True)
                # plt.subplot(1,2,1)
                bsn = os.path.basename(path)
                shutil.copy(path,os.path.join(save_dir,bsn))
                prob_map = draw_probmap(pred)
                pred_path = os.path.join(save_dir,bsn.replace('.png',f'_c{kwargs["max_clicks"]}.png'))
                # print(pred_path)
                cv2.imwrite(pred_path, prob_map[:,:,::-1])
                np.save(pred_path.replace('.png',f'.npy'),pred)
                #
                # plt.imshow(pred>0.49)
                # plt.title(best_iou)
                # plt.subplot(1,2,2)
                # plt.imshow(sample.gt_mask(object_id))
                # plt.title('GT')
                # plt.show()
        end_time = time()
        elapsed_time = end_time - start_time
        preds = np.array(preds)
        gts = np.array(gts)
        best_ious = np.array(best_ious).mean()
        # print(gts.shape)
        ap =  metrics.average_precision_score(gts.ravel().astype(int),preds.ravel())

        pixel_auc = compute_pixelwise_retrieval_metrics(preds, gts)["auroc"]
        # image_auc = compute_imagewise_retrieval_metrics(test_scores, test_anomaly_label)['auroc']

        pro = compute_pro_original_mvtec(gts, preds)

        iou_thrs = np.arange(0.8, 0.95 + 0.001, 0.05).tolist()
        noc_list, noc_list_std, over_max_list = compute_noc_metric(all_ious, iou_thrs=iou_thrs)
        noc80,noc85,noc90 = noc_list[0],noc_list[1],noc_list[2]
        print(ap,pro,pixel_auc,"iou=",best_ious,"noc80-98-90",noc80,noc85,noc90)
        avg_ap = avg_ap + ap/eval_times
        avg_pro = avg_pro + pro/eval_times
        avg_pixel_auc = avg_pixel_auc + pixel_auc/eval_times
        avg_iou = avg_iou + best_ious/eval_times

        avg_noc80 = avg_noc80 + noc80/eval_times
        avg_noc85 = avg_noc85 + noc85/eval_times
        avg_noc90 = avg_noc90 + noc90/eval_times
        # print(np.array(all_ious).shape)
    return (all_ious, elapsed_time),(avg_ap,avg_pro,avg_pixel_auc,avg_iou,avg_noc80,avg_noc85,avg_noc90)

def evaluate_dataset_ad(dataset, predictor, **kwargs):
    eval_times = 1
    avg_ap,avg_pro,avg_pixel_auc,avg_image_auc=0,0,0,0
    for t in range(eval_times):
        # all_ious = []
        # best_ious = []
        start_time = time()
        preds,gts = [],[]
        for index in tqdm(range(len(dataset)), leave=False):
            sample,info  = dataset.get_sample(index)

            path=info['path']
            prompt = info['prompt']
            for object_id in sample.objects_ids:
                pred = evaluate_sample(sample.image,sample.residual,prompt, sample.gt_mask(object_id), predictor,
                                                    sample_id=index, **kwargs)
                # all_ious.append(sample_ious)
                preds.append(pred)
                gts.append(sample.gt_mask(object_id))

                #
                # dir_list = path.split('/')
                # save_dir = os.path.join(f'simpleclick_vi_results',*dir_list[-5:-1])
                # os.makedirs(save_dir,exist_ok=True)
                # # plt.subplot(1,2,1)
                # bsn = os.path.basename(path)
                # shutil.copy(path,os.path.join(save_dir,bsn))
                # prob_map = draw_probmap(pred)
                # pred_path = os.path.join(save_dir,bsn.replace('.png',f'_c{kwargs["max_clicks"]}.png'))
                # # print(pred_path)
                # cv2.imwrite(pred_path, prob_map[:,:,::-1])
                # np.save(pred_path.replace('.png',f'.npy'),pred)
                #
                # plt.imshow(pred>0.49)
                # plt.title(best_iou)
                # plt.subplot(1,2,2)
                # plt.imshow(sample.gt_mask(object_id))
                # plt.title('GT')
                # plt.show()
        end_time = time()
        elapsed_time = end_time - start_time
        preds = np.array(preds)
        gts = np.array(gts)
        # best_ious = np.array(best_ious).mean()
        # print(gts.shape)
        ap =  metrics.average_precision_score(gts.ravel().astype(int),preds.ravel())

        pixel_auc = compute_pixelwise_retrieval_metrics(preds, gts)["auroc"]
        test_scores = np.max(preds.reshape(preds.shape[0],-1),axis=1)
        image_gts = np.max(gts.reshape(gts.shape[0],-1),axis=1)
        print(test_scores.shape,image_gts.shape)
        image_auc = compute_imagewise_retrieval_metrics(test_scores, image_gts)['auroc']

        pro = compute_pro_original_mvtec(gts, preds)


        print(ap,pro,pixel_auc,image_auc)
        avg_ap = avg_ap + ap/eval_times
        avg_pro = avg_pro + pro/eval_times
        avg_pixel_auc = avg_pixel_auc + pixel_auc/eval_times
        avg_image_auc = avg_image_auc + image_auc/eval_times

        # print(np.array(all_ious).shape)
    return (avg_ap,avg_pro,avg_pixel_auc,avg_image_auc)

def evaluate_sample(image,residual, prompt,gt_mask, predictor, max_iou_thr ,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    # print(gt_mask.shape)
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    # pred_mask = pre_mask
    # plt.imshow(pred_mask > 0.49)
    # plt.show()
    # print(pred_mask.shape)

    ious_list = []
    # print(pred_thr)
    with torch.no_grad():

        if max_clicks >0:

            predictor.set_input_image(image,residual,prompt)
            best_iou=0

            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker)

                pred_mask = pred_probs > pred_thr

                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

                iou = utils.get_iou(gt_mask, pred_mask)

                # print(iou,gt_mask.sum())
                ious_list.append(iou)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_probs
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break


            return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs,iou
        else:
            pred_list = []
            for p in prompt:
                predictor.set_input_image(image, residual, p)
                pred_probs = predictor.get_prediction(clicker)
                pred_list.append(pred_probs[None,:,:])
            pred_probs = np.concatenate(pred_list)
            # print(pred_probs.shape)
            pred_probs =np.max(pred_probs,axis=0)
            # print(pred_probs.shape)
            return pred_probs