import os.path
import random
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import Sampler
from tqdm import tqdm

from model.images import IMAGENET_MEAN, IMAGENET_STD
from model.patchcore.common import RescaleSegmentor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def predict_with_fg(loader,model, path_info, knn_info, args,l=None):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts = [], [], []
    fgs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image']
            features = batch['feature']
            masks = batch['mask']
            label = batch['label']
            filenames = batch['filename']
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            start_time = time.time()
            pred = model(features)
            #
            if args.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
            if args.slide_window is not None:
                pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                t = torch.zeros(outputsize,device='cuda')
                index = 0
                for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
            # print(time.time()-start_time,features.shape)
            #     plt.subplot(1,3,1)
            #     plt.imshow(masks[0,0])
            #     plt.subplot(1,3,2)
            #     plt.imshow(out[0])
            #
            #     plt.subplot(1,3,3)
            #     plt.imshow(time)
            #     plt.show()
            if l is not None:
                features = features.numpy()
                features = np.sum(features,axis=1)
                # print(features.shape,pred.shape)
                pred = features*l+pred*(1-l)
            for i, filename in enumerate(filenames):
                origin_path = path_info[os.path.basename(filename)]
                basename = os.path.basename(origin_path)
                fg = np.load(os.path.join(args.fg_dir,origin_path.replace(basename,f'f_{basename.split(".")[0]+".npy"}')))[None,:,:]
                # print(query_fg.shape)
                knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                for p in knn_list[:args.fg_knn]:
                    basename = os.path.basename(p)
                    ref_fg = np.load(os.path.join(args.fg_dir,args.dataset,p.replace(basename,f'f_{basename.split(".")[0]+".npy"}')))
                    fg = np.concatenate([fg,ref_fg[None,:,:]])
                # print(fg.shape)
                fg = np.max(fg,axis=0)
                # plt.subplot(1,4,1)
                # plt.title('gt')
                # plt.imshow(masks[0,0])
                # plt.subplot(1,4,2)
                # plt.title('output')
                # plt.imshow(pred[0].cpu())
                # plt.subplot(1,4,3)
                # plt.title(f'knn fg(k={args.fg_knn})')
                # t = np.concatenate([pred.cpu().numpy(),cv2.resize(fg,(64,64))[None,:,:]])
                # t =np.min(t,axis=0)
                # plt.imshow(t)
                # plt.subplot(1,4,4)
                # plt.imshow(fg)
                # plt.show()
                fg = cv2.resize(fg,(masks.shape[-1],masks.shape[-2]))
                # fg = cv2.blur(fg,(20,20))
                # plt.imshow(fg)
                # plt.show()
                # pred[i] = pred[i] * torch.tensor(fg).cuda()
                fgs.append(torch.tensor(fg)[None,:,:])
            preds.append(pred)
            gts.append(masks)
            image_gts.append(label)
    preds = torch.cat(preds)  # 132 1024
    fgs = torch.cat(fgs)

    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)
    anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=gts.shape[-2:],gaussian=args.gaussian
    )

    preds = torch.reshape(preds, (-1, outputsize[0], outputsize[1]))
    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds))
    # print(fgs.shape,preds.shape)
    #
    preds = fgs**0.2 * preds
    # preds = (preds - torch.min(preds))/(torch.max(preds) - torch.min(preds))
    # plt.imshow(preds[0])
    # plt.show()
    preds = preds.cuda()

    image_scores = torch.max(preds.view(preds.shape[0],-1), dim=-1)[0].cuda()

    # image_gts = np.max(masks, axis=(1, 2))
    return preds, gts, image_scores, image_gts

def predict(loader,model,args,l=None):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts = [], [], []
    image_scores =[]
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image']
            features = batch['feature']
            masks = batch['mask']
            label = batch['label']

            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            start_time = time.time()
            pred = model(features)
            if args.image_cls:
                pred,cls_pred = pred
                cls_pred = torch.softmax(cls_pred, dim=1)[:, 1]
                image_scores.append(cls_pred)
            #
            if args.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
            if args.slide_window is not None:
                pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                t = torch.zeros(outputsize,device='cuda')
                index = 0
                for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
            # print(time.time()-start_time,features.shape)
            #     plt.subplot(1,3,1)
            #     plt.imshow(masks.cpu()[0,0])
            #     plt.subplot(1,3,2)
            #     plt.imshow(out.cpu()[0])
            #
            #     # plt.subplot(1,3,3)
            #     # plt.imshow(time)
            #     plt.show()
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


    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds)).cuda()
    if args.image_cls:
        image_scores = torch.cat(image_scores)
    else:
        image_scores = torch.max(torch.nn.functional.avg_pool2d(preds,16,stride=2).view(preds.shape[0], -1), dim=-1)[0]
    # image_gts = np.max(masks, axis=(1, 2))
    # plt.imshow(preds[0].cpu())
    # plt.show()

    # for i in range(preds.shape[0]):
    #     plt.subplot(1,3,1)
    #     img = loader.dataset[i]['image'].numpy().transpose(1,2,0) * IMAGENET_STD + IMAGENET_MEAN
    #     plt.imshow(img)
    #     plt.subplot(1,3,2)
    #     plt.imshow(gts[i],cmap='gray')
    #     plt.subplot(1,3,3)
    #     plt.imshow(preds[i].cpu())
    #     plt.title(f'score: {image_scores[i]}')
    #     plt.show()
    return preds, gts, image_scores, image_gts


def predict_text(loader,model,mm_swin,embeddings,attention_masks,args,l=None,path_info=None,knn_info=None):
    outputsize = (model.feature_size[0]*8,model.feature_size[1]*8)
    # print(outputsize)
    preds, gts, image_gts,fgs = [], [], [], []
    with torch.no_grad():

        for batch in tqdm(loader):
            images = batch['image'].cuda()
            features = batch['feature'].cuda()
            masks = batch['mask']
            label = batch['label']
            filenames = batch['filename']
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            avg_pred = torch.zeros((images.shape[0], *outputsize),device='cuda')
            for eid, embedding in enumerate(embeddings):

                if attention_masks.shape[0] > 1:
                    fusion_features = mm_swin(images, embedding.unsqueeze(0), l_mask=attention_masks[eid].unsqueeze(0))
                else:
                    fusion_features = mm_swin(images, embedding.unsqueeze(0), l_mask=attention_masks)
                # f1 = torch.nn.functional.interpolate(f1, model.feature_size, mode='bilinear', align_corners=False)
                # fusion_features = torch.cat([f1, f2], dim=1)
                # features = features.cuda()

                pred = model(features,fusion_features)
                #

                # if pred == 1:
                pred = torch.sigmoid(pred)
                # else:
                # pred = torch.softmax(pred, dim=1)[:,:,1] # 100,256,1
                pred = pred.reshape(images.shape[0], -1, *outputsize)
                # print(pred.shape)
                if args.slide_window is not None:

                    out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                    t = torch.zeros(outputsize,device='cuda')
                    index = 0
                    for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                        for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                            out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                            t[i:i + args.slide_window, j:j + args.slide_window] += 1
                            index += 1
                    pred = out / t
                    pred = pred.unsqueeze(1)
                # print(embeddings.shape[0])
                # plt.subplot(1,3,1)
                # plt.imshow(pred[0,0].cpu())
                # plt.title(torch.max(pred[0].cpu()))
                # plt.subplot(1,3,2)
                # plt.imshow(masks[0,0])
                # plt.subplot(1,3,3)
                # plt.imshow(images[0].cpu().numpy().transpose((1,2,0)) * IMAGENET_STD + IMAGENET_MEAN)
                # plt.show()
                # print(avg_pred.shape,pred.shape)
                avg_pred = torch.max(torch.cat([avg_pred,pred[0]]),dim=0)[0].unsqueeze(0)
                # torch.cuda.synchronize()
                # print(time.time() - sttatee)

            if l is not None:
                features = features.numpy()
                features = np.sum(features,axis=1)
                # print(features.shape,pred.shape)
                avg_pred = features*l+avg_pred*(1-l)
            preds.append(avg_pred)
            gts.append(masks)
            # plt.subplot(1,3,1)
            # plt.imshow(avg_pred[0].cpu())
            # plt.title(torch.max(pred[0].cpu()))
            # plt.subplot(1,3,2)
            # plt.imshow(masks[0,0])
            # plt.subplot(1,3,3)
            # plt.imshow(images[0].cpu().numpy().transpose((1,2,0)) * IMAGENET_STD + IMAGENET_MEAN)
            # plt.show()
            image_gts.append(label)
            if args.with_fg:
                for i, filename in enumerate(filenames):
                    origin_path = path_info[os.path.basename(filename)]
                    basename = os.path.basename(origin_path)
                    fg_path = os.path.join(args.fg_dir,origin_path.replace(basename,f'f_{basename.replace("png","npy")}'))
                    if not os.path.exists(fg_path):
                        args.with_fg=False
                        break
                    fg = np.load(fg_path)[None,:,:]
                    # print(query_fg.shape)
                    knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                    for p in knn_list[:args.fg_knn]:
                        basename = os.path.basename(p)
                        ref_fg = np.load(os.path.join(args.fg_dir,args.dataset,p.replace(basename,f'f_{basename.replace("png","npy")}')))
                        fg = np.concatenate([fg,ref_fg[None,:,:]])
                    # print(fg.shape)
                    fg = np.max(fg,axis=0)
                    fg = cv2.resize(fg,masks.shape[-2:])
                    # fg = cv2.blur(fg,(20,20))

                    fgs.append(torch.tensor(fg)[None,:,:])
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
    if args.with_fg:
        fgs = torch.cat(fgs)
        preds = fgs * preds
    # image_gts = np.max(masks, axis=(1, 2))
    return preds, gts, image_scores, image_gts
def predict_text_weight(loader,model,embeddings,attention_masks,args,l=None):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts = [], [], []
    with torch.no_grad():

        for batch in tqdm(loader):
            images = batch['image'].cuda()
            residual = batch['feature']
            masks = batch['mask']
            label = batch['label']

            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            avg_pred = torch.zeros((images.shape[0], *outputsize),device='cuda')
            for eid, embedding in enumerate(embeddings):
                embedding = embedding.unsqueeze(0)
                features = residual.clone().cuda()
                features = features.permute(0,2,3,1)
                b, h, w, c = features.shape

                features = features.view(b, h * w, len(args.feature_folder), args.feature_channel).flatten(1,
                                                                                                           2) * embedding
                features = features.view(b, h, w, c).permute(0, 3, 1, 2)
                # print(embedding.shape,features.shape)
                pred = model(features)
                #

                if args.num_classes == 1:
                    pred = torch.sigmoid(pred)
                else:
                    pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
                if args.slide_window is not None:
                    pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                    out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                    t = torch.zeros(outputsize,device='cuda')
                    index = 0
                    for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                        for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                            out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                            t[i:i + args.slide_window, j:j + args.slide_window] += 1
                            index += 1
                    pred = out / t
                # print(embeddings.shape[0])
                # plt.subplot(1,3,1)
                # plt.imshow(pred[0].cpu())
                # plt.title(torch.max(pred[0].cpu()))
                # plt.subplot(1,3,2)
                # plt.imshow(masks[0,0])
                # plt.subplot(1,3,3)
                # plt.imshow(images[0].cpu().numpy().transpose((1,2,0)) * IMAGENET_STD + IMAGENET_MEAN)
                # plt.show()
                avg_pred = torch.max(torch.cat([avg_pred,pred]),dim=0)[0].unsqueeze(0)
                # print(avg_pred.shape)

                # torch.cuda.synchronize()
                # print(time.time() - sttatee)

            if l is not None:
                features = features.numpy()
                features = np.sum(features,axis=1)
                # print(features.shape,pred.shape)
                avg_pred = features*l+avg_pred*(1-l)
            preds.append(avg_pred)
            gts.append(masks)
            # plt.subplot(1,3,1)
            # plt.imshow(avg_pred[0].cpu())
            # plt.title(torch.max(pred[0].cpu()))
            # plt.subplot(1,3,2)
            # plt.imshow(masks[0,0])
            # plt.subplot(1,3,3)
            # plt.imshow(images[0].cpu().numpy().transpose((1,2,0)) * IMAGENET_STD + IMAGENET_MEAN)
            # plt.show()
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

def predict_ann_head(loader,model,args,seg_model=None,path_info=None,knn_info=None,l=None):
    outputsize = (model.feature_size[0] * 8, model.feature_size[1] * 8)
    # print(outputsize)
    preds, gts, image_gts, coord_features, all_ious, fgs = [], [], [], [], [], []
    sample_iou_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].cuda()
            features = batch['feature'].cuda()
            masks = batch['mask']
            label = batch['label']

            filenames = batch['filename']
            assert images.shape[0] == 1


            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0



            pred, coord_feature, best_iou,sample_ious = model.annotation(features, masks, args=args,seg_model=seg_model)


            preds.append(pred)
            gts.append(masks)
            image_gts.append(label)
            coord_features.append(coord_feature)
            all_ious.append(best_iou)
            sample_iou_list.append(sample_ious)
            if args.with_fg:
                for i, filename in enumerate(filenames):
                    origin_path = path_info[os.path.basename(filename)]
                    basename = os.path.basename(origin_path)
                    fg_path = os.path.join(args.fg_dir,
                                           origin_path.replace(basename, f'f_{basename.replace("png", "npy")}'))
                    if not os.path.exists(fg_path):
                        args.with_fg = False
                        break
                    fg = np.load(fg_path)[None, :, :]
                    # print(query_fg.shape)
                    knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                    for p in knn_list[:args.fg_knn]:
                        basename = os.path.basename(p)
                        ref_fg = np.load(os.path.join(args.fg_dir, args.dataset,
                                                      p.replace(basename, f'f_{basename.replace("png", "npy")}')))
                        fg = np.concatenate([fg, ref_fg[None, :, :]])
                    # print(fg.shape)
                    fg = np.max(fg, axis=0)
                    fg = cv2.resize(fg, masks.shape[-2:])
                    # fg = cv2.blur(fg,(20,20))

                    fgs.append(torch.tensor(fg)[None, :, :])
    preds = torch.cat(preds)  # 132 1024
    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)
    anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=gts.shape[-2:], gaussian=args.gaussian
    )
    preds = torch.reshape(preds, (-1, outputsize[0], outputsize[1]))
    image_scores = torch.max(preds.view(preds.shape[0], -1), dim=-1)[0]
    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds))



    if args.with_fg:
        fgs = torch.cat(fgs)
        preds = fgs * preds
        # preds[fgs < 0.5] = 0
        # all_ious = []
        # for p,gt in zip(preds,gts):
        #     all_ious.append(get_iou(gt,p>args.pred_thres))
        # plt.subplot(1,2,1)
        # plt.imshow(p.cpu()>args.pred_thres)
        # plt.title(f'iou={get_iou(gt,p>args.pred_thres)}')
        # plt.subplot(1,2,2)
        # plt.imshow(gt)
        # plt.title('GT')
        # plt.show()

    # print(len(all_ious))
    return preds, gts, image_scores, image_gts, np.array(all_ious).mean(),sample_iou_list


def predict_click_text(loader,model,mm_swin,args,seg_model=None,embeddings=None, attention_masks=None,path_info=None,knn_info=None):
    outputsize =  (model.feature_size[0]*8,model.feature_size[1]*8)
    # print(outputsize)
    preds, gts, image_gts, coord_features,all_ious,fgs = [], [], [], [],[],[]
    sample_iou_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].cuda()
            features = batch['feature'].cuda()
            masks = batch['mask']
            label = batch['label']
            sens = batch['sens']
            filenames = batch['filename']
            assert images.shape[0] == 1
            # print(len(sens))
            if embeddings is None:
                embeddings = batch['avg_emb'].cuda()
                attention_masks = batch['avg_amask'].cuda()
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            if args.slide_window is not None:
                b, c, h, w = images.shape
                images = torch.nn.functional.unfold(images,
                                                    (args.slide_window * model.out_stride,
                                                     args.slide_window * model.out_stride),
                                                    stride=args.slide_stride * model.out_stride).transpose(1,
                                                                                                           2).flatten(
                    0,
                    1)
                images = images.view(b, -1, c, args.slide_window * model.out_stride,
                                     args.slide_window * model.out_stride)
                if embeddings.shape[1] != images.shape[1]:
                    embeddings = embeddings.unsqueeze(1).repeat(1, images.shape[1], 1, 1)
                    attention_masks = attention_masks.unsqueeze(1).repeat(1, images.shape[1], 1, 1)
                images = images.flatten(0, 1)

            avg_pred = torch.zeros((images.shape[0], *outputsize),device='cuda')
            # print(embeddings.shape)
            for eid, embedding in enumerate(embeddings):

                if args.slide_window is not None:
                    if attention_masks.shape[0] > 1:
                        fusion_features = mm_swin(images, embedding,
                                                  l_mask=attention_masks[eid])
                    else:
                        fusion_features = mm_swin(images, embedding, l_mask=attention_masks[0])
                else:
                    if attention_masks.shape[0] > 1:
                        fusion_features = mm_swin(images, embedding.unsqueeze(0),
                                                  l_mask=attention_masks[eid].unsqueeze(0))
                    else:
                        fusion_features = mm_swin(images, embedding.unsqueeze(0), l_mask=attention_masks)

                pred,coord_feature,best_ious,sample_ious = model.annotation(features,fusion_features,masks,args,seg_model)

                avg_pred = torch.max(torch.cat([avg_pred, pred[0]]), dim=0)[0].unsqueeze(0)
            preds.append(avg_pred)
            gts.append(masks)
            image_gts.append(label)
            coord_features.append(coord_feature)
            all_ious.append(best_ious)
            sample_iou_list.append(sample_ious)
            if args.with_fg:
                for i, filename in enumerate(filenames):
                    origin_path = path_info[os.path.basename(filename)]
                    basename = os.path.basename(origin_path)
                    fg_path = os.path.join(args.fg_dir,origin_path.replace(basename,f'f_{basename.replace("png","npy")}'))
                    if not os.path.exists(fg_path):
                        args.with_fg=False
                        break
                    fg = np.load(fg_path)[None,:,:]
                    # print(query_fg.shape)
                    knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                    for p in knn_list[:args.fg_knn]:
                        basename = os.path.basename(p)
                        ref_fg = np.load(os.path.join(args.fg_dir,args.dataset,p.replace(basename,f'f_{basename.replace("png","npy")}')))
                        fg = np.concatenate([fg,ref_fg[None,:,:]])
                    # print(fg.shape)
                    fg = np.max(fg,axis=0)
                    fg = cv2.resize(fg,masks.shape[-2:])
                    # fg = cv2.blur(fg,(20,20))

                    fgs.append(torch.tensor(fg)[None,:,:])
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


    if args.with_fg:
        fgs = torch.cat(fgs)
        preds = fgs * preds

    return preds, gts, image_scores, image_gts,np.array(all_ious).mean(),sample_iou_list


def predict_click_text_no_head(loader,model,mm_swin,args,seg_model=None,embeddings=None, attention_masks=None,path_info=None,knn_info=None):
    outputsize = (model.feature_size[0] * 8, model.feature_size[1] * 8)
    # print(outputsize)
    preds, gts, image_gts, coord_features, all_ious, fgs = [], [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].cuda()
            features = batch['feature'].cuda()
            masks = batch['mask']
            label = batch['label']
            sens = batch['sens']
            filenames = batch['filename']
            assert images.shape[0] == 1
            # print(len(sens))
            if embeddings is None:
                embeddings = batch['avg_emb'].cuda()
                attention_masks = batch['avg_amask'].cuda()
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            if args.slide_window is not None:
                b, c, h, w = images.shape
                images = torch.nn.functional.unfold(images,
                                                    (args.slide_window * model.out_stride,
                                                     args.slide_window * model.out_stride),
                                                    stride=args.slide_stride * model.out_stride).transpose(1,
                                                                                                           2).flatten(
                    0,
                    1)
                images = images.view(b, -1, c, args.slide_window * model.out_stride,
                                     args.slide_window * model.out_stride)
                if embeddings.shape[1] != images.shape[1]:
                    embeddings = embeddings.unsqueeze(1).repeat(1, images.shape[1], 1, 1)
                    attention_masks = attention_masks.unsqueeze(1).repeat(1, images.shape[1], 1, 1)
                images = images.flatten(0, 1)

            avg_pred = torch.zeros((images.shape[0], *outputsize), device='cuda')
            for eid, embedding in enumerate(embeddings):
                if args.slide_window is not None:
                    if attention_masks.shape[0] > 1:
                        fusion_features = mm_swin(images, embedding,
                                                  l_mask=attention_masks[eid])
                    else:
                        fusion_features = mm_swin(images, embedding, l_mask=attention_masks)
                else:
                    if attention_masks.shape[0] > 1:
                        fusion_features = mm_swin(images, embedding.unsqueeze(0), l_mask=attention_masks[eid].unsqueeze(0))
                    else:
                        fusion_features = mm_swin(images, embedding.unsqueeze(0), l_mask=attention_masks[0])

                f1,f2 = fusion_features
                f1 = torch.nn.functional.interpolate(f1, model.vit_img_size, mode='bilinear', align_corners=False)
                fusion_features = torch.cat([f1, f2], dim=1)

                pred, coord_feature, sample_ious = model.annotation(features, fusion_features, masks, args, seg_model)
                #
                # plt.subplot(1,2,1)
                # plt.imshow(pred.cpu()[0,0])
                # plt.title(f'iou={sample_ious}')
                # plt.subplot(1,2,2)
                # plt.imshow(masks[0,0])
                # plt.title('GT')
                # plt.show()
                # pred = torch.sigmoid(pred)
                # pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
                # if args.slide_window is not None:
                #     pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                #     out = torch.zeros((images.shape[0], *outputsize), device='cuda')
                #     t = torch.zeros(outputsize, device='cuda')
                #     index = 0
                #     for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                #         for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                #             out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                #             t[i:i + args.slide_window, j:j + args.slide_window] += 1
                #             index += 1
                #     pred = out / t
                # print(pred.shape,avg_pred.shape)
                avg_pred = torch.max(torch.cat([avg_pred, pred[0]]), dim=0)[0].unsqueeze(0)
            preds.append(avg_pred)
            gts.append(masks)
            image_gts.append(label)
            coord_features.append(coord_feature)
            all_ious.append(sample_ious)
            if args.with_fg:
                for i, filename in enumerate(filenames):
                    origin_path = path_info[os.path.basename(filename)]
                    basename = os.path.basename(origin_path)
                    fg_path = os.path.join(args.fg_dir,
                                           origin_path.replace(basename, f'f_{basename.replace("png", "npy")}'))
                    if not os.path.exists(fg_path):
                        args.with_fg = False
                        break
                    fg = np.load(fg_path)[None, :, :]
                    # print(query_fg.shape)
                    knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                    for p in knn_list[:args.fg_knn]:
                        basename = os.path.basename(p)
                        ref_fg = np.load(os.path.join(args.fg_dir, args.dataset,
                                                      p.replace(basename, f'f_{basename.replace("png", "npy")}')))
                        fg = np.concatenate([fg, ref_fg[None, :, :]])
                    # print(fg.shape)
                    fg = np.max(fg, axis=0)
                    fg = cv2.resize(fg, masks.shape[-2:])
                    # fg = cv2.blur(fg,(20,20))

                    fgs.append(torch.tensor(fg)[None, :, :])
    preds = torch.cat(preds)  # 132 1024
    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)
    anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=gts.shape[-2:], gaussian=args.gaussian
    )
    preds = torch.reshape(preds, (-1, outputsize[0], outputsize[1]))
    image_scores = torch.max(preds.view(preds.shape[0], -1), dim=-1)[0]
    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds))

    # print(preds.shape,coord_features.shape) # (83, 512, 512) (83, 3, 512, 512)
    # plt.subplot(1,2,1)
    # plt.imshow(preds[-1])
    # preds = (preds+coord_features[:,1,:,:]-coord_features[:,2,:,:]).clip(0,1)
    # preds[coord_features[:,1,:,:] > 0.5] = 1
    # preds[coord_features[:,2,:,:] > 0.5] = 0
    # plt.subplot(1,2,2)
    # plt.imshow(preds[-1])
    # plt.show()
    # image_gts = np.max(masks, axis=(1, 2))

    if args.with_fg:
        fgs = torch.cat(fgs)
        preds = fgs * preds
    # print(len(all_ious))
    return preds, gts, image_scores, image_gts, np.array(all_ious).mean()
def predict_click_image(loader,model,feature_aggregator,args,seg_model=None,path_info=None,knn_info=None):
    outputsize =  (model.feature_size[0]*8,model.feature_size[1]*8)
    # print(outputsize)
    preds, gts, image_gts, coord_features,all_ious,fgs = [], [], [], [],[],[]
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].cuda()
            features = batch['feature'].cuda()
            masks = batch['mask']
            label = batch['label']
            filenames = batch['filename']
            assert images.shape[0] == 1
            # print(len(sens))
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            image_feature = feature_aggregator(images)
            image_feature = [image_feature[k] for k in feature_aggregator.layers_to_extract_from]  # 256,512,1792,1920

            pred,coord_feature,sample_ious = model.annotation(features,image_feature,masks,args,seg_model)

            # plt.subplot(1,2,1)
            # plt.imshow(pred.cpu()[0,0])
            # plt.title(f'iou={sample_ious}')
            # plt.subplot(1,2,2)
            # plt.imshow(masks[0,0])
            # plt.title('GT')
            # plt.show()
            # pred = torch.sigmoid(pred)
            # pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
            if args.slide_window is not None:
                pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                t = torch.zeros(outputsize,device='cuda')
                index = 0
                for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
            # print(pred.shape,avg_pred.shape)
            # avg_pred = torch.max(torch.cat([avg_pred, pred[0]]), dim=0)[0].unsqueeze(0)
            preds.append(pred)
            gts.append(masks)
            image_gts.append(label)
            coord_features.append(coord_feature)
            all_ious.append(sample_ious)
            if args.with_fg:
                for i, filename in enumerate(filenames):
                    origin_path = path_info[os.path.basename(filename)]
                    basename = os.path.basename(origin_path)
                    fg_path = os.path.join(args.fg_dir,origin_path.replace(basename,f'f_{basename.replace("png","npy")}'))
                    if not os.path.exists(fg_path):
                        args.with_fg=False
                        break
                    fg = np.load(fg_path)[None,:,:]
                    # print(query_fg.shape)
                    knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                    for p in knn_list[:args.fg_knn]:
                        basename = os.path.basename(p)
                        ref_fg = np.load(os.path.join(args.fg_dir,args.dataset,p.replace(basename,f'f_{basename.replace("png","npy")}')))
                        fg = np.concatenate([fg,ref_fg[None,:,:]])
                    # print(fg.shape)
                    fg = np.max(fg,axis=0)
                    fg = cv2.resize(fg,masks.shape[-2:])
                    # fg = cv2.blur(fg,(20,20))

                    fgs.append(torch.tensor(fg)[None,:,:])
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

    # print(preds.shape,coord_features.shape) # (83, 512, 512) (83, 3, 512, 512)
    # plt.subplot(1,2,1)
    # plt.imshow(preds[-1])
    # preds = (preds+coord_features[:,1,:,:]-coord_features[:,2,:,:]).clip(0,1)
    # preds[coord_features[:,1,:,:] > 0.5] = 1
    # preds[coord_features[:,2,:,:] > 0.5] = 0
    # plt.subplot(1,2,2)
    # plt.imshow(preds[-1])
    # plt.show()
    # image_gts = np.max(masks, axis=(1, 2))

    if args.with_fg:
        fgs = torch.cat(fgs)
        preds = fgs * preds
        # preds[fgs < 0.5] = 0
        # all_ious = []
        # for p,gt in zip(preds,gts):
        #     all_ious.append(get_iou(gt,p>args.pred_thres))
            # plt.subplot(1,2,1)
            # plt.imshow(p.cpu()>args.pred_thres)
            # plt.title(f'iou={get_iou(gt,p>args.pred_thres)}')
            # plt.subplot(1,2,2)
            # plt.imshow(gt)
            # plt.title('GT')
            # plt.show()

    # print(len(all_ious))
    return preds, gts, image_scores, image_gts,np.array(all_ious).mean()


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
    # torch.backends.cudnn.enabled = False

