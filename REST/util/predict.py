import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from model.patchcore.common import RescaleSegmentor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_with_coreset_fg(loader,model,  knn_info, args,l=None):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts,image_scores = [], [], [],[]
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

            pred = model(features)
            if args.image_cls:
                pred,cls_pred = pred
                cls_pred = torch.softmax(cls_pred, dim=1)[:, 1]
                image_scores.append(cls_pred)
            #
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
                basename = os.path.basename(filename)
                # fg = np.load(os.path.join(args.fg_dir,args.dataset,f'{basename.split(".")[0]+".npy"}'))[None,:,:]
                # print(query_fg.shape)
                fg_list = []
                knn_list = knn_info[basename]
                for p in knn_list[:args.fg_knn]:

                    ref_fg = np.load(os.path.join(args.fg_dir,args.dataset,f'{p.split(".")[0]+".npy"}'))
                    fg_list.append(ref_fg[None,::])
                # print(fg.shape)
                fg = np.max(np.concatenate(fg_list,axis=0),axis=0)
                # fg = np.max(np.clip(fg,0,1), axis=0)
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
                # pred = torch.tensor(fg.flatten(),device='cuda')**0.15 * pred
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
    # image_scores = torch.max(preds.view(preds.shape[0],-1), dim=-1)[0].cuda()
    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds))
    # print(fgs.shape,preds.shape)
    #
    preds = fgs**0.15 * preds
    # preds = (preds - torch.min(preds))/(torch.max(preds) - torch.min(preds))
    # for p in preds:
    #     plt.imshow(p)
    #     plt.show()
    preds = preds.cuda()
    if args.image_cls:
        image_scores = torch.cat(image_scores)
    else:
        image_scores = torch.max(torch.nn.functional.avg_pool2d(preds,16,stride=2).view(preds.shape[0], -1), dim=-1)[0]
    # image_gts = np.max(masks, axis=(1, 2))
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

    # image_gts = np.max(masks, axis=(1, 2))
    return preds, gts, image_scores, image_gts
