import glob
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from REST.model.text_cls_model import TextClassifier
from REST.util.mvtec import MVTecTextDataset
from REST.util.util import fix_seed

text_dir = '/media/wu/E/ADClick/data/mvtec3d_text'
categories = ['bagel','carrot',  'dowel',  'potato',      'rope',
'cable_gland',  'cookie',  'foam',   'peach','tire']


fix_seed(0)
batchsize=512
LEARNING_RATE =  1e-5
weight_decay = 0.05
# work_dir = f'../weights/mvtec_text_classifier'
# os.makedirs(work_dir,exist_ok=True)
dataset = MVTecTextDataset(text_dir,categories)
train_loader = DataLoader(dataset,batchsize,shuffle=True)


model = TextClassifier()
model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=weight_decay)
ce_loss = torch.nn.CrossEntropyLoss()
best_acc = 0
step = 0
model.zero_grad()
while True:
    model.train()
    for embeddings,labels in tqdm(train_loader):
        embeddings = embeddings.cuda()
        labels = labels.cuda()

        output = model(embeddings)
        loss = ce_loss(output,labels.cuda())

        loss.backward()
        optimizer.step()
        model.zero_grad()
        step = step + 1

    model.eval()
    with torch.no_grad():
        total_labels = []
        total_preds = []
        for embeddings, labels in tqdm(train_loader):
            embeddings = embeddings.cuda()
            labels = labels.cuda()
            output = model(embeddings)
            output = torch.argmax(torch.softmax(output,dim=-1),dim=-1)
            total_preds.append(output)
            total_labels.append(labels)
        total_preds =torch.cat(total_preds)
        total_labels =torch.cat(total_labels)
        acc = (total_labels == total_preds).sum()/total_labels.shape[0]
        print(f'step: {step}, acc: {acc}')
        if acc > best_acc:
            best_acc = acc
            # torch.save(model.state_dict(),f'{work_dir}/best.pth')
    if best_acc == 1 or step >= 1000:
        break
model.eval()
pathlist = sorted(glob.glob(f'{text_dir}/*_anomaly.pkl'))
for path in pathlist:
    feature_info = {}
    feature_list = []
    with open(path,'rb') as f:
        prompt_info = pickle.load(f)
        for k, v in prompt_info.items():
            if 'embedding' not in k: continue
            anomaly = k.replace('_embedding', '')
            if anomaly == 'good': continue
            v = torch.Tensor(v).cuda()
            with torch.no_grad():
                classifier_feature = model.forward_feature(v).cpu()
            feature_info[k] = classifier_feature


            feature_list.append(classifier_feature)
        feature_list = torch.cat(feature_list)
        feature_info['all_embedding'] = feature_list
    with open(path.replace("anomaly","classifier"), 'wb') as f:
        pickle.dump(feature_info, f)