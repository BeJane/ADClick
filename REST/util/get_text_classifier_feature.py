import json
import os
import pickle

import torch

from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer
from get_other_model import get_anomaly_avg_text_embedding
from model.text_cls_model import TextClassifier
from model.text_tokenizer import get_text_token


def get_anomaly_avg_text_classifier_feature(single_dataset,text_classifier,tokenizer,bert_model,root_dir='../data/mvtec_text',regenerate=False):
    embedding, attention_mask = [], []
    embedding_path = f'{root_dir}/{single_dataset}_anomaly_avg_text_classifier.pkl'
    with open(f'{root_dir}/{single_dataset}.json', 'r', encoding='utf-8') as f1:

        texts = json.load(f1)[single_dataset]
    if not os.path.exists(embedding_path) or regenerate:
        print('Get average text embedding...')

        embedding, attention_mask = [], []
        emb_dict = {}
        for k in texts.keys():
            # print(k)
            sens = texts[k][:40]

            padded_sent_toks_list, attention_mask_list = [], []
            with torch.no_grad():
                for sentence in sens:
                    padded_sent_toks, attention_masks = get_text_token(tokenizer, sentence)
                    padded_sent_toks_list.append(padded_sent_toks)
                    attention_mask_list.append(attention_masks)
                padded_sent_toks_list = torch.cat(padded_sent_toks_list).cuda()
                attention_mask_list = torch.cat(attention_mask_list).cuda()

                last_hidden_states = bert_model(padded_sent_toks_list, attention_mask=attention_mask_list)[0]

                last_hidden_states = last_hidden_states.flatten(1, -1)

                embeddings = text_classifier.forward_feature(last_hidden_states)

            anomaly_attention_mask = torch.mean(attention_mask_list.float(),0).unsqueeze(0).unsqueeze(-1)
            anomaly_embedding = torch.mean(embeddings,0).unsqueeze(0)
            emb_dict[f'{k}_embedding'] = anomaly_embedding.cpu()
            emb_dict[f'{k}_attention_mask'] = anomaly_attention_mask.cpu()
            if  k == 'false_ng': continue
            embedding.append(anomaly_embedding)
            attention_mask.append(anomaly_attention_mask)
        embedding = torch.cat(embedding)
        attention_mask = torch.cat(attention_mask)
        with open(embedding_path,'wb') as f:
            pickle.dump(emb_dict,f)
    else:
        with open(embedding_path,'rb') as f:
            info = pickle.load(f)
        for k in texts.keys():
            if k == 'false_ng': continue
            anomaly_embedding = info[f'{k}_embedding']
            anomaly_attention_mask = info[f'{k}_attention_mask']
            embedding.append(anomaly_embedding)
            attention_mask.append(anomaly_attention_mask)
        embedding = torch.cat(embedding)
        attention_mask = torch.cat(attention_mask)
    return  embedding.cuda(),attention_mask.cuda()
if __name__ == '__main__':

    data_dir = '../data/defect_512/mvtec'
    text_classifier_path = '../work_dirs/mvtec_text_classifier3/best-504.pth'
    device = 'cuda:0'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_class = BertModel
    single_bert_model = model_class.from_pretrained('../work_dirs/bert-base-uncased')
    single_bert_model.pooler = None
    text_classifier = TextClassifier()
    text_classifier.load_state_dict(torch.load(text_classifier_path,map_location='cpu'))


    bert_model = single_bert_model.to(device)
    text_classifier = text_classifier.to(device)
    for single_dataset in os.listdir(data_dir):
        embeddings, attention_masks =  get_anomaly_avg_text_classifier_feature(single_dataset,text_classifier,tokenizer,bert_model,regenerate=True)