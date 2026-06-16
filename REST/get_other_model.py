import json
import os
import pickle

import torch

from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer

from model.text_tokenizer import get_text_token


def get_bert(device='cuda'):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_class = BertModel
    single_bert_model = model_class.from_pretrained('/media/wu/E/PRN_Vit/work_dirs/bert-base-uncased')
    single_bert_model.pooler = None


    bert_model = single_bert_model.to(device)

    return tokenizer,bert_model

def get_avg_text_embedding(single_dataset,tokenizer,bert_model,regenerate=False):
    embedding_path = f'data/mvtec_text/{single_dataset}_avg.pkl'
    if not os.path.exists(embedding_path) or regenerate:
        print('Get average text embedding...')
        with open(f'data/mvtec_text/{single_dataset}.json', 'r', encoding='utf-8') as f1:

            texts = json.load(f1)[single_dataset]

        sens = []
        for k in texts.keys():
           
            # print(k)
            sens.extend(texts[k][:40])
        padded_sent_toks_list, attention_mask_list = [], []
        with torch.no_grad():
            for sentence in sens:
                padded_sent_toks, attention_mask = get_text_token(tokenizer, sentence)
                padded_sent_toks_list.append(padded_sent_toks)
                attention_mask_list.append(attention_mask)
            padded_sent_toks_list = torch.cat(padded_sent_toks_list).cuda()
            attention_mask_list = torch.cat(attention_mask_list).cuda()

            last_hidden_states = bert_model(padded_sent_toks_list, attention_mask=attention_mask_list)[0]
        embeddings = last_hidden_states.permute(0, 2, 1)
        attention_mask = torch.mean(attention_mask_list.float(),0).unsqueeze(0).unsqueeze(-1)
        embedding = torch.mean(embeddings,0).unsqueeze(0)
        with open(embedding_path,'wb') as f:
            pickle.dump({"embedding": embedding, "attention_mask": attention_mask},f)
    else:
        with open(embedding_path,'rb') as f:
            info = pickle.load(f)
            embedding = info['embedding']
            attention_mask = info["attention_mask"]
    return  embedding.cuda(),attention_mask.cuda()
def get_anomaly_text_embedding(single_dataset,tokenizer,bert_model,root_dir='data/mvtec_text',regenerate=False):
    embedding, attention_mask = [], []
    embedding_path = f'{root_dir}/{single_dataset}_anomaly.pkl'
    with open(f'{root_dir}/{single_dataset}.json', 'r', encoding='utf-8') as f1:

        texts = json.load(f1)[single_dataset]
    if not os.path.exists(embedding_path) or regenerate:
        # print('Get average text embedding...')

        embedding, attention_mask = [], []
        emb_dict = {}
        for k in texts.keys():
            sens = texts[k][:40]
            assert len(sens)>=40

            padded_sent_toks_list, attention_mask_list = [], []
            with torch.no_grad():
                for sentence in sens:
                    padded_sent_toks, attention_masks = get_text_token(tokenizer, sentence)
                    padded_sent_toks_list.append(padded_sent_toks)
                    attention_mask_list.append(attention_masks)
                padded_sent_toks_list = torch.cat(padded_sent_toks_list).cuda()
                attention_mask_list = torch.cat(attention_mask_list).cuda()

                last_hidden_states = bert_model(padded_sent_toks_list, attention_mask=attention_mask_list)[0]
            embeddings = last_hidden_states.permute(0, 2, 1)
            anomaly_attention_mask = attention_mask_list
            anomaly_embedding = embeddings

            emb_dict[f'{k}_embedding'] = anomaly_embedding.cpu()
            emb_dict[f'{k}_attention_mask'] = anomaly_attention_mask.cpu()

            # if  k == 'false_ng': continue
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
            if  k == 'false_ng': continue
            anomaly_embedding = info[f'{k}_embedding']
            anomaly_attention_mask = info[f'{k}_attention_mask']
            embedding.append(anomaly_embedding)
            attention_mask.append(anomaly_attention_mask)
        embedding = torch.cat(embedding)
        attention_mask = torch.cat(attention_mask)
    return  embedding.cuda(),attention_mask.cuda()




if __name__ == '__main__':
    tokenizer, bert_model = get_bert()
    text_dir = '/media/wu/E/ADClick/data/mvtec3d_text'
    categories = ['bagel', 'carrot', 'dowel', 'potato', 'rope',
                  'cable_gland', 'cookie', 'foam', 'peach', 'tire']

    #
    # save_dependencies_files(os.path.join(root_dir, os.path.join(sys.argv[0])))

    for single_dataset in categories:

        embeddings, attention_masks = get_anomaly_text_embedding(single_dataset, tokenizer, bert_model,root_dir=text_dir,regenerate=True)