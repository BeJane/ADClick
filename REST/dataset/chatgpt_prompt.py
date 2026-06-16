"""
answer the following questions with 100 phrases in json format:
{“carpet”:{“color”:[2 phrases describing the hole defect on the carpet, all in lowercase,with carpet],“hole”:[2 phrases describing the hole defect on the carpet, all in lowercase,with carpet]}
"""
import json
import os
root_dir = '/media/szcyxy/新加卷/qi_data/mvtec_anomaly_detection'
out_path = 'mvtec_chatgpt_prompt.json'
chat_prompt = {}
phrases_num = 100
for single_dataset in os.listdir(root_dir):
    if '.' in  single_dataset:continue

    with open(f'../data/mvtec_text/{single_dataset}.json', 'r', encoding='utf-8') as f1:

        texts = json.load(f1)[single_dataset]
        # print(texts.keys())
        # for k in texts.keys():
        #     print(len(texts[k]),k)
        #     if len(texts[k]) < 40:
        #         print(single_dataset,k)

    chat_prompt[single_dataset] = {'good':[f'{phrases_num} phrases describing a {single_dataset} without industrial defect, all in lowercase,with {single_dataset}?'],
                                   'false_ng':[f'{phrases_num} phrases describing the industrial contamination defect on the {single_dataset}, all in lowercase,with {single_dataset}?']}
    for category in sorted(os.listdir(os.path.join(root_dir,single_dataset,'ground_truth'))):
        print(single_dataset,category)
        print(len(texts[category]))
        chat_prompt[single_dataset][category] = [f'{phrases_num} phrases describing the industrial {category} defect on the {single_dataset}, all in lowercase,with {single_dataset}?']
with open(out_path, 'w', encoding="utf-8") as f:
    json.dump(chat_prompt, f, indent=4, ensure_ascii=False)