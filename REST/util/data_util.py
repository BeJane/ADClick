import json
import os.path
import pickle


def get_info(data_dir,dataset,args):
    origin_path = f'{data_dir}/origin_path.json'
    with open(origin_path, 'r', encoding='utf-8') as f1:
        path_info = json.load(f1)
    knn_info_path = f'{args.fg_dir}/{dataset}/r_result.json'
    if os.path.exists(knn_info_path):
        with open(knn_info_path, 'r', encoding='utf-8') as f2:
            knn_info = json.load(f2)
    else:
        print(f'{dataset} does not have foreground information!')
        knn_info = None

    with open(f'data/{args.text_dir}/{dataset}.json', 'r', encoding='utf-8') as f1:

        text_info = json.load(f1)[dataset]
    with open(f'data/{args.text_dir}/{dataset}_{args.text_avg}.pkl','rb') as f:
        avg_text_info = pickle.load(f)

    return {'dataset':dataset,
        'path':path_info,
            'knn': knn_info,
            'text': text_info,
            'avg_text': avg_text_info}