import os
import copy

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sensory_cokge.utils import load_pickle_obj
from sensory_cokge.relative_embedding import DescriptorColors
from sensory_cokge.graph import description_graph

def concat_folder(file_dict, folder):
    for ftype in file_dict:
        file_dict[ftype] = os.path.join(folder, file_dict[ftype])
        
    return file_dict

def load_embeddings(file_dict):
    embeddings = {}
    for embedding_type in file_dict:
        if os.path.isfile(file_dict[embedding_type]):
            embeddings[embedding_type] = load_pickle_obj(file_dict[embedding_type])
        else:
            embeddings[embedding_type] = None
            
    return embeddings

def construct_dataset(embedding_dict, pretrain_status, struc, target, mapping_label):
    embeddings, labels = [], []
    for idx in embedding_dict[pretrain_status]:
        single_embedding = embedding_dict[pretrain_status][idx][f'{target}{struc}_embedding']
        description = embedding_dict[pretrain_status][idx]['description']
        single_label = mapping_label[description]

        embeddings.append(single_embedding)
        labels.append(single_label)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype = torch.int32)
    return embeddings, labels

def silhouette_score_torch(X: torch.Tensor,
                           labels: torch.Tensor,
                           metric: str = 'l2') -> torch.Tensor:
    """
    Compute mean silhouette score for clustering in PyTorch.
    
    Args:
      X        Tensor[N, D] of embeddings
      labels   Tensor[N] of integer cluster labels
      metric   'l2'      → Euclidean distance
               'cosine'  → 1 – cosine similarity
    
    Returns:
      scalar mean silhouette score
    """
    # 1) Pairwise distance matrix
    if metric == 'l2':
        D = torch.cdist(X, X, p=2)               # (N, N)
    elif metric == 'cosine':
        Xn = F.normalize(X, dim=1, eps=1e-8)     # (N, D)
        sim = Xn @ Xn.t()                        # (N, N)
        D   = 1. - sim                           # (N, N)
    else:
        raise ValueError("metric must be 'l2' or 'cosine'")

    N           = X.size(0)
    uniq_labels = torch.unique(labels)
    a = torch.zeros(N, device=X.device)         # intra-cluster dist
    b = torch.full((N,), float('inf'), device=X.device)  # nearest-cluster dist

    # 2) For each cluster, compute a and b
    for c in uniq_labels:
        mask_c    = (labels == c)
        mask_out  = ~mask_c
        
        # a[i] = mean distance to all other pts in same cluster
        D_in = D[mask_c][:, mask_c]
        if D_in.size(1) > 1:
            # sum over row minus the zero self-dist, divided by (n-1)
            a[mask_c] = (D_in.sum(dim=1) / (D_in.size(1) - 1))
        else:
            a[mask_c] = 0.0

        # b[i] = min over other clusters of mean distance to that cluster
        for c2 in uniq_labels:
            if c2 == c:
                continue
            mask_c2 = (labels == c2)
            D_out   = D[mask_c][:, mask_c2]        # pts in c → pts in c2
            b[mask_c] = torch.minimum(b[mask_c],
                                     D_out.mean(dim=1))

    # 3) silhouette for each point and then average
    sil = (b - a) / torch.maximum(a, b)
    return sil.mean().item()

def main():
    folder_name = './outputs'
    albert_files = {'pretrained': 'pretrained-ALBERT_embeddings.pkl',
                    'finetuned': 'finetuned-ALBERT-N5kE3_embeddings.pkl'}
    bart_files = {'pretrained': 'pretrained-BART_embeddings.pkl',
                  'finetuned': 'finetuned-BART-N10kE3_embeddings.pkl'}
    bert_files = {'pretrained': 'pretrained-BERT_embeddings.pkl',
                  'finetuned': 'finetuned-BERT-N20kE3_embeddings.pkl'}
    gpt_files = {'pretrained': 'pretrained-GPT2_embeddings.pkl',
                 'finetuned': 'finetuned-GPT2-N100kE8_embeddings.pkl'}
    roberta_files = {'pretrained': 'pretrained-RoBERTa_embeddings.pkl',
                     'finetuned': 'finetuned-RoBERTa-N100kE1_embeddings.pkl'}
    t5_files = {'pretrained': 'pretrained-T5_embeddings.pkl',
                'finetuned': 'finetuned-T5-N100kE8_embeddings.pkl'}

    cat_index = {'floral': 0,
                 'fruity': 1,
                 'sour/fermented': 2,
                 'green/vegetative': 3,
                 'other': 4,
                 'roasted': 5,
                 'spices': 6,
                 'nutty/cocoa': 7,
                 'sweet': 8}

    graph = description_graph()
    categorized_descriptions, mapping_label = {}, {}
    for descritpion in graph.descriptions:
        for cat in DescriptorColors:
            if graph.distance_between_descriptions(descritpion, cat) < 10.:
                if cat not in categorized_descriptions.keys():
                    categorized_descriptions[cat] = [descritpion]
                else:
                    categorized_descriptions[cat].append(descritpion)  
                    
    for cat in categorized_descriptions:
        categorized_descriptions[cat] = list(set(categorized_descriptions[cat]))
        for description in categorized_descriptions[cat]:
            mapping_label[description] = cat_index[cat]

    albert_files = concat_folder(albert_files, folder_name)
    bart_files = concat_folder(bart_files, folder_name)
    bert_files = concat_folder(bert_files, folder_name)
    gpt_files = concat_folder(gpt_files, folder_name)
    roberta_files = concat_folder(roberta_files, folder_name)
    t5_files = concat_folder(t5_files, folder_name)
    
    albert_embedding_dict = load_embeddings(albert_files)
    bart_embedding_dict = load_embeddings(bart_files)
    bert_embedding_dict = load_embeddings(bert_files)
    gpt_embedding_dict = load_embeddings(gpt_files)
    roberta_embedding_dict = load_embeddings(roberta_files)
    t5_embedding_dict = load_embeddings(t5_files)

    target = '' # empty use original embedding, put in 'relative_' if want to use relative embedding
    results = {'l2': {}, 'cosine': {}}
    for dist_func in results:
        for pretrain_status in ['pretrained', 'finetuned']:
            (albert_enc_embeddings,
             albert_enc_labels) = construct_dataset(albert_embedding_dict,
                                                    pretrain_status,
                                                    'encoder',
                                                    target,
                                                    mapping_label)

            score = silhouette_score_torch(albert_enc_embeddings, albert_enc_labels, dist_func)
            results[dist_func][f'{pretrain_status}_albert'] = score

            (bert_enc_embeddings,
             bert_enc_labels) = construct_dataset(bert_embedding_dict,
                                                  pretrain_status,
                                                  'encoder',
                                                  target,
                                                  mapping_label)

            score = silhouette_score_torch(bert_enc_embeddings, bert_enc_labels, dist_func)
            results[dist_func][f'{pretrain_status}_bert'] = score

            (roberta_enc_embeddings,
             roberta_enc_labels) = construct_dataset(roberta_embedding_dict,
                                                     pretrain_status,
                                                     'encoder',
                                                     target,
                                                     mapping_label)

            score = silhouette_score_torch(roberta_enc_embeddings, roberta_enc_labels, dist_func)
            results[dist_func][f'{pretrain_status}_roberta'] = score

            (bart_enc_embeddings,
             bart_enc_labels) = construct_dataset(bart_embedding_dict,
                                                  pretrain_status,
                                                  'encoder',
                                                  target,
                                                  mapping_label)

            score = silhouette_score_torch(bart_enc_embeddings, bart_enc_labels, dist_func)
            results[dist_func][f'{pretrain_status}_bart_enc'] = score

            (bart_dec_embeddings,
             bart_dec_labels) = construct_dataset(bart_embedding_dict,
                                                  pretrain_status,
                                                  'decoder',
                                                  target,
                                                  mapping_label)

            score = silhouette_score_torch(bart_dec_embeddings, bart_dec_labels, dist_func)
            results[dist_func][f'{pretrain_status}_bart_dec'] = score

            (t5_enc_embeddings,
             t5_enc_labels) = construct_dataset(t5_embedding_dict,
                                                pretrain_status,
                                                'encoder',
                                                target,
                                                mapping_label)

            score = silhouette_score_torch(t5_enc_embeddings, t5_enc_labels, dist_func)
            results[dist_func][f'{pretrain_status}_t5_enc'] = score

            (t5_dec_embeddings,
             t5_dec_labels) = construct_dataset(t5_embedding_dict,
                                                pretrain_status,
                                                'decoder',
                                                target,
                                                mapping_label)

            score = silhouette_score_torch(t5_dec_embeddings, t5_dec_labels, dist_func)
            results[dist_func][f'{pretrain_status}_t5_dec'] = score

            (gpt_dec_embeddings,
             gpt_dec_labels) = construct_dataset(gpt_embedding_dict,
                                                 pretrain_status,
                                                 'decoder',
                                                 target,
                                                 mapping_label)

            score = silhouette_score_torch(gpt_dec_embeddings, gpt_dec_labels, dist_func)
            results[dist_func][f'{pretrain_status}_gpt'] = score

    model_names = ['albert', 'bert', 'roberta', 'bart_enc', 'bart_dec', 't5_enc', 't5_dec', 'gpt']
    layout_filename = f'{target}embedding_silhouette_score.csv'
    lines = ['model,l2_pretrained,l2_finetined,angle_pretrained,angle_finetuned,\n']
    for model_name in model_names:
        single_line = model_name + ','
        for dist_func in results:
            for pretrain_status in ['pretrained', 'finetuned']:
                target_key = f'{pretrain_status}_{model_name}'
                score = results[dist_func][target_key]
                single_line += '{0:.5f},'.format(score)

        single_line += '\n'
        lines.append(single_line)

    with open(layout_filename, 'w') as F:
        F.writelines(lines)
        F.close()

    print(f'Successfully layout results in {layout_filename}.')

    return None
    

if __name__ == '__main__':
    main()
