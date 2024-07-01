import os
import sys
import copy

from src.utils import (save_object,
                       load_pickle_obj)

from src.relative_embedding import (AnchorDescriptors,
                                    construct_relative_embeddings)

def main():
    anchor_descirptions = copy.deepcopy(AnchorDescriptors)

    hint_flag = False
    folder_name = './outputs'
    checked_files = ['BERT_embeddings.pkl', 'BART_embeddings.pkl', 'GPT2_embeddings.pkl']
    for fname in checked_files:
        if not os.path.isfile(os.path.join(folder_name, fname)):
            hint_flag = True

    if hint_flag:
        print('Please run embeddings_from_pretrained.py first.')
        sys.exit(0)

    print('Start process relative embedding of BERT model ...')
    bert_embeddings = load_pickle_obj(os.path.join(folder_name, 'BERT_embeddings.pkl'))
    bert_embeddings = construct_relative_embeddings('BERT', 'encoder', 
            anchor_descirptions, bert_embeddings)

    print('Start process relative embedding of BART encoder ...')
    bart_embeddings = load_pickle_obj(os.path.join(folder_name, 'BART_embeddings.pkl'))
    bart_embeddings = construct_relative_embeddings('BART', 'encoder', 
            anchor_descirptions, bart_embeddings)

    print('Start process relative embedding of BART decoder ...')
    bart_embeddings = construct_relative_embeddings('BART', 'decoder', 
            anchor_descirptions, bart_embeddings)

    print('Start process relative embedding of GPT-2 model ...')
    gpt_embeddings = load_pickle_obj(os.path.join(folder_name, 'GPT2_embeddings.pkl'))
    gpt_embeddings = construct_relative_embeddings('GPT-2', 'decoder',
            anchor_descirptions, gpt_embeddings)

    print('Processing relative embedding finish, start saving files ...')
    save_object(os.path.join(folder_name, 'BERT_embeddings.pkl'), bert_embeddings)
    save_object(os.path.join(folder_name, 'BART_embeddings.pkl'), bart_embeddings)
    save_object(os.path.join(folder_name, 'GPT2_embeddings.pkl'), gpt_embeddings)

    print('Program finish.')
    return None

if __name__ == '__main__':
    main()


