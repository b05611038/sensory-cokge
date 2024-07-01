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
    checked_files = ['finetuned-BERT_embeddings.pkl', 
                     'finetuned-BART_embeddings.pkl', 
                     'finetuned-GPT2_embeddings.pkl']

    for fname in checked_files:
        if not os.path.isfile(os.path.join(folder_name, fname)):
            hint_flag = True

    if hint_flag:
        print('Please run embeddings_from_finetuned.py first.')
        sys.exit(0)

    print('Start process relative embedding of finetuned BERT model ...')
    bert_embeddings = load_pickle_obj(os.path.join(folder_name, 
                                                   'finetuned-BERT_embeddings.pkl'))

    bert_embeddings = construct_relative_embeddings('BERT', 'encoder', 
            anchor_descirptions, bert_embeddings)

    print('Start process relative embedding of finetuned BART encoder ...')
    bart_embeddings = load_pickle_obj(os.path.join(folder_name, 
                                                   'finetuned-BART_embeddings.pkl'))

    bart_embeddings = construct_relative_embeddings('BART', 'encoder', 
            anchor_descirptions, bart_embeddings)

    print('Start process relative embedding of finetuned BART decoder ...')
    bart_embeddings = construct_relative_embeddings('BART', 'decoder',
            anchor_descirptions, bart_embeddings)

    print('Start process relative embedding of finetuned GPT-2 model ...')
    gpt_embeddings = load_pickle_obj(os.path.join(folder_name, 
                                                  'finetuned-GPT2_embeddings.pkl'))

    gpt_embeddings = construct_relative_embeddings('GPT-2', 'decoder',
            anchor_descirptions, gpt_embeddings)

    print('Processing relative embedding finish, start saving files ...')
    save_object(os.path.join(folder_name, 'finetuned-BERT_embeddings.pkl'),
                bert_embeddings)
    save_object(os.path.join(folder_name, 'finetuned-BART_embeddings.pkl'),
                bart_embeddings)
    save_object(os.path.join(folder_name, 'finetuned-GPT2_embeddings.pkl'),
                gpt_embeddings)

    print('Program finish.')
    return None

if __name__ == '__main__':
    main()


