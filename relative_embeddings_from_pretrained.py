import os
import sys
import copy
import argparse

from src.utils import (save_object,
                       load_pickle_obj)

from src.relative_embedding import (AnchorDescriptors,
                                    construct_relative_embeddings)

def main():
    anchor_descirptions = copy.deepcopy(AnchorDescriptors)
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type = str, default = './outputs',
            help = 'The folder that you want to sture these object.')
    parser.add_argument('--enable_LLMs', action = 'store_true',
            help = 'To determine calculte gigantic LM results.')

    args = parser.parse_args()

    hint_flag = False
    folder_name = './outputs'
    checked_files = ['pretrained-ALBERT_embeddings.pkl',
                     'pretrained-BERT_embeddings.pkl', 
                     'pretrained-BART_embeddings.pkl', 
                     'pretrained-GPT2_embeddings.pkl',
                     'pretrained-RoBERTa_embeddings.pkl',
                     'pretrained-T5_embeddings.pkl']

    if args.enable_LLMs:
        checked_files.append('pretrained-Gemma2_embeddings.pkl')
        checked_files.append('pretrained-Llama3_embeddings.pkl')
        checked_files.append('pretrained-Qwen2_embeddings.pkl')

    for fname in checked_files:
        if not os.path.isfile(os.path.join(args.folder_name, fname)):
            hint_flag = True

    if hint_flag:
        print('Please run embeddings_from_pretrained.py first.')
        sys.exit(0)

    for filename in checked_files:
        model_type = filename.replace('pretrained-', '')
        model_type = model_type.replace('_embeddings.pkl', '')

        model_embeddings = load_pickle_obj(os.path.join(args.folder_name, filename))
        has_encoder, has_decoder = True, True
        for key in model_embeddings:
            if model_embeddings[key].get('encoder_embedding', None) is None:
                has_encoder = False

            if model_embeddings[key].get('decoder_embedding', None) is None:
                has_decoder = False

        if has_encoder:
            print('Start process relative embedding of {0} pretrained encoder ... '\
                    .format(model_type))

            model_embeddings = construct_relative_embeddings(model_type, 
                                                             'encoder', 
                                                             anchor_descirptions, 
                                                             model_embeddings)

        if has_decoder:
            print('Start process relative embedding of {0} pretrained decoder ... '\
                    .format(model_type))

            model_embeddings = construct_relative_embeddings(model_type, 
                                                             'decoder',
                                                             anchor_descirptions,
                                                             model_embeddings)


        if has_encoder or has_decoder:
            fname = os.path.join(args.folder_name, filename)
            save_object(fname, model_embeddings)
            print('Save relative embedding to file: {0}'.format(fname))

    print('Program finish.')

    return None

if __name__ == '__main__':
    main()


