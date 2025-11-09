import os
import sys
import copy
import argparse

from sensory_cokge.utils import (save_object,
                                 load_pickle_obj)

from sensory_cokge.relative_embedding import (AnchorDescriptors,
                                              construct_relative_embeddings)

def main():
    anchor_descirptions = copy.deepcopy(AnchorDescriptors)
    parser = argparse.ArgumentParser()
    parser.add_argument('extended_name', type = str,
            help = 'The extended name in the training config.')

    parser.add_argument('--folder_name', type = str, default = './outputs',
            help = 'The folder that you want to sture these object.')

    args = parser.parse_args()

    hint_flag = False
    checked_files = ['finetuned-ALBERT-{0}_embeddings.pkl'.format(args.extended_name),
                     'finetuned-BERT-{0}_embeddings.pkl'.format(args.extended_name), 
                     'finetuned-BART-{0}_embeddings.pkl'.format(args.extended_name), 
                     'finetuned-GPT2-{0}_embeddings.pkl'.format(args.extended_name),
                     'finetuned-RoBERTa-{0}_embeddings.pkl'.format(args.extended_name),
                     'finetuned-T5-{0}_embeddings.pkl'.format(args.extended_name)]

    for fname in checked_files:
        if not os.path.isfile(os.path.join(args.folder_name, fname)):
            hint_flag = True

    if hint_flag:
        print('Please run embeddings_from_finetuned.py first.')
        sys.exit(0)

    for filename in checked_files:
        model_type = filename.replace('finetuned-', '')
        model_type = filename.split('-')[0]

        model_embeddings = load_pickle_obj(os.path.join(args.folder_name, filename))
        has_encoder, has_decoder = True, True
        for key in model_embeddings:
            if model_embeddings[key].get('encoder_embedding', None) is None:
                has_encoder = False

            if model_embeddings[key].get('decoder_embedding', None) is None:
                has_decoder = False

        if has_encoder:
            print('Start process relative embedding of finetuned {0} encoder ... '\
                    .format(model_type))

            model_embeddings = construct_relative_embeddings(model_type,
                                                             'encoder',
                                                             anchor_descirptions,
                                                             model_embeddings)

        if has_decoder:
            print('Start process relative embedding of finetuned {0} decoder ... '\
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


