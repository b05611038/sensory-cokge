import os
import sys
import argparse

import torch

from src.utils import (init_directory,
                       load_pickle_obj,
                       save_object)

from src.models import (ALBERT_embeddings,
                        ALBERT_NAME,
                        BERT_embeddings,
                        BERT_NAME,
                        BART_embeddings,
                        BART_NAME,
                        GPT2_embeddings,
                        GPT2_NAME,
                        RoBERTa_embeddings,
                        RoBERTa_NAME,
                        T5_embeddings,
                        T5_NAME)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('extended_name', type = str,
            help = 'The extended name in the training config.')

    parser.add_argument('--folder_name', type = str, default = './outputs',
            help = 'The folder that you want to sture these object.')
    parser.add_argument('--model_folder', type = str, default = 'finetuned_models',
            help = 'The folder storing the finetuned LMs.')
    parser.add_argument('--ALBERT_select', type = str, default = ALBERT_NAME,
            help = 'The model config of evaluated finetuned ALBERT.')
    parser.add_argument('--BERT_select', type = str, default = BERT_NAME,
            help = 'The model config of evaluated finetuned BERT.')
    parser.add_argument('--BART_select', type = str, default = BART_NAME.split('/')[-1],
            help = 'The model config of evaluated finetuned BART.')
    parser.add_argument('--GPT2_select', type = str, default = GPT2_NAME,
            help = 'The model config of evaluated finetuned GPT2.')
    parser.add_argument('--RoBERTa_select', type = str, default = RoBERTa_NAME,
            help = 'The model config of evaluated finetuned RoBERTa.')
    parser.add_argument('--T5_select',  type = str, default = T5_NAME,
            help = 'The model config of evaluated finetuned T5.')
    parser.add_argument('--batch_size', type = int, default = 4,
            help = 'The mini-batch size you want to use when inference.')
    parser.add_argument('--device', type = str, default = 'auto',
            help = 'Select the computing device.')

    args = parser.parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda' or args.device == 'cpu':
        device = torch.device(args.device)
    else:
        raise ValueError('Device: {0} is not a valid argument.'.format(args.device))

    init_directory(args.folder_name)
    if not os.path.isfile(os.path.join(args.folder_name, 'graph_properties.pkl')):
        print('Please run embeddings_from_pretrained.py first.')
        sys.exit(0)

    print('Load the saved graph properties in {0}'.format(args.folder_name))
    graph_attr = load_pickle_obj(os.path.join(args.folder_name, 'graph_properties.pkl'))
    descriptions = graph_attr['descriptions']

    head = 'finetuned-'
    extend_name = '-' + args.extended_name
    finetuned_ALBERT = os.path.join(args.model_folder, head + args.ALBERT_select + extend_name)
    ALBERT_embedding_name = os.path.join(args.folder_name, 
                                         'finetuned-ALBERT-' + args.extended_name + '_embeddings.pkl')
    finetuned_BART = os.path.join(args.model_folder, head + args.BART_select + extend_name)
    BART_embedding_name = os.path.join(args.folder_name, 
                                       'finetuned-BART-' + args.extended_name + '_embeddings.pkl')
    finetuned_BERT = os.path.join(args.model_folder, head + args.BERT_select + extend_name)
    BERT_embedding_name = os.path.join(args.folder_name, 
                                       'finetuned-BERT-' + args.extended_name + '_embeddings.pkl')
    finetuned_GPT2 = os.path.join(args.model_folder, head + args.GPT2_select + extend_name)
    GPT2_embedding_name = os.path.join(args.folder_name, 
                                       'finetuned-GPT2-' + args.extended_name + '_embeddings.pkl')
    finetuned_RoBERTa = os.path.join(args.model_folder, head + args.RoBERTa_select + extend_name)
    RoBERTa_embedding_name = os.path.join(args.folder_name, 
                                          'finetuned-RoBERTa-' + args.extended_name + '_embeddings.pkl')
    finetuned_T5 = os.path.join(args.model_folder, head + args.T5_select + extend_name)
    T5_embedding_name = os.path.join(args.folder_name, 
                                     'finetuned-T5-' + args.extended_name + '_embeddings.pkl')

    if os.path.isdir(finetuned_ALBERT):
        print('Process finetuned ALBERT embeddings ...')
        albert_embeddings = ALBERT_embeddings(descriptions,
                                              device = device,
                                              batch_size = args.batch_size,
                                              finetuned_model = finetuned_ALBERT)

        save_object(ALBERT_embedding_name, albert_embeddings)
    else:
        print('Cannot find the finetuned ALBERT model.')

    if os.path.isdir(finetuned_BART):
        print('Process finetuned BART embeddings ...')
        bart_embeddings = BART_embeddings(descriptions,
                                          device = device,
                                          batch_size = args.batch_size,
                                          finetuned_model = finetuned_BART)

        save_object(BART_embedding_name, bart_embeddings)
    else:
        print('Cannot find the finetuned BART model.')

    if os.path.isdir(finetuned_BERT):
        print('Process finetuned BERT embeddings ...')
        bert_embeddings = BERT_embeddings(descriptions,
                                          device = device,
                                          batch_size = args.batch_size,
                                          finetuned_model = finetuned_BERT)
 
        save_object(BERT_embedding_name, bert_embeddings)
    else:
        print('Cannot find the finetuned BERT model.')

    if os.path.isdir(finetuned_GPT2):
        print('Process finetuned-GPT2 embeddings ...')
        gpt2_embeddings = GPT2_embeddings(descriptions,
                                          device = device,
                                          batch_size = args.batch_size,
                                          finetuned_model = finetuned_GPT2)
 
        save_object(GPT2_embedding_name, gpt2_embeddings)
    else:
        print('Cannot find the finetuned GPT2 model.')

    if os.path.isdir(finetuned_RoBERTa):
        print('Process finetuned-RoBERTa embeddings ...')
        roberta_embeddings = RoBERTa_embeddings(descriptions,
                                                device = device,
                                                batch_size = args.batch_size,
                                                finetuned_model = finetuned_RoBERTa)

        save_object(RoBERTa_embedding_name, roberta_embeddings)
    else:
        print('Cannot find the finetuned RoBERTa model.')

    if os.path.isdir(finetuned_T5):
        print('Process T5 embeddings ...')
        t5_embeddings = T5_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size,
                                      finetuned_model = finetuned_T5)

        save_object(T5_embedding_name, t5_embeddings)
    else:
        print('Cannot find the finetuned T5 model.')

    print('All propeties extracted from finetuned LLMs done.')
    return None

if __name__ == '__main__':
    main()


