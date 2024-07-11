import os
import argparse

import torch

from src.utils import (init_directory,
                       save_object)

from src.graph import graph_properties
from src.models import (ALBERT_embeddings,
                        ALBERT_NAME,
                        BERT_embeddings,
                        BERT_NAME,
                        BART_embeddings,
                        BART_NAME,
                        Gemma2_embeddings,
                        GEMMA2_NAME,
                        GPT2_embeddings,
                        GPT2_NAME,
                        Llama3_embeddings,
                        LLAMA3_NAME,
                        Qwen2_embeddings,
                        QWEN2_NAME,
                        RoBERTa_embeddings,
                        RoBERTa_NAME,
                        T5_embeddings,
                        T5_NAME)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_name', type = str, default = './outputs',
            help = 'The folder that you want to sture these object.')
    parser.add_argument('--ALBERT_select', type = str, default = ALBERT_NAME,
            help = 'The model config of evaluated pretrained ALBERT.')
    parser.add_argument('--BERT_select', type = str, default = BERT_NAME,
            help = 'The model config of evaluated pretrained BERT.')
    parser.add_argument('--BART_select', type = str, default = BART_NAME,
            help = 'The model config of evaluated pretrained BART.')
    parser.add_argument('--Gemma2_select', type = str, default = GEMMA2_NAME,
            help = 'The model config of evaluated pretrained Gemma2.')
    parser.add_argument('--GPT2_select', type = str, default = GPT2_NAME,
            help = 'The model config of evaluated pretrained GPT2.')
    parser.add_argument('--Llama3_select', type = str, default = LLAMA3_NAME,
            help = 'The model config of evaluated pretrained Llama3.')
    parser.add_argument('--Qwen2_select', type = str, default = QWEN2_NAME,
            help = 'The model config of evaluated pretrained Qwen2.')
    parser.add_argument('--RoBERTa_select', type = str, default = RoBERTa_NAME,
            help = 'The model config of evaluated pretrained RoBERTa.')
    parser.add_argument('--T5_select',  type = str, default = T5_NAME,
            help = 'The model config of evaluated pretrained T5.')
    parser.add_argument('--batch_size', type = int, default = 4,
            help = 'The mini-batch size you want to use when inference.')
    parser.add_argument('--batch_size_LLM', type = int, default = 1,
            help = 'The mini-batch size for gigantic LM you want to use when inference.')
    parser.add_argument('--device', type = str, default = 'auto',
            help = 'Select the computing device.')
    parser.add_argument('--enable_LLM', action = 'store_true',
            help = 'To determine calculte gigantic LM results.')

    args = parser.parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda' or args.device == 'cpu':
        device = torch.device(args.device)
    else:
        raise ValueError('Device: {0} is not a valid argument.'.format(args.device))

    init_directory(args.folder_name)

    print('Extract graph structure from CoffeeDatabase ...')
    graph_attr = graph_properties()
    descriptions = graph_attr['descriptions']
    save_object(os.path.join(args.folder_name, 'graph_properties.pkl'), graph_attr)

    print('Process ALBERT embeddings ...')
    albert_embeddings = ALBERT_embeddings(descriptions,
                                          device = device,
                                          batch_size = args.batch_size,
                                          pretrained_model = args.ALBERT_select)

    save_object(os.path.join(args.folder_name, 'pretrained-ALBERT_embeddings.pkl'), albert_embeddings)

    print('Process BART embeddings ...')
    bart_embeddings = BART_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size,
                                      pretrained_model = args.BART_select)

    save_object(os.path.join(args.folder_name, 'pretrained-BART_embeddings.pkl'), bart_embeddings)

    print('Process BERT embeddings ...')
    bert_embeddings = BERT_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size,
                                      pretrained_model = args.BERT_select)

    save_object(os.path.join(args.folder_name, 'pretrained-BERT_embeddings.pkl'), bert_embeddings)

    print('Process GPT-2 embeddings ...')
    gpt2_embeddings = GPT2_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size,
                                      pretrained_model = args.GPT2_select)

    save_object(os.path.join(args.folder_name, 'pretrained-GPT2_embeddings.pkl'), gpt2_embeddings)

    print('Process RoBERTa embeddings ...')
    roberta_embeddings = RoBERTa_embeddings(descriptions,
                                            device = device,
                                            batch_size = args.batch_size,
                                            pretrained_model = args.RoBERTa_select)

    save_object(os.path.join(args.folder_name, 'pretrained-RoBERTa_embeddings.pkl'), roberta_embeddings)

    print('Process T5 embeddings ...')
    t5_embeddings = T5_embeddings(descriptions,
                                  device = device,
                                  batch_size = args.batch_size,
                                  pretrained_model = args.T5_select)

    save_object(os.path.join(args.folder_name, 'pretrained-T5_embeddings.pkl'), t5_embeddings)

    if args.enable_LLM:
        print('Process Gemma2 embeddings ...')
        gemma2_embeddings = Gemma2_embeddings(descriptions,
                                              device = device,
                                              batch_size = args.batch_size_LLM,
                                              pretrained_model = args.Gemma2_select)

        save_object(os.path.join(args.folder_name, 'pretrained-Gemma2_embeddings.pkl'), gemma2_embeddings)

        print('Process Llama3 embeddings ...')
        llama3_embeddings = Llama3_embeddings(descriptions,
                                              device = device,
                                              batch_size = args.batch_size_LLM,
                                              pretrained_model = args.Llama3_select)

        save_object(os.path.join(args.folder_name, 'pretrained-Llama3_embeddings.pkl'), llama3_embeddings)

        print('Process Qwen2 embeddings ...')
        qwen2_embeddings = Qwen2_embeddings(descriptions,
                                            device = device,
                                            batch_size = args.batch_size_LLM,
                                            pretrained_model = args.Qwen2_select)

        save_object(os.path.join(args.folder_name, 'pretrained-Qwen2_embeddings.pkl'), qwen2_embeddings)

    print('All propeties extracted from graph and LLMs done.')

    return None

if __name__ == '__main__':
    main()


