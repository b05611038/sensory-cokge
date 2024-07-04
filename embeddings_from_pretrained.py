import os
import argparse

import torch

from src.utils import (init_directory,
                       save_object)

from src.graph import graph_properties
from src.models import (ALBERT_embeddings,
                        BERT_embeddings,
                        BART_embeddings,
                        GPT2_embeddings,
                        RoBERTa_embeddings,
                        T5_embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type = str, default = './outputs',
        help = 'The folder that you want to sture these object.')
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

    print('Extract graph structure from CoffeeDatabase ...')
    graph_attr = graph_properties()
    descriptions = graph_attr['descriptions']
    save_object(os.path.join(args.folder_name, 'graph_properties.pkl'), graph_attr)

    print('Process ALBERT embeddings ...')
    albert_embeddings = ALBERT_embeddings(descriptions,
                                          device = device,
                                          batch_size = args.batch_size)

    save_object(os.path.join(args.folder_name, 'ALBERT_embeddings.pkl'), albert_embeddings)

    print('Process BERT embeddings ...')
    bert_embeddings = BERT_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size)

    save_object(os.path.join(args.folder_name, 'BERT_embeddings.pkl'), bert_embeddings)

    print('Process BART embeddings ...')
    bart_embeddings = BART_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size)

    save_object(os.path.join(args.folder_name, 'BART_embeddings.pkl'), bart_embeddings)

    print('Process GPT-2 embeddings ...')
    gpt2_embeddings = GPT2_embeddings(descriptions,
                                      device = device,
                                      batch_size = args.batch_size)

    save_object(os.path.join(args.folder_name, 'GPT2_embeddings.pkl'), gpt2_embeddings)

    print('Process RoBERTa embeddings ...')
    roberta_embeddings = RoBERTa_embeddings(descriptions,
                                            device = device,
                                            batch_size = args.batch_size)

    save_object(os.path.join(args.folder_name, 'RoBERTa_embeddings.pkl'), roberta_embeddings)

    print('Process T5 embeddings ...')
    t5_embeddings = T5_embeddings(descriptions,
                                  device = device,
                                  batch_size = args.batch_size)

    save_object(os.path.join(args.folder_name, 'T5_embeddings.pkl'), t5_embeddings)

    print('All propeties extracted from graph and LLMs done.')

    return None

if __name__ == '__main__':
    main()


