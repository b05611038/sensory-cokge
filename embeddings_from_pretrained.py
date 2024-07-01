import os

from src.utils import (init_directory,
                       save_object)

from src.graph import graph_properties
from src.models import (BERT_embeddings,
                        BART_embeddings,
                        GPT2_embeddings)

def main():
    folder_name = './outputs'
    init_directory(folder_name)

    print('Extract graph structure from CoffeeDatabase ...')
    graph_attr = graph_properties()
    descriptions = graph_attr['descriptions']
    save_object(os.path.join(folder_name, 'graph_properties.pkl'), graph_attr)

    print('Process BERT embeddings ...')
    bert_embeddings = BERT_embeddings(descriptions)
    save_object(os.path.join(folder_name, 'BERT_embeddings.pkl'), bert_embeddings)

    print('Process BART embeddings ...')
    bart_embeddings = BART_embeddings(descriptions)
    save_object(os.path.join(folder_name, 'BART_embeddings.pkl'), bart_embeddings)

    print('Process GPT-2 embeddings ...')
    gpt_embeddings = GPT2_embeddings(descriptions)
    save_object(os.path.join(folder_name, 'GPT2_embeddings.pkl'), gpt_embeddings)

    print('All propeties extracted from graph and LLMs done.')
    return None

if __name__ == '__main__':
    main()


