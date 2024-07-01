import os
import sys

from src.utils import (init_directory,
                       load_pickle_obj,
                       save_object)

from src.models import (BERT_embeddings,
                        BART_embeddings,
                        GPT2_embeddings)

def main():
    folder_name = './outputs'
    model_folder = 'finetuned_models'
    finetuned_BERT = os.path.join(model_folder, 'finetuned-bert-base')
    finetuned_BART = os.path.join(model_folder, 'finetuned-bart-base')
    finetuned_GPT2 = os.path.join(model_folder, 'finetuned-gpt2')

    init_directory(folder_name)

    if not os.path.isfile(os.path.join(folder_name, 'graph_properties.pkl')):
        print('Please run embeddings_from_pretrained.py first.')
        sys.exit(0)

    print('Load the saved graph properties in {0}'.format(folder_name))
    graph_attr = load_pickle_obj(os.path.join(folder_name, 'graph_properties.pkl'))
    descriptions = graph_attr['descriptions']

    print('Process finetuned BERT embeddings ...')
    bert_embeddings = BERT_embeddings(descriptions, finetuned_model = finetuned_BERT)
    save_object(os.path.join(folder_name, 'finetuned-BERT_embeddings.pkl'), bert_embeddings)

    print('Process finetuned BART embeddings ...')
    bart_embeddings = BART_embeddings(descriptions, finetuned_model = finetuned_BART) 
    save_object(os.path.join(folder_name, 'finetuned-BART_embeddings.pkl'), bart_embeddings)

    print('Process finetuned-GPT-2 embeddings ...')
    gpt_embeddings = GPT2_embeddings(descriptions, finetuned_model = finetuned_GPT2)
    save_object(os.path.join(folder_name, 'finetuned-GPT2_embeddings.pkl'), gpt_embeddings)

    print('All propeties extracted from finetuned LLMs done.')
    return None

if __name__ == '__main__':
    main()


