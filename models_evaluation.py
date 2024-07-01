import os
import sys

from src.utils import load_pickle_obj
from src.metrics import EvaluationMetrics

def main():
    if len(sys.argv) != 2:
       print('Usage python3 models_evaluation.py [condition]')
       sys.exit(0)

    condition = sys.argv[1]
    if condition.lower() not in ['pretrained', 'finetuned']:
        print("Condition must be 'pretrained' or 'finetuned'")
        sys.exit(0)

    folder_name = './outputs'
    if condition.lower() == 'pretrained':
        checked_files = {'graph': 'graph_properties.pkl',
                         'BERT': 'BERT_embeddings.pkl',
                         'BART': 'BART_embeddings.pkl',
                         'GPT2': 'GPT2_embeddings.pkl'}
    elif condition.lower() == 'finetuned':
        checked_files = {'graph': 'graph_properties.pkl',
                         'BERT': 'finetuned-BERT_embeddings.pkl',
                         'BART': 'finetuned-BART_embeddings.pkl',
                         'GPT2': 'finetuned-GPT2_embeddings.pkl'}

    hint_flag = False
    for fname in checked_files:
        if not os.path.isfile(os.path.join(folder_name, checked_files[fname])):
            hint_flag = True

    if hint_flag:
        if condition.lower() == 'pretrained':
            print('Please run embeddings_from_pretrained.py first.')
        elif condition.lower() == 'finetuned':
            print('Please run embeddings_from_pretrained.py first.')
            print('Afterwards, run generate_finetuned_data.py first to generate training data.')
            print('Then, use finetune_[model]_by_sequence_classification.py to finetune LMs.')
            print('Finally, use embeddings_from_pretrained.py to acquire the needed files.')

        sys.exit(0)

    graph_properties = load_pickle_obj(os.path.join(folder_name, checked_files['graph']))
    bert_embeddings = load_pickle_obj(os.path.join(folder_name, checked_files['BERT']))
    bart_embeddings = load_pickle_obj(os.path.join(folder_name, checked_files['BART']))
    gpt_embeddings = load_pickle_obj(os.path.join(folder_name, checked_files['GPT2']))

    print('Start evaluate BERT model ...')
    metrics = EvaluationMetrics('BERT', 'encoder', bert_embeddings, graph_properties)
    metrics.summary()

    print('Start evaluate BART encoder ...')
    metrics = EvaluationMetrics('BART', 'encoder', bart_embeddings, graph_properties)
    metrics.summary()

    print('Start evaluate BART decoder ...')
    metrics = EvaluationMetrics('BART', 'decoder', bart_embeddings, graph_properties)
    metrics.summary()

    print('Start evalutate GPT-2 model ...')
    metrics = EvaluationMetrics('GPT-2', 'decoder', gpt_embeddings, graph_properties)
    metrics.summary()

    print('Program finish.')
    return None

if __name__ == '__main__':
    main()


