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
                         'ALBERT': 'ALBERT_embeddings.pkl',
                         'BERT': 'BERT_embeddings.pkl',
                         'BART': 'BART_embeddings.pkl',
                         'GPT2': 'GPT2_embeddings.pkl',
                         'RoBERTa': 'RoBERTa_embeddings.pkl',
                         'T5': 'T5_embeddings.pkl'}

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
    albert_embeddings = checked_files.get('ALBERT', None)
    if albert_embeddings is not None:
        albert_embeddings = load_pickle_obj(os.path.join(folder_name, albert_embeddings)) 

    bert_embeddings = checked_files.get('BERT', None)
    if bert_embeddings is not None:
        bert_embeddings = load_pickle_obj(os.path.join(folder_name, bert_embeddings))

    bart_embeddings = checked_files.get('BART', None)
    if bart_embeddings is not None:
        bart_embeddings = load_pickle_obj(os.path.join(folder_name, bart_embeddings))

    gpt2_embeddings = checked_files.get('GPT2', None)
    if gpt2_embeddings is not None:
        gpt2_embeddings = load_pickle_obj(os.path.join(folder_name, gpt2_embeddings))

    roberta_embeddings = checked_files.get('RoBERTa', None)
    if roberta_embeddings is not None:
        roberta_embeddings = load_pickle_obj(os.path.join(folder_name, roberta_embeddings))

    t5_embeddings = checked_files.get('T5', None)
    if t5_embeddings is not None:
        t5_embeddings = load_pickle_obj(os.path.join(folder_name, t5_embeddings))

    if albert_embeddings is not None:
        print('Start evaluate ALBERT model ...')
        metrics = EvaluationMetrics('ALBERT', 'encoder', albert_embeddings, graph_properties) 
        metrics.summary()

    if bert_embeddings is not None:
        print('Start evaluate BERT model ...')
        metrics = EvaluationMetrics('BERT', 'encoder', bert_embeddings, graph_properties)
        metrics.summary()

    if bart_embeddings is not None:
        print('Start evaluate BART encoder ...')
        metrics = EvaluationMetrics('BART', 'encoder', bart_embeddings, graph_properties)
        metrics.summary()

        print('Start evaluate BART decoder ...')
        metrics = EvaluationMetrics('BART', 'decoder', bart_embeddings, graph_properties)
        metrics.summary()

    if gpt2_embeddings is not None:
        print('Start evalutate GPT-2 model ...')
        metrics = EvaluationMetrics('GPT2', 'decoder', gpt2_embeddings, graph_properties)
        metrics.summary()

    if roberta_embeddings is not None:
        print('Start evalutate RoBERTa model ...')
        metrics = EvaluationMetrics('RoBERTa', 'encoder', roberta_embeddings, graph_properties)
        metrics.summary()

    if t5_embeddings is not None:
        print('Start evaluate T5 encoder ...')
        metrics = EvaluationMetrics('T5', 'encoder', t5_embeddings, graph_properties)
        metrics.summary()

        print('Start evaluate T5 decoder ...')
        metrics = EvaluationMetrics('T5', 'decoder', t5_embeddings, graph_properties)
        metrics.summary()

    print('Program finish.')
    return None

if __name__ == '__main__':
    main()


