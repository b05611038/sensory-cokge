import os
import sys
import argparse

from src.utils import load_pickle_obj
from src.metrics import EvaluationMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('condition', type = str,
            help = "The evaluated condition of the program, ['pretrained', 'finetuned'] is valid.")

    parser.add_argument('--folder_name', type = str, default = './outputs',
            help = 'The folder that you want to sture these object.')
    parser.add_argument('--finetune_model_extended_name', type = str, default = 'example',
            help = 'The extended name in the training config.')
    parser.add_argument('--layout_file', type = str, default = '',
            help = 'Layout the result as a csv file.')

    args = parser.parse_args()
    if args.condition not in ['pretrained', 'finetuned']:
        print("Condition must be 'pretrained' or 'finetuned'")
        sys.exit(0)

    folder_name = './outputs'
    if args.condition == 'pretrained':
        checked_files = {'graph': 'graph_properties.pkl',
                         'ALBERT': 'pretrained-ALBERT_embeddings.pkl',
                         'BART': 'pretrained-BART_embeddings.pkl',
                         'BERT': 'pretrained-BERT_embeddings.pkl',
                         'GPT2': 'pretrained-GPT2_embeddings.pkl',
                         'Gemma2': 'pretrained-Gemma2_embeddings.pkl',
                         'Llama3': 'pretrained-Llama3_embeddings.pkl',
                         'Qwen2': 'pretrained-Qwen2_embeddings.pkl',
                         'RoBERTa': 'pretrained-RoBERTa_embeddings.pkl',
                         'T5': 'pretrained-T5_embeddings.pkl'}

    elif args.condition == 'finetuned':
        extended_name = '-' + args.finetune_model_extended_name
        checked_files = {'graph': 'graph_properties.pkl',
                         'ALBERT': 'finetuned-ALBERT{0}_embeddings.pkl'.format(extended_name),
                         'BART': 'finetuned-BART{0}_embeddings.pkl'.format(extended_name),
                         'BERT': 'finetuned-BERT{0}_embeddings.pkl'.format(extended_name),
                         'GPT2': 'finetuned-GPT2{0}_embeddings.pkl'.format(extended_name),
                         'RoBERTa': 'finetuned-RoBERTa{0}_embeddings.pkl'.format(extended_name),
                         'T5': 'finetuned-T5{0}_embeddings.pkl'.format(extended_name)}

    graph_hint_flag, model_hint_flag = False, False
    for fname in checked_files:
        if not os.path.isfile(os.path.join(args.folder_name, checked_files[fname])):
            if fname == 'graph':
                graph_hint_flag = True
            else:
                model_hint_flag = True

    if graph_hint_flag:
        print('Please run embeddings_from_pretrained.py first.')
        sys.exit(0)

    if model_hint_flag:
        if args.condition == 'pretrained':
            print('Please run embeddings_from_pretrained.py first.')
            sys.exit(0)

        elif args.condition == 'finetuned':
            print('Please run embeddings_from_pretrained.py first.')
            print('Afterwards, run generate_finetuned_data.py first to generate training data.')
            print('Then, use finetune_[model]_by_sequence_classification.py to finetune LMs.')
            print('Finally, use embeddings_from_pretrained.py to acquire the needed files.')
            print('\nIf cannot detect one of embedding also show this message, can skip if result is not complete.')

    graph_properties = load_pickle_obj(os.path.join(args.folder_name, checked_files['graph']))
    albert_embeddings = checked_files.get('ALBERT', None)
    if albert_embeddings is not None:
        albert_embeddings = load_pickle_obj(os.path.join(args.folder_name, albert_embeddings)) 

    bert_embeddings = checked_files.get('BERT', None)
    if bert_embeddings is not None:
        bert_embeddings = load_pickle_obj(os.path.join(args.folder_name, bert_embeddings))

    bart_embeddings = checked_files.get('BART', None)
    if bart_embeddings is not None:
        bart_embeddings = load_pickle_obj(os.path.join(args.folder_name, bart_embeddings))

    gemma2_embeddings = checked_files.get('Gemma2', None)
    if gemma2_embeddings is not None:
        gemma2_embeddings = load_pickle_obj(os.path.join(args.folder_name, gemma2_embeddings))

    gpt2_embeddings = checked_files.get('GPT2', None)
    if gpt2_embeddings is not None:
        gpt2_embeddings = load_pickle_obj(os.path.join(args.folder_name, gpt2_embeddings))

    llama3_embeddings = checked_files.get('Llama3', None)
    if llama3_embeddings is not None:
        llama3_embeddings = load_pickle_obj(os.path.join(args.folder_name, llama3_embeddings))

    qwen2_embeddings = checked_files.get('Qwen2', None)
    if qwen2_embeddings is not None:
        qwen2_embeddings = load_pickle_obj(os.path.join(args.folder_name, qwen2_embeddings))

    roberta_embeddings = checked_files.get('RoBERTa', None)
    if roberta_embeddings is not None:
        roberta_embeddings = load_pickle_obj(os.path.join(args.folder_name, roberta_embeddings))

    t5_embeddings = checked_files.get('T5', None)
    if t5_embeddings is not None:
        t5_embeddings = load_pickle_obj(os.path.join(args.folder_name, t5_embeddings))

    result_dict = {}
    if albert_embeddings is not None:
        print('Start evaluate ALBERT model ...')
        metrics = EvaluationMetrics('ALBERT', 'encoder', albert_embeddings, graph_properties) 
        metrics.summary()
        result_dict['ALBERT'] = {'encoder': metrics.metrics}

    if bart_embeddings is not None:
        print('Start evaluate BART encoder ...')
        metrics = EvaluationMetrics('BART', 'encoder', bart_embeddings, graph_properties)
        metrics.summary()
        result_dict['BART'] = {'encoder': metrics.metrics}

        print('Start evaluate BART decoder ...')
        metrics = EvaluationMetrics('BART', 'decoder', bart_embeddings, graph_properties)
        metrics.summary()
        result_dict['BART']['decoder'] = metrics.metrics

    if bert_embeddings is not None:
        print('Start evaluate BERT model ...')
        metrics = EvaluationMetrics('BERT', 'encoder', bert_embeddings, graph_properties)
        metrics.summary()
        result_dict['BERT'] = {'encoder': metrics.metrics}

    if gemma2_embeddings is not None:
        print('Start evluate Gemma2 model ...')
        metrics = EvaluationMetrics('Gemma2', 'decoder', gemma2_embeddings, graph_properties)
        metrics.summary()
        result_dict['Gemma2'] = {'decoder': metrics.metrics}

    if gpt2_embeddings is not None:
        print('Start evalutate GPT-2 model ...')
        metrics = EvaluationMetrics('GPT2', 'decoder', gpt2_embeddings, graph_properties)
        metrics.summary()
        result_dict['GPT2'] = {'decoder': metrics.metrics}

    if llama3_embeddings is not None:
        print('Start evaluate LLAMA3 model ...')
        metrics = EvaluationMetrics('Llama3', 'decoder', llama3_embeddings, graph_properties)
        metrics.summary()
        result_dict['Llama3'] = {'decoder': metrics.metrics}

    if qwen2_embeddings is not None:
        print('Start evaluate Qwen2 model ...')
        metrics = EvaluationMetrics('Qwen2', 'decoder', qwen2_embeddings, graph_properties)
        metrics.summary()
        result_dict['Qwen2'] = {'decoder': metrics.metrics}

    if roberta_embeddings is not None:
        print('Start evalutate RoBERTa model ...')
        metrics = EvaluationMetrics('RoBERTa', 'encoder', roberta_embeddings, graph_properties)
        metrics.summary()
        result_dict['RoBERTa'] = {'encoder': metrics.metrics}

    if t5_embeddings is not None:
        print('Start evaluate T5 encoder ...')
        metrics = EvaluationMetrics('T5', 'encoder', t5_embeddings, graph_properties)
        metrics.summary()
        result_dict['T5'] = {'encoder': metrics.metrics}

        print('Start evaluate T5 decoder ...')
        metrics = EvaluationMetrics('T5', 'decoder', t5_embeddings, graph_properties)
        metrics.summary()
        result_dict['T5']['decoder'] = metrics.metrics

    item_names = ['adjacency_matching_l2', 'adjacency_matching_l2_NoRB', 'adjacency_matching_angle',
            'adjacency_matching_angle_NoRB', 'distances_matching_l2', 'distances_matching_l2_NoRB',
            'distances_matching_angle', 'distances_matching_angle']
    if len(args.layout_file):
        lines = []
        head_line = 'LM,structure,'
        for col in item_names:
            head_line += '{0},'.format(col)

        head_line = head_line[: -1]
        head_line += '\n'
        lines.append(head_line)

        for model in result_dict:
            for struc in result_dict[model]:
                single_result = result_dict[model][struc]
                if args.condition == 'pretrained':
                    single_line = '{0},{1},'.format('pretrained-' + model, struc)
                elif args.condition == 'finetuned':
                    single_line = '{0},{1},'.format(model + '-{0}'.format(args.finetune_model_extended_name),
                                                    struc)
                for col in item_names:
                    single_line += '{0},'.format(single_result[col])

                single_line = single_line[: -1]
                single_line += '\n'
                lines.append(single_line)

        filename = args.layout_file
        if not filename.endswith('.csv'):
            filename += '.csv'

        with open(filename, 'w') as f:
            f.writelines(lines)
            f.close()

        print('Successfully layout evaluated results as file:{0}'.format(filename))

    print('Program finish.')

    return None

if __name__ == '__main__':
    main()


