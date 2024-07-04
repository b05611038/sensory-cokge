import os
import argparse

from src.finetune import generate_finetune_data

def layout_data(data, filename):
    assert len(data) > 0
    number_of_selection = len(data[0]['selections'])

    header = ''
    for idx in range(number_of_selection):
        header += 'text{0},'.format(idx)

    header += 'ground_truth\n'
    lines = [header]
    for single_data in data:
        single_line = ''
        for single_text in single_data['selections']:
            single_line += '"'
            single_line += single_text
            single_line += '",'

        single_line += '{0}\n'.format(single_data['ground_truth'])
        lines.append(single_line)

    if not filename.endswith('.csv'):
        filename += '.csv'

    with open(filename, 'w') as f:
        f.writelines(lines)
        f.close()

    print('Successfully layout file: {0}.'.format(filename))
    return None
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_sample_number', type = int, default = 50000,
            help = 'The training sample number generated in the csv file.')
    parser.add_argument('--eval_sample_number', type = int, default = 10000,
            help = 'The evaluating sample number generated in the csv file.')
    parser.add_argument('--train_csv_name', type = str, default = './outputs/train.csv',
            help = 'The saved training csv.')
    parser.add_argument('--eval_csv_name', type = str, default = './outputs/eval.csv',
            help = 'The saved eval csv.')

    args = parser.parse_args()
    data_number = {'train': args.train_sample_number, 
                   'eval': args.eval_sample_number}

    filenames = {'train': args.train_csv_name,
                 'eval': args.eval_csv_name}

    generated_data = generate_finetune_data(data_number = data_number)
    for set_name in generated_data:
        layout_data(generated_data[set_name], filenames[set_name])

    print('Program finish.')

    return None

if __name__ == '__main__':
    main()


