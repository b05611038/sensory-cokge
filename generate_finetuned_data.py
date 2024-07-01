import os

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
    folder_name = 'outputs'
    data_number = {'train': 50000, 'eval': 10000}
    generated_data = generate_finetune_data(data_number = data_number)
    for set_name in generated_data:
        layout_data(generated_data[set_name], os.path.join(folder_name, set_name + '.csv'))

    print('Program finish.')

    return None

if __name__ == '__main__':
    main()


