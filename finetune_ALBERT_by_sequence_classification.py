import os
import sys
import argparse

import pandas as pd

import torch
from datasets import Dataset
from transformers import (AlbertTokenizer,
                          AlbertForSequenceClassification,
                          Trainer,
                          TrainingArguments)

from src.models import ALBERT_NAME
from src.utils import parse_training_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_argument', type = str,
            help = 'Path of the finetuning arguments')

    parser.add_argument('--model_select', type = str, default = ALBERT_NAME,
            help = 'The BERT version you want to choose.')
    parser.add_argument('--model_name', type = str, default = 'finetuned-' + ALBERT_NAME,
            help = 'The saved name of the finetuned model.')
    parser.add_argument('--device', type = str, default = 'auto',
            help = 'Select the computing device.')
    parser.add_argument('--save_sequence_classification', action = 'store_true',
            help = 'Save the ModelForSequenceClassification object.')

    args = parser.parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda' or args.device == 'cpu':
        device = torch.device(args.device)
    else:
        raise ValueError('Device: {0} is not a valid argument.'.format(args.device))

    print('Setup device done. Using device: {0}'.format(device))
    print('Load training arguments in the config file ...')

    training_args = parse_training_args(args.training_argument)
    # I/O
    folder_name = training_args.get('model_folder', None)
    if folder_name is None:
        print('Please set the argument::model_folder in config file.')
        sys.exit(0)

    saved_model_name = args.model_name
    if training_args.get('extended_name', None) is not None:
        saved_model_name = saved_model_name + '-' + training_args.get('extended_name', '')

    log_folder = os.path.join(folder_name, saved_model_name + '-log')

    train_data_path = training_args.get('train_csv', None)
    eval_data_path = training_args.get('eval_csv', None)
    if train_data_path is None or eval_data_path is None:
        print('Please set argument::train_csv and argument::eval_csv in config file.')
        sys.exit(0)

    # fintuned arguments
    training_epoch = training_args.get('epochs', None)
    if training_epoch is None:
        print('Please set argument::epochs in config file.')
        sys.exit(0)

    per_device_batch_size = training_args.get('per_device_batch_size', 16)
    base_lr = training_args.get('base_lr', 1e-5)
    weight_decay = training_args.get('weight_decay', 0.01)
    cuda_amp = training_args.get('cuda_amp', False)
    if device.type == 'cpu':
        cuda_amp = False

    print('Load finetuned data ...')
    if (not os.path.isfile(train_data_path)) or (not os.path.join(eval_data_path)):
        print('Please run generate_finetuned_data.py first and verify filename is correct.')
        sys.exit(0)

    train_df = pd.read_csv(train_data_path)
    eval_df = pd.read_csv(eval_data_path)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    print('Start process dataset ...')

    tokenizer = AlbertTokenizer.from_pretrained(args.model_select)
    def preprocess_function(examples):
        inputs = [f"{examples['text0'][i]} [SEP] {examples['text1'][i]} [SEP] " + \
                  f"{examples['text2'][i]} [SEP] {examples['text3'][i]} [SEP] " + \
                  f"{examples['text4'][i]}" for i in range(len(examples['text0']))]

        tokenized_inputs = tokenizer(inputs, truncation = True, padding = True)

        tokenized_inputs['labels'] = torch.tensor(examples['ground_truth'],
                dtype = torch.int32)

        return tokenized_inputs

    train_dataset = train_dataset.map(preprocess_function, batched = True)
    eval_dataset = eval_dataset.map(preprocess_function, batched = True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    print('Dataset prepare done.')
    print('Load pre-trained ALBERT ...')
    model = AlbertForSequenceClassification.from_pretrained(args.model_select,
                                                            num_labels = 5).to(device)

    print('Start finetune ALBERT ...')
    training_args = TrainingArguments(output_dir = folder_name,
                                      eval_strategy = 'epoch',
                                      learning_rate = base_lr,
                                      per_device_train_batch_size = per_device_batch_size,
                                      per_device_eval_batch_size = per_device_batch_size,
                                      num_train_epochs = training_epoch,
                                      weight_decay = weight_decay,
                                      logging_dir = log_folder,
                                      logging_steps = 10,
                                      save_steps = 10000,
                                      save_total_limit = 2,
                                      report_to = 'tensorboard',
                                      fp16 = cuda_amp)

    trainer = Trainer(model = model,
                      args = training_args,
                      train_dataset = train_dataset,
                      eval_dataset = eval_dataset,
                      tokenizer = tokenizer)

    trainer.train()

    print('Start saving model ...')
    model = model.cpu()
    if args.save_sequence_classification:
        task_model_path = os.path.join(folder_name, saved_model_name + '-sequence-classification')
        print('Task-specific model is saved at {0}'.format(task_model_path))
        tokenizer.save_pretrained(task_model_path)
        model.save_pretrained(task_model_path)

    base_model_path = os.path.join(folder_name, saved_model_name)
    print('Basic model is saved at {0}'.format(base_model_path))
    tokenizer.save_pretrained(base_model_path)
    model.albert.save_pretrained(base_model_path)

    print('Finish ALBERT fine-tuning.')
    return None

if __name__ == '__main__':
    main()


