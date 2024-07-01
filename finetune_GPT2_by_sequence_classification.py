import os
import sys

import pandas as pd

import torch
from datasets import Dataset
from transformers import (GPT2Tokenizer,
                          GPT2ForSequenceClassification,
                          Trainer,
                          TrainingArguments)

from src.models import GPT_NAME

def main():
    # path info
    data_folder = './outputs'
    folder_name = './finetuned_models'
    saved_model_name = 'finetuned-gpt2'
    log_folder = os.path.join(folder_name, saved_model_name + '-log')

    # training info
    training_epoch = 5
    per_device_batch_size = 16
    base_lr = 1e-5
    weight_decay = 0.01
    cuda_amp = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Setup device done. Using device: {0}'.format(device))

    print('Load stored data ...')
    train_data_path = os.path.join(data_folder, 'train.csv')
    eval_data_path = os.path.join(data_folder, 'eval.csv')
    if (not os.path.isfile(train_data_path)) or (not os.path.join(eval_data_path)):
        print('Please run generate_finetuned_data.py first.')
        sys.exit(0)

    train_df = pd.read_csv('./outputs/train.csv')
    eval_df = pd.read_csv('./outputs/eval.csv')
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    print('Start process dataset ...')

    tokenizer = GPT2Tokenizer.from_pretrained(GPT_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    def preprocess_function(examples):
        inputs = [f"{examples['text0'][i]} {tokenizer.eos_token}" + \
                  f"{examples['text1'][i]} {tokenizer.eos_token}" + \
                  f"{examples['text2'][i]} {tokenizer.eos_token}" + \
                  f"{examples['text3'][i]} {tokenizer.eos_token}" + \
                  f"{examples['text4'][i]} {tokenizer.eos_token}" for i in range(len(examples['text0']))]

        tokenized_inputs = tokenizer(inputs, truncation = True, 
                padding = 'max_length', max_length = 512)

        tokenized_inputs['labels'] = examples['ground_truth']
        return tokenized_inputs

    train_dataset = train_dataset.map(preprocess_function, batched = True)
    eval_dataset = eval_dataset.map(preprocess_function, batched = True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    print('Dataset prepare done.')
    print('Load pre-trained GPT-2 ...')

    model = GPT2ForSequenceClassification.from_pretrained(GPT_NAME, 
                                                          num_labels = 5).to(device)
    model.config.pad_token_id = 50256

    print('Start finetune GPT-2 ...')
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
    task_model_path = os.path.join(folder_name, saved_model_name + '-sequence-classification')
    print('Task-specific model is saved at {0}'.format(task_model_path))
    tokenizer.save_pretrained(task_model_path)
    model.save_pretrained(task_model_path)

    base_model_path = os.path.join(folder_name, saved_model_name)
    print('Basic model is saved at {0}'.format(base_model_path))
    tokenizer.save_pretrained(base_model_path)
    model.transformer.save_pretrained(base_model_path)

    print('Finish GPT-2 fine-tuning.')

    return None

if __name__ == '__main__':
    main()


