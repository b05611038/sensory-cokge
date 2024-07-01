import os

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (BertTokenizer, 
                          BertModel,
                          BartTokenizer, 
                          BartModel,
                          GPT2Tokenizer, 
                          GPT2Model)

from .utils import use_a_or_an

__all__ = ['BERT_embeddings', 'BART_embeddings', 'GPT2_embeddings',
        'BERT_NAME', 'BART_NAME', 'GPT_NAME']

CONTEXT = 'This cup of coffee has {0} {1} flavor.'
TOKEN_ARGS = {
        'return_tensors': 'pt',
        'padding': 'max_length',
        'truncation': True,
        'max_length': 64,
}

BERT_NAME = 'bert-base-uncased'
BART_NAME = 'facebook/bart-base'
GPT_NAME = 'gpt2'

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, 
            tokenizer_arguments = TOKEN_ARGS):

        self.texts = texts
        self.tokenizer = tokenizer
        self.tokenizer_arguments = tokenizer_arguments

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], **self.tokenizer_arguments) 


def BERT_embeddings(descriptions, 
        batch_size = 4, 
        device = torch.device('cpu'),
        context = CONTEXT,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
        model = BertModel.from_pretrained(BERT_NAME)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = BertTokenizer.from_pretrained(finetuned_model)
        model = BertModel.from_pretrained(finetuned_model)

    model = model.to(device)

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    running_idx = 0
    for batch_data in dataloader:
        input_ids = batch_data['input_ids'].squeeze(1).to(device)
        attention_mask = batch_data['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            outputs = model(input_ids = input_ids, 
                            attention_mask = attention_mask)

            last_token_indices = attention_mask.sum(dim = 1) - 1
            last_hidden_states = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                last_embedding = last_hidden_states[mini_idx, seq_length, :].cpu()
                embeddings[running_idx]['encoder_embedding'] = last_embedding
                running_idx += 1

    return embeddings

def BART_embeddings(descriptions,
        batch_size = 4,
        device = torch.device('cpu'),
        context = CONTEXT,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = BartTokenizer.from_pretrained(BART_NAME)
        model = BartModel.from_pretrained(BART_NAME)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = BartTokenizer.from_pretrained(finetuned_model)
        model = BartModel.from_pretrained(finetuned_model)

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    running_idx = 0
    for batch_data in dataloader:
        input_ids = batch_data['input_ids'].squeeze(1).to(device)
        attention_mask = batch_data['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask)

            last_token_indices = attention_mask.sum(dim = 1) - 1
            encoder_last_hidden_state = outputs.encoder_last_hidden_state
            decoder_last_hidden_state = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                last_encoder_embedding = encoder_last_hidden_state[mini_idx, seq_length, :].cpu()
                last_decoder_embedding = decoder_last_hidden_state[mini_idx, seq_length, :].cpu()
                embeddings[running_idx]['encoder_embedding'] = last_encoder_embedding
                embeddings[running_idx]['decoder_embedding'] = last_decoder_embedding
                running_idx += 1

    return embeddings

def GPT2_embeddings(descriptions,
        batch_size = 4,
        device = torch.device('cpu'),
        context = CONTEXT,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = GPT2Tokenizer.from_pretrained(GPT_NAME)
        model = GPT2Model.from_pretrained(GPT_NAME)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model)
        model = GPT2Model.from_pretrained(finetuned_model)

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    tokenizer.pad_token = tokenizer.eos_token

    running_idx = 0
    for batch_data in dataloader:
        input_ids = batch_data['input_ids'].squeeze(1).to(device)
        attention_mask = batch_data['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask)

            last_token_indices = attention_mask.sum(dim = 1) - 1
            last_hidden_state = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                last_decoder_embedding = last_hidden_state[mini_idx, seq_length, :].cpu()
                embeddings[running_idx]['decoder_embedding'] = last_decoder_embedding
                running_idx += 1

    return embeddings


