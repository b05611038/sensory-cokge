import os

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (AlbertTokenizer,
                          AlbertModel,
                          AutoTokenizer,
                          BertTokenizer, 
                          BertModel,
                          BartTokenizer, 
                          BartModel,
                          Gemma2Model,
                          GPT2Tokenizer, 
                          GPT2Model,
                          LlamaModel,
                          Qwen2Model,
                          Qwen2Tokenizer,
                          RobertaTokenizer,
                          RobertaModel,
                          T5Tokenizer,
                          T5Model)

from .utils import use_a_or_an

__all__ = ['ALBERT_NAME', 'ALBERT_embeddings',
           'BERT_NAME', 'BERT_embeddings',
           'BART_NAME', 'BART_embeddings',
           'GEMMA2_NAME', 'Gemma2_embeddings',
           'GPT2_NAME', 'GPT2_embeddings',
           'LLAMA3_NAME', 'Llama3_embeddings',
           'QWEN2_NAME', 'Qwen2_embeddings',
           'RoBERTa_NAME', 'RoBERTa_embeddings',
           'T5_NAME', 'T5_embeddings']


CONTEXT = 'This cup of coffee has {0} {1} flavor.'
TOKEN_ARGS = {
        'return_tensors': 'pt',
        'padding': 'max_length',
        'truncation': True,
        'max_length': 64,
}

ALBERT_NAME = 'albert-base-v2'
BERT_NAME = 'bert-base-uncased'
BART_NAME = 'facebook/bart-base'
GEMMA2_NAME = '/shared_data/gemma-2-9b'
GPT2_NAME = 'gpt2'
LLAMA3_NAME = '/shared_data/Meta-Llama-3-8B'
QWEN2_NAME = '/shared_data/Qwen2-7B'
RoBERTa_NAME = 'roberta-base'
T5_NAME = 't5-base'


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


def ALBERT_embeddings(descriptions,
        batch_size = 4,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = ALBERT_NAME,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_model)
        model = AlbertModel.from_pretrained(pretrained_model)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = AlbertTokenizer.from_pretrained(finetuned_model)
        model = AlbertModel.from_pretrained(finetuned_model)

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


def BERT_embeddings(descriptions, 
        batch_size = 4, 
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = BERT_NAME,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        model = BertModel.from_pretrained(pretrained_model)
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
        pretrained_model = BART_NAME,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = BartTokenizer.from_pretrained(pretrained_model)
        model = BartModel.from_pretrained(pretrained_model)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = BartTokenizer.from_pretrained(finetuned_model)
        model = BartModel.from_pretrained(finetuned_model)

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


def Gemma2_embeddings(descriptions,
        batch_size = 1,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = GEMMA2_NAME):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = Gemma2Model.from_pretrained(pretrained_model)
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
            decoder_last_hidden_state = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                last_decoder_embedding = decoder_last_hidden_state[mini_idx, seq_length, :].cpu()
                embeddings[running_idx]['decoder_embedding'] = last_decoder_embedding
                running_idx += 1

    return embeddings


def GPT2_embeddings(descriptions,
        batch_size = 4,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = GPT2_NAME, 
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        model = GPT2Model.from_pretrained(pretrained_model)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model)
        model = GPT2Model.from_pretrained(finetuned_model)

    model = model.to(device)

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


def Llama3_embeddings(descriptions,
        batch_size = 1,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = LLAMA3_NAME):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model,
                                              legacy = False)

    tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})

    model = LlamaModel.from_pretrained(pretrained_model)
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
            decoder_last_hidden_state = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                last_decoder_embedding = decoder_last_hidden_state[mini_idx, seq_length, :].cpu()
                embeddings[running_idx]['decoder_embedding'] = last_decoder_embedding
                running_idx += 1

    return embeddings


def Qwen2_embeddings(descriptions,
        batch_size = 1,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = QWEN2_NAME):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    tokenizer = Qwen2Tokenizer.from_pretrained(pretrained_model)
    model = Qwen2Model.from_pretrained(pretrained_model)
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
            decoder_last_hidden_state = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                last_decoder_embedding = decoder_last_hidden_state[mini_idx, seq_length, :].cpu()
                embeddings[running_idx]['decoder_embedding'] = last_decoder_embedding
                running_idx += 1

    return embeddings


def RoBERTa_embeddings(descriptions,
        batch_size = 4,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = RoBERTa_NAME,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        model = RobertaModel.from_pretrained(pretrained_model)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = RobertaTokenizer.from_pretrained(finetuned_model)
        model = RobertaModel.from_pretrained(finetuned_model)

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


def T5_embeddings(descriptions,
        batch_size = 4,
        device = torch.device('cpu'),
        context = CONTEXT,
        pretrained_model = T5_NAME,
        finetuned_model = None):

    embeddings, texts = {}, []
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        text = context.format(use_a_or_an(description), description)
        texts.append(text)
        embeddings[des_idx] = {'description': description,
                               'text': text}

    if finetuned_model is None:
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model, legacy = False)
        model = T5Model.from_pretrained(pretrained_model)
    else:
        if not os.path.isdir(finetuned_model):
            raise OSError('Cannot find model: {0}'.format(finetuned_model))

        tokenizer = T5Tokenizer.from_pretrained(finetuned_model, legacy = False)
        model = T5Model.from_pretrained(finetuned_model)

    model = model.to(device)

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    running_idx = 0
    for batch_data in dataloader:
        input_ids = batch_data['input_ids'].squeeze(1).to(device)
        attention_mask = batch_data['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            decoder_input_id = tokenizer.encode('summarize:',
                                                return_tensors = TOKEN_ARGS['return_tensors'])

            decoder_input_id = decoder_input_id
            decoder_input_ids = [decoder_input_id for _ in range(input_ids.shape[0])]
            decoder_input_ids = torch.cat(decoder_input_ids, dim = 0).to(device)
            last_token_indices = attention_mask.sum(dim = 1) - 1

            outputs = model(input_ids = input_ids,
                            decoder_input_ids = decoder_input_ids,
                            attention_mask = attention_mask)

            encoder_last_hidden_state = outputs.encoder_last_hidden_state
            decoder_last_hidden_state = outputs.last_hidden_state
            for mini_idx in range(last_token_indices.shape[0]):
                seq_length = int(last_token_indices[mini_idx])
                decoded_seq_length = decoder_last_hidden_state.shape[1] - 1
                last_encoder_embedding = encoder_last_hidden_state[mini_idx, seq_length, :].cpu()
                last_decoder_embedding = decoder_last_hidden_state[mini_idx, decoded_seq_length, :].cpu()
                embeddings[running_idx]['encoder_embedding'] = last_encoder_embedding
                embeddings[running_idx]['decoder_embedding'] = last_decoder_embedding
                running_idx += 1

    return embeddings


