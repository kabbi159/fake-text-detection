import json
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import os

import requests

ALL_DATASETS = [
    'webtext',
    'small-117M',  'small-117M-k40',  'small-117M-nucleus',
    'medium-345M', 'medium-345M-k40', 'medium-345M-nucleus',
    'large-762M',  'large-762M-k40',  'large-762M-nucleus',
    'xl-1542M',    'xl-1542M-k40',    'xl-1542M-nucleus'
]


def download(*datasets, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)

    for ds in datasets:
        assert ds in ALL_DATASETS, f'Unknown dataset {ds}'

        for split in ['train', 'valid', 'test']:
            filename = ds + "." + split + '.jsonl'
            output_file = os.path.join(data_dir, filename)
            if os.path.isfile(output_file):
                continue

            r = requests.get("https://storage.googleapis.com/gpt-2/output-dataset/v1/" + filename, stream=True)

            with open(output_file, 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


def load_texts(data_file, expected_size=None):
    texts = []

    for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line)['text'])

    return texts


class Corpus:
    def __init__(self, name, data_dir='data', skip_train=False):
        download(name, data_dir=data_dir)
        self.name = name
        self.train = load_texts(f'{data_dir}/{name}.train.jsonl', expected_size=250000) if not skip_train else None
        self.test = load_texts(f'{data_dir}/{name}.test.jsonl', expected_size=5000)
        self.valid = load_texts(f'{data_dir}/{name}.valid.jsonl', expected_size=5000)


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            label = 1
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            label = 0

        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.model_max_length - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0

        return tokens, mask, label



class BERTEncodedDataset(Dataset): # bert
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            label = 1
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            label = 0

        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.model_max_length - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0

        return tokens, mask, label


def load_datasets(args, tokenizer):

    download(args.real_dataset, args.fake_dataset, data_dir=args.data_dir)

    real_corpus = Corpus(args.real_dataset, data_dir=args.data_dir)

    fake_corpus = Corpus(args.fake_dataset, data_dir=args.data_dir)

    real_train, real_valid, real_test = real_corpus.train[:100], real_corpus.valid[:100], real_corpus.test[:100]
    fake_train, fake_valid, fake_test = fake_corpus.train[:100], fake_corpus.valid[:100], fake_corpus.test[:100]

    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, args.max_sequence_length, args.seed)
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=RandomSampler(train_dataset), num_workers=8)

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=SequentialSampler(validation_dataset), num_workers=8)

    test_dataset = EncodedDataset(real_test, fake_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=8)

    return train_loader, validation_loader, test_loader
