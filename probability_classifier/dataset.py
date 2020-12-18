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

class GPT2EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None,
                 token_dropout: float = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

        #Make BOS tensor Separately
        self.bos_token=torch.tensor(self.tokenizer.encode('<|endoftext|>'))

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            label = 1
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            label = 0

        tokens = self.tokenizer.encode(text)
        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.max_len - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens)+1)
            tokens=torch.tensor(tokens)
            return torch.cat([self.bos_token, tokens], dim=0), mask, label

        padding = [0] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor(tokens  + padding)
        tokens= torch.cat([self.bos_token, tokens], dim=0)

        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label
