import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from transformers import *

import os
import multiprocessing
from multiprocessing import Process

from tqdm import tqdm

from dataset import download,Corpus,GPT2EncodedDataset

from scipy.special import softmax
import numpy as np
import random

#Load Datasets with GPT2EncodedDataset
def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, token_dropout=None, seed=None,
                  num_train_pairs=None, num_workers=1):

    download(real_dataset, fake_dataset, data_dir=data_dir)

    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

    real_train, real_valid, real_test = real_corpus.train, real_corpus.valid, real_corpus.test
    fake_train, fake_valid, fake_test = fake_corpus.train, fake_corpus.valid, fake_corpus.test

    if num_train_pairs:
        real_sample=np.random.choice(len(real_train),num_train_pairs)
        fake_sample=np.random.choice(len(fake_train),num_train_pairs)
        real_train=[real_train[i] for i in real_sample]
        fake_train=[fake_train[i] for i in fake_sample]

    sampler=SequentialSampler
    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = GPT2EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                    token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=SequentialSampler(train_dataset), num_workers=num_workers)

    validation_dataset = GPT2EncodedDataset(real_valid, fake_valid, tokenizer,max_sequence_length,min_sequence_length)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=SequentialSampler(validation_dataset) ,num_workers=num_workers)

    test_dataset = GPT2EncodedDataset(real_test, fake_test, tokenizer, max_sequence_length, min_sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset), num_workers=num_workers)

    return train_loader, validation_loader, test_loader

def get_rank_prob(i,return_dict,texts,out_logit,labels):
    train_probs=[]
    train_ranks=[]
    train_labels=[]
    for idx in range(texts.shape[0]):
        context=texts[idx,1:]
        logit=out_logit[idx]
        yhat = softmax(logit, axis=-1)
        sorted_preds = np.argsort(-yhat)
        y=texts[idx,1:]
        real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])
            for i in range(y.shape[0])])
        real_topk_probs = yhat[np.arange(
            0, y.shape[0], 1), y]
        train_probs.append(real_topk_probs)
        train_ranks.append(real_topk_pos)
    return_dict[i]={'probs':train_probs,'ranks':train_ranks,'labels':labels}

def extract_data(model,dataloader,device,num_workers):
    collected_probs=[]
    collected_ranks=[]
    collected_masks=[]
    collected_labels=[]

    progress_bar = tqdm(dataloader)
    for i,(texts, masks, labels) in enumerate(progress_bar):
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        out_logit=model(input_ids=texts,attention_mask=masks,return_dict=True).logits

        texts=texts.detach().cpu().numpy()
        out_logit=out_logit.detach().cpu().numpy()
        labels=labels.detach().cpu().numpy()
        masks=masks.detach().cpu().numpy()

        #Split Into Processes
        subprocesses = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        split_size=int(texts.shape[0]/(num_workers))+1
        for j in range(num_workers):
            start_idx=split_size*j
            end_idx=min(split_size*(j+1),texts.shape[0])
            process = Process(target=get_rank_prob, args=(j,return_dict,texts[start_idx:end_idx],out_logit[start_idx:end_idx],labels[start_idx:end_idx]))
            process.start()
            subprocesses.append(process)

        for proc in subprocesses:
            proc.join()

        #Combine
        for j in range(num_workers):
            collected_probs+=return_dict[j]['probs']
            collected_ranks+=return_dict[j]['ranks']
        collected_masks.append(masks)
        collected_labels.append(labels)

    #Save Combined Train Prob/Rank/Mask/Labels
    collected_probs=np.array(collected_probs)
    collected_ranks=np.array(collected_ranks)
    collected_masks=np.concatenate(collected_masks,axis=0)
    collected_labels=np.concatenate(collected_labels,axis=0)
    return collected_probs,collected_ranks,collected_masks,collected_labels

def run(device=None,
        batch_size=24,
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        data_dir='data',
        save_dir='extracted',
        real_dataset='webtext',
        fake_dataset='xl-1542M-nucleus',
        token_dropout=None,
        num_workers=1,
        num_train_pairs=25000,
        **kwargs):

    args = locals()
    print("Data: {}".format(fake_dataset))
    # Setting Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device="cpu"

    #Make Save Dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Load Model & Tokenizer
    model=GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.to(device)

    train_loader, validation_loader,test_loader = load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length,
                                                    token_dropout, seed,num_train_pairs=num_train_pairs,num_workers=num_workers)

    train_probs,train_ranks,train_masks,train_labels=extract_data(model,train_loader,device,num_workers)
    np.save(save_dir+'/train_{}_probs.npy'.format(fake_dataset),train_probs)
    np.save(save_dir+'/train_{}_ranks.npy'.format(fake_dataset),train_ranks)
    np.save(save_dir+'/train_{}_masks.npy'.format(fake_dataset),train_masks)
    np.save(save_dir+'/train_{}_labels.npy'.format(fake_dataset),train_labels)

    del train_probs
    del train_ranks
    del train_masks
    del train_labels

    val_probs,val_ranks,val_masks,val_labels=extract_data(model,validation_loader,device,num_workers)
    np.save(save_dir+'/val_{}_probs.npy'.format(fake_dataset),val_probs)
    np.save(save_dir+'/val_{}_ranks.npy'.format(fake_dataset),val_ranks)
    np.save(save_dir+'/val_{}_masks.npy'.format(fake_dataset),val_masks)
    np.save(save_dir+'/val_{}_labels.npy'.format(fake_dataset),val_labels)

    del val_probs
    del val_ranks
    del val_masks
    del val_labels

    test_probs,test_ranks,test_masks,test_labels=extract_data(model,test_loader,device,num_workers)
    np.save(save_dir+'/test_{}_probs.npy'.format(fake_dataset),test_probs)
    np.save(save_dir+'/test_{}_ranks.npy'.format(fake_dataset),test_ranks)
    np.save(save_dir+'/test_{}_masks.npy'.format(fake_dataset),test_masks)
    np.save(save_dir+'/test_{}_labels.npy'.format(fake_dataset),test_labels)

    del test_probs
    del test_ranks
    del test_masks
    del test_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--save-dir', type=str, default='extracted')
    parser.add_argument('--real-dataset', type=str, default='webtext')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M-k40')
    parser.add_argument('--token-dropout', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-train-pairs', type=int, default=50000)
    args = parser.parse_args()
    # torch.cuda.set_device(arg.local_rank)

    run(**vars(args))
