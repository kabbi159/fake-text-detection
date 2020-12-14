# fake-text-detection
Fake Text Detection Toy Project in ADS5035 (Data-driven Security and Privacy)

## baseline Result in gpt2-output-dataset
|Model|train Acc.|dev Acc.|test Acc.|
|---|---|---|---|
|bert-base-uncased|58.0|59.2|56.7|
|roberta-base|60.6|-|-|
|electra-base|-|-|-|


## Usage
Before run this code, construct dataset  ```data/webtext.{train,dev,test}.jsonl```, ```data/xl-1542M-{k40,nucleus}.{train,dev,test}.jsonl``` with this format.  
You can run this code: (수정 예정)
```bash
python train.py
--transformer_type=bert \
--model_name=bert-base-cased \
--seed=42 \
--lr=3e-5 \
--wandb=True
```

## Issue (check == solved) (수정 예정)
- [x] apply wandb and pytorch lighting - but it is my first attempt to use these libraries, please excuse my poor code :)
- [ ] roberta model training failure
- [x] long sequence processing implementation
- [ ] long sequence processing harms performance
