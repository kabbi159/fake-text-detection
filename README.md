# fake-text-detection
Fake Text Detection Toy Project in ADS5035 (Data-driven Security and Privacy)  
This is a experiment code on WebText and gpt2-output-dataset

## Fine-tuned language model classifier
|Model|Train|Top-k 40|Nucleus|Random|
|:---:|:---:|:---:|:---:|:---:|
|BERT|Top-k<br>Nucleus<br>Random|89.79%<br>82.68%<br>47.3%|72.22%<br>78.84%<br>53.9%|43.79%<br>64.23%<br>80.45%|
|RoBERTa|Top-k<br>Nucleus<br>Random|98.35%<br>90.84%<br>51.17%|69.47%<br>88.36%<br>58.75%|49.22%<br>75.43%<br>91.34%|

## Token probability-base Classifier

## Baseline Usage
Before run this code, construct dataset  ```data/webtext.{train,dev,test}.jsonl```, ```data/xl-1542M-{k40,nucleus}.{train,dev,test}.jsonl``` with this format.  
You can run this code:
```bash
python baseline.py
--max-epochs=2 \
--batch-size=32 \
--max-sequence-length=128 \
--data-dir='data' \
--real-dataset='webtext' \
--fake-dataset='xl-1542M-nucleus' \
--save-dir='logs' \
--learning-rate=2e-5 \
--weight-decay=0 \
--model-name='bert-base-cased' \
--wandb=True
```

## Probability/Rank Extractor Usage
Extract Probability & Rank of each token with 16 Threads
num-train-pairs 50000 means Real:Fake = 50,000:50,000
python prob_extract.py
--batch-size=32 \
--max-sequence-length=128 \
--seed 10 \
--num-workers 16\
--num-train-pairs 50000
