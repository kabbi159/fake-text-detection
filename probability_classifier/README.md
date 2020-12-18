# fake-text-detection
Fake Text Detection Toy Project in ADS5035 (Data-driven Security and Privacy)  
This is a experiment code on WebText and gpt2-output-dataset

## Token probability-base Classifier

## Probability/Rank Extractor Usage
Extract Probability & Rank of each token with 16 Threads
num-train-pairs 50000 means Real:Fake = 50,000:50,000
```bash
python prob_extract.py
--batch-size=32 \
--max-sequence-length=128 \
--seed 10 \
--num-workers 16\
--num-train-pairs 50000
```

## Baseline Usage


|Model|Train|Top-k 40|Nucleus|Random|
|:---:|:---:|:---:|:---:|:---:|
|Prob|GRU<br>Bi-GRU|84.9%<br>87.31%|68.28%<br>73.5%|47.75%<br>41.41|
|Rank|GRU<br>Bi-GRU|81.51%<br>90.86|65.84%<br>69.92%|39.17%<br>46.42%|
