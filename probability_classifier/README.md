# fake-text-detection
Fake Text Detection Toy Project in ADS5035 (Data-driven Security and Privacy)  
This is a experiment code on WebText and gpt2-output-dataset

## Token probability-base Classifier

## Probability/Rank Extractor Usage
Extract Probability & Rank of each token with 16 Threads
num-train-pairs 50000 means Real:Fake = 50,000:50,000
python prob_extract.py
--batch-size=32 \
--max-sequence-length=128 \
--seed 10 \
--num-workers 16\
--num-train-pairs 50000

## Baseline Usage
