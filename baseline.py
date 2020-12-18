import argparse
import wandb
import torch
import os

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW, tokenization_utils

from train import train, validation, test
from dataset import load_datasets


def main():
    # argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-sequence-length', type=int, default=128)

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='webtext')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M-nucleus')
    parser.add_argument('--save-dir', type=str, default='bert_logs')

    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--model-name', type=str, default='bert-base-cased')
    parser.add_argument('--wandb', type=bool, default=True)

    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.model_name)

    # config, tokenizer, model
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenization_utils.logger.setLevel('DEBUG')

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config
    )

    # load data
    train_loader, validation_loader, test_loader = load_datasets(args, tokenizer)

    # my model
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = 0.
    for i in range(args.max_epochs):
        train(model, optimizer, train_loader, args)
        val_acc = validation(model, validation_loader, args)
        test_acc = test(model, test_loader, args)

        print(f"Epoch {i + 1} | val_acc: {val_acc} test_acc: {test_acc}")

        if val_acc > best_val:
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_name + '.pt'))
            print("Model saved to", args.save_dir)
            best_val = val_acc


if __name__ == '__main__':
    main()
