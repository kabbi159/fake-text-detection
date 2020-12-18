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

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model.to(device)

    # load data
    train_loader, validation_loader, test_loader = load_datasets(args, tokenizer)

    # my model
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = 0.
    for epoch in range(args.max_epochs):
        train(model, optimizer, train_loader, args, device)
        val_acc = validation(model, validation_loader, args, device)
        test_acc = test(model, test_loader, args, device)

        print(f"Epoch {epoch + 1} | val_acc: {val_acc} test_acc: {test_acc}")

        if val_acc > best_val:
            os.makedirs(args.save_dir, exist_ok=True)
            model_name = 'baseline_' + args.model_name + '.pt'
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(dict(
                epoch=epoch+1,
                model_state_dict=model_to_save.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                args=args
            ),
                os.path.join(args.save_dir, model_name)
            )
            print("Model saved to", args.save_dir)
            best_val = val_acc


if __name__ == '__main__':
    main()
