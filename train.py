from tqdm import tqdm
from util import accuracy_sum
import wandb

def train(model, optimizer, train_loader, args, device):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0

    with tqdm(train_loader, desc='Train') as loop:
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()
            loss, logits = model(texts, attention_mask=masks, labels=labels)

            loss.backward()
            optimizer.step()
            batch_accuracy = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)
            if args.wandb:
                wandb.log({"loss": loss.item()})

    if args.wandb:
        wandb.log({"train_acc": train_accuracy / train_epoch_size})


def validation(model, validation_loader, args, device):
    model.eval()

    val_accuracy = 0
    val_epoch_size = 0
    val_loss = 0

    with tqdm(validation_loader, desc='Validation') as loop:
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            loss, logits = model(texts, attention_mask=masks, labels=labels)

            batch_accuracy = accuracy_sum(logits, labels)
            val_accuracy += batch_accuracy
            val_epoch_size += batch_size
            val_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=val_accuracy / val_epoch_size)

    if args.wandb:
        wandb.log({"val_acc": val_accuracy / val_epoch_size, "val_loss": val_loss / val_epoch_size})

    return val_accuracy / val_epoch_size


def test(model, test_lodaer, args, device):
    model.eval()

    test_accuracy = 0
    test_epoch_size = 0

    with tqdm(test_lodaer, desc='Validation') as loop:
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            loss, logits = model(texts, attention_mask=masks, labels=labels)

            batch_accuracy = accuracy_sum(logits, labels)
            test_accuracy += batch_accuracy
            test_epoch_size += batch_size

            loop.set_postfix(acc=test_accuracy / test_epoch_size)

    if args.wandb:
        wandb.log({"test_acc": test_accuracy / test_epoch_size})

    return test_accuracy / test_epoch_size
