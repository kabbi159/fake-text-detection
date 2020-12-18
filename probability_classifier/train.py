import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import torch.nn.functional as F
from prob_classifier import *
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--max_seq_len', type=int, default=128)

parser.add_argument('--bi', action='store_true',help="Train using bi-directional gru")
parser.add_argument('--prob', action='store_true',help="Train using probability values, set false to train with rank")

parser.add_argument('--train_sample', type=int, default=2)
parser.add_argument('--test_sample', type=int, default=2)

parser.add_argument('--prob_dir', type=str, default='extracted')
args = parser.parse_args()

seed=args.seed
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

#Model Save Dir
if not os.path.exists('weights'):
    os.makedirs('weights')

#Select Train/Val Data
sample_types=['','-k40','-nucleus']
train_sample=sample_types[args.train_sample]
val_sample=sample_types[args.train_sample]
test_sample=sample_types[args.test_sample]

def load_data(prob_dir,load_type,fake_dataset):
    probs=np.load(prob_dir+'/{}_{}_probs.npy'.format(load_type,fake_dataset))
    ranks=np.load(prob_dir+'/{}_{}_ranks.npy'.format(load_type,fake_dataset))
    labels=np.load(prob_dir+'/{}_{}_labels.npy'.format(load_type,fake_dataset))
    masks=np.load(prob_dir+'/{}_{}_masks.npy'.format(load_type,fake_dataset))
    return probs,ranks,labels,masks

#Load Train
train_fake='xl-1542M'+train_sample
train_probs,train_ranks,train_labels,train_masks=load_data(args.prob_dir,'train',train_fake)

val_fake='xl-1542M'+val_sample
val_probs,val_ranks,val_labels,val_masks=load_data(args.prob_dir,'val',val_fake)

test_fake='xl-1542M'+test_sample
test_probs,test_ranks,test_labels,test_masks=load_data(args.prob_dir,'test',test_fake)

def apply_mask(data,masks):
    masks=masks[:,1:]
    data[masks==0]=-1
    return data

train_prob=args.prob
bi_model=args.bi

#train_prob False -> Train with rank
if train_prob:
    train_data=train_probs*100
    val_data=val_probs*100
    test_data=test_probs*100
    if bi_model:
        model_name='prob_bi'
    else:
        model_name='prob'
else:
    train_data=train_ranks
    val_data=val_ranks
    test_data=test_ranks
    if bi_model:
        model_name='rank_bi'
    else:
        model_name='rank'

print(model_name)
val_data=apply_mask(val_data,val_masks)
test_data=apply_mask(test_data,test_masks)


if bi_model:
    model=BiGRUClassifier(args.max_seq_len).to(device)
else:
    model=GRUClassifier(args.max_seq_len).to(device)

#Xavier Init
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)

class ProbDataset(Dataset):
    def __init__(self, data,labels,masks):
        data=self.apply_mask(data,masks)
        self.x=torch.tensor(data,dtype=torch.float)
        self.y=torch.tensor(labels, dtype=torch.long)

    def apply_mask(self,data,masks):
        masks=masks[:,1:]
        data[masks==0]=-1
        return data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # pass
        return self.x[index],self.y[index]

num_epochs = 100
batch_size=512

train_dataset=ProbDataset(train_data,train_labels,train_masks)
train_loader=DataLoader(train_dataset,batch_size,sampler=RandomSampler(train_dataset),num_workers=8)

best_model=None
best_acc=0
best_epoch=0
for epoch in range(num_epochs):
    model.train()
    with tqdm(train_loader,unit="batch") as tepoch:
        total_loss = 0
        for i,(x,y) in enumerate(tepoch):
            model.zero_grad()
            tepoch.set_description(f"Epoch {epoch}")
            x,y=x.to(device),y.to(device)
            # loss = 0
            pred = model(x)

            loss=criterion(pred,y)
            acc=accuracy_score(y.detach().cpu().numpy(),np.argmax(pred.detach().cpu().numpy(),axis=1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tepoch.set_postfix(loss=loss.item(), accuracy=acc.item())

    model.eval()
    with torch.no_grad():
        x,y=torch.tensor(val_data,dtype=torch.float).to(device),torch.tensor(val_labels,dtype=torch.long).to(device)
        pred=model(x)
        loss=criterion(pred,y)
        pred=F.softmax(pred,dim=-1)
        pred_ans=np.argmax(pred.detach().cpu().numpy(),axis=1)
        acc=accuracy_score(y.detach().cpu().numpy(),np.argmax(pred.detach().cpu().numpy(),axis=1))
    print("Val Acc {:.4f} Loss {:.4f}\n".format(acc.item(),loss.item()))

    if acc>best_acc:
        best_acc=acc
        best_epoch=epoch+1
        torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict()
                    ),
                    "weights/best_{}_{}.pt".format(model_name,val_fake)
                )
print("Best ACC",best_acc)
print("Best Epoch",best_epoch)

#Log Results
with open("train_log.txt", "a") as myfile:
    myfile.write("best_{}_{}\n".format(model_name,val_fake))
    myfile.write("Best ACC {:.5f}\n".format(best_acc))
    myfile.write("Best Epoch {}\n".format(best_epoch))
