import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self,max_seq_len):
        super(SimpleClassifier,self).__init__()
        self.max_seq_len=max_seq_len
        self.model = nn.Sequential(
            nn.Linear(seq_len, 1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

    def forward(self,x):
        pred=self.model(x)
        return pred

class BiGRUClassifier(nn.Module):
    def __init__(self,max_seq_len):
        super(BiGRUClassifier,self).__init__()
        self.hid_size=8
        self.max_seq_len=max_seq_len
        # drop_p=0.2
        self.gru = nn.GRU(1,self.hid_size,num_layers=4,bidirectional=True)#,dropout=drop_p
        self.model = nn.Sequential(
            nn.Linear(self.hid_size*2*self.max_seq_len, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )


    def forward(self,x):
        x=torch.unsqueeze(x,2).view(self.max_seq_len,-1,1)
        out,_=self.gru(x)

        x=out.view(-1,self.hid_size*2*self.max_seq_len)
        pred=self.model(x)
        return pred

class GRUClassifier(nn.Module):
    def __init__(self,max_seq_len):
        super(GRUClassifier,self).__init__()
        self.hid_size=8
        self.max_seq_len=max_seq_len
        # drop_p=0.2
        self.gru = nn.GRU(1,self.hid_size,num_layers=4,bidirectional=False)#,dropout=drop_p
        self.model = nn.Sequential(
            nn.Linear(self.hid_size*self.max_seq_len, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

    def forward(self,x):
        x=torch.unsqueeze(x,2).view(self.max_seq_len,-1,1)
        out,_=self.gru(x)

        fwd=out.view(-1,self.hid_size*self.max_seq_len)

        pred=self.model(fwd)
        return pred
