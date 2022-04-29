import torch
import torch.utils.data as tud
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchtext.vocab as vocab
from transformers import BertTokenizer
from transformers import BertForMaskedLM


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,word_embedding,co_occurrence):
        super().__init__()
        self.word_embedding = word_embedding
        self.co_occurrence = co_occurrence
    def __len__(self):
        return len(self.word_embedding)

    def __getitem__(self, idx):
        return self.word_embedding[idx],self.co_occurrence[idx]

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel,self).__init__()
        self.vocab_size = vocab_size  # 10
        self.embed_size = embed_size  # 5
        self.linear = nn.Linear(vocab_size,embed_size,bias=False)
        self.weight = self.linear.weight
    def __getweight__(self):
        return self.weight

    def forward(self, x):
        out = self.linear(x)
        return out


# prepare the input
#word_size = 400000
#word_piece_size = 30000
#embed_size = 300
BATCH_SIZE = 5000
cache_dir = 'GloVe6B5429'
#Tw = torch.randint(high=2,low=0,size=[word_size,word_piece_size],dtype=torch.float )
Tw = torch.load('word_piece_co.pt')
#Tw = torch.randint(size=[50,2],low=0,high=2,dtype=torch.float)
E = vocab.GloVe(name='6B', dim=300, cache=cache_dir).vectors
#E  = torch.rand([50,2],dtype=torch.float)

# prepare dataloader
dt = WordEmbeddingDataset(E,Tw)
dataloader = tud.DataLoader(dt, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# define the hyperprameters
LR = 0.01
# define the nn model and optimizer and loss function
net_sgd = EmbeddingModel(Tw.shape[1],E.shape[1])
opt_sgd = torch.optim.SGD(net_sgd.parameters(),lr=LR)
loss_func = torch.nn.L1Loss(reduction='sum')
loss_his = []


# training
EPOCH = 100
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step,(y,x) in enumerate(dataloader):
        output = net_sgd(x)
        loss = loss_func(output,y)
        opt_sgd.zero_grad()
        loss.backward()
        opt_sgd.step()
        loss_his.append(loss.data.numpy())
    if  np.mod(epoch,10) == 0:
        print('epoch:{},loss:{}'.format(epoch, loss))
    #print(net_sgd.weight)
torch.save(net_sgd.weight,'./word_piece_embedding.pt')





