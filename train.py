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
        self.linear = nn.Linear(vocab_size,embed_size,bias=True)


    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        return out


# prepare the input
#word_size = 400000
#word_piece_size = 30000
#embed_size = 300
BATCH_SIZE = 5000
cache_dir = 'GloVe6B5429'
#Tw = torch.randint(high=2,low=0,size=[word_size,word_piece_size],dtype=torch.float )
Tw = torch.load('word-word-piece.pt')
#Ew  = torch.zeros([word_piece_size,embed_size],dtype=torch.long)
E = vocab.GloVe(name='6B', dim=300, cache=cache_dir).vectors


# prepare dataloader
dt = WordEmbeddingDataset(E,Tw)
dataloader = tud.DataLoader(dt, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# define the hyperprameters
LR = 0.01
# define the nn model and optimizer and loss function
net_sgd = EmbeddingModel(Tw.shape[1],E.shape[1])
opt_sgd = torch.optim.SGD(net_sgd.parameters(),lr=LR)
loss_func = torch.nn.L1Loss(size_average=False, reduce=True)
loss_his = []


# training
EPOCH = 5
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step,(y,x) in enumerate(dataloader):
        output = net_sgd(x)
        loss = loss_func(output,y)
        opt_sgd.zero_grad()
        loss.backward()
        opt_sgd.step()
        loss_his.append(loss.data.numpy())
        print('epoch:{},loss:{}'.format(epoch,loss))

print(len(loss_his))
# for i in loss_his:
#     plt.plot(i,lable='SGD')
#     plt.legend(loc='best')
#     plt.xlabel('Steps')
#     plt.ylabel('Loss')
#     plt.show()






