###############################################
#   File    : train.py
#   Author  : Jin luo
#   Date    : 2022-04-23
#   Input   : word_piece_co.pt
#   Output  : word_piece_embedding.pt
###############################################
'''
Description:

this script train a word-piece embeddingv 'E′' based on the GloVe embeddings 'E(w) '
and a word-piece indicator matrix 'T (w)' that represents relationship
between the BERT 30k vocabs and GloVe 400k vocabs.

it simply use a shallow NN with just one Linear layer to minimize the
L1 loss of |E(w) − T (w)E′|. not sure if need a activation function or not ?
the SGD optimizer was used to train this model with batch size 5000 and learning rate 0.01.

'''
import numpy
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
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.weight = self.linear.weight
    def __getweight__(self):
        return self.weight

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmod(out)
        return out


# prepare the input
BATCH_SIZE = 100
cache_dir = 'GloVe6B5429'
#Tw = torch.load('word_piece_co.pt')
Tw = torch.load('test/tw.pt')
#E = vocab.GloVe(name='6B', dim=300, cache=cache_dir).vectors
E = torch.load('test/e.pt')
# prepare dataloader
dt = WordEmbeddingDataset(E,Tw)
dataloader = tud.DataLoader(dt, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# define the hyperprameters
LR = 0.5
# define the nn model and optimizer and loss function
net_sgd = EmbeddingModel(Tw.shape[1],E.shape[1])
opt_sgd = torch.optim.SGD(net_sgd.parameters(),lr=LR)
#loss_func = torch.nn.L1Loss(reduction='sum')
loss_func = torch.nn.SmoothL1Loss(reduction='mean')
loss_his = []
loss_epoch = []

# training
EPOCH = 100000
for epoch in range(EPOCH):
    #print('Epoch: ', epoch)
    for step,(y,x) in enumerate(dataloader):
        output = net_sgd(x)
        loss = loss_func(output,y)
        opt_sgd.zero_grad()
        loss.backward()
        opt_sgd.step()
        loss_his.append(loss.data.numpy())
    if np.mod(epoch,100) == 0:
        print('epoch:{}, loss:{}'.format(epoch, numpy.mean(loss_his[int(epoch*len(dt)/BATCH_SIZE):int((epoch+1)*len(dt)/BATCH_SIZE-1)])))
torch.save(loss_his,'loss.pt')
torch.save(net_sgd.weight,'test/weight.pt')

loss_t = torch.tensor([torch.from_numpy(a) for a in loss_his])
loss_t = loss_t.view(100,4000)

plt.plot(torch.arange(loss_t.size(0)),torch.mean(loss_t,dim=1))
plt.show()





