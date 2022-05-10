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
import torch
import torch.utils.data as tud
import torch.nn as nn
from matplotlib import pyplot as plt
import time
import numpy as np
import sys
import getopt
import torchtext.vocab as vocab


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
        self.vocab_size = vocab_size  # 30k number of tokens in BERT
        self.embed_size = embed_size  # 300
        self.linear = nn.Linear(vocab_size,embed_size,bias=False)
        self.sigmod = nn.Sigmoid()
        self.weight = self.linear.weight
    def __getweight__(self):
        return self.weight

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmod(out)
        return out

def train(epoch:int,batch:int,lr:float):
    # prepare the input
    BATCH_SIZE = batch
    cache_dir = 'GloVe6B5429'
    Tw = torch.load('word_piece_co.pt') # 400k x 30k
    #Tw = torch.randint(low=0,high=1,size=[400,300])
    E = vocab.GloVe(name='6B', dim=300, cache=cache_dir).vectors # 400k x 300
    #E = torch.rand(size=[400,30])
    # prepare dataloader
    dt = WordEmbeddingDataset(E,Tw)
    dataloader = tud.DataLoader(dt, batch_size=BATCH_SIZE, shuffle=False, num_workers=48)

    # define the hyperprameters
    LR = lr
    # define the nn model and optimizer and loss function
    net_sgd = EmbeddingModel(Tw.shape[1], E.shape[1])
    opt_sgd = torch.optim.SGD(net_sgd.parameters(), lr=LR)
    #loss_func = torch.nn.L1Loss(reduction='mean')
    loss_func = torch.nn.SmoothL1Loss()
    loss_his = []
    loss_epoch = []

    # training
    EPOCH = epoch
    time1 = time.time()
    for epoch in range(EPOCH):
        # print('Epoch: ', epoch)
        for step,(y,x) in enumerate(dataloader):
            output = net_sgd(x)
            loss = loss_func(output,y)
            opt_sgd.zero_grad()
            loss.backward()
            opt_sgd.step()
            loss_his.append(loss.data.numpy())
        if  np.mod(epoch,10) == 0:
            time2 = time.time()
            interval = time2 - time1
            time1 = time2
            torch.save(loss_his, 'loss.pt')
            torch.save(net_sgd.weight, 'weight.pt')
            print('epoch:{},runtime:{},loss:{}'.format(epoch,interval,np.mean(loss_his[int(epoch*len(dt)/BATCH_SIZE):int((epoch+1)*len(dt)/BATCH_SIZE)-1])))

def main():
    arg_epoch = None
    arg_batch = None
    arg_lr = None
    arg_help = "{0} -e <epoch> -b <batch> -l <learning rate>".format(sys.argv[0])

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:e:b:l:", ["help", "epoch=","batch=", "lr="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-e", "--epoch"):
            arg_epoch = int(arg)
        elif opt in ("-b", "--batch"):
            arg_batch = int(arg)
        elif opt in ("-l", "--lr"):
            arg_lr = float(arg)

    train(epoch=arg_epoch,batch=arg_batch,lr=arg_lr)


if __name__ == "__main__":
    sys.exit(main())


