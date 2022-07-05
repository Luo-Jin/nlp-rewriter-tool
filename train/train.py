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
from transformers import BertTokenizer

cache_dir = 'GloVe6B5429'
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,model):
        super().__init__()
        self._glove = vocab.GloVe(name='840B', dim=300, cache=cache_dir)
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-{}'.format(model))

    def __len__(self):
        return len(self._glove)

    def __getitem__(self, idx):
        Ew = self._glove.vectors[idx]
        word = self._glove.itos[idx]
        Tw = torch.zeros(self._tokenizer.vocab_size,dtype=torch.float)
        for tok_id in self._tokenizer.convert_tokens_to_ids(
                self._tokenizer.tokenize(word)):
            Tw[tok_id] = 1
        return Tw, Ew


def train(epoch:int,batch:int,lr:float,model):
    # prepare dataloader
    dt = WordEmbeddingDataset(model)
    dataloader = tud.DataLoader(dt, batch_size=batch, shuffle=True, num_workers=48,drop_last=True)

    # define the nn model and optimizer and loss function
    linear = nn.Linear(dt._tokenizer.vocab_size, 300, bias=False)
    opt = torch.optim.SGD(lr=lr, momentum=0.5, params=linear.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    linear.weight.data.zero_()
    for word, id in dt._tokenizer.vocab.items():
        if word in dt._glove.itos:
            linear.weight.data[:, id] = dt._glove.vectors[dt._glove.stoi[word]]
    loss_func = torch.nn.L1Loss(reduction='mean')
    loss_his = []

    # training
    EPOCH = epoch
    time1 = time.time()
    for epoch in range(EPOCH):
        # print('Epoch: ', epoch)
        for step,(x,y) in enumerate(dataloader):
            y_hat = linear(x)
            loss = loss_func(y_hat,y)
            #print("len:{},sum:{}".format(len(x),torch.sum(x,dim=-1)))
            #print(y)
            #print(y_hat)
            #print(loss)
            #input()
            loss_his.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        if  np.mod(epoch,10) == 0:
            time2 = time.time()
            interval = time2 - time1
            time1 = time2
            torch.save(loss_his, 'loss.pt')
            torch.save(linear.weight.data.cpu().numpy(), '{}_{}_{}_{}_word_piece_em.pt'.format(model,epoch,batch,lr))
            print('epoch:{},runtime:{},loss:{}'
                  .format(epoch
                          ,interval
                          ,np.mean(loss_his[-int(10*len(dt)/batch):])))

def main():
    arg_epoch = None
    arg_batch = None
    arg_lr = None
    arg_model = 'bert-base-uncased'
    arg_help = "{0} -e <epoch> -b <batch> -l <learning rate> -m <bert model>".format(sys.argv[0])

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:e:b:l:m:", ["help", "epoch=","batch=", "lr=","model="])
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
        elif opt in ("-m", "--model"):
            arg_model = arg

    train(epoch=arg_epoch,batch=arg_batch,lr=arg_lr,model=arg_model)


if __name__ == "__main__":
    sys.exit(main())


