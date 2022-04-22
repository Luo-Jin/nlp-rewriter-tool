import torch
import torch.utils.data as tud
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchtext.vocab as vocab
from transformers import BertTokenizer
from transformers import BertForMaskedLM

x = torch.rand([300,100])
torch.save(x,"weight.pt")
y = torch.load("weight.pt")
print(y.shape[0])

cache_dir = 'GloVe6B5429'
#Ew  = torch.zeros([word_piece_size,embed_size],dtype=torch.long)
#glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
#print(glove.vectors.shape)