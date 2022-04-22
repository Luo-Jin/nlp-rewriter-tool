import torch
import numpy as np
import torchtext.vocab as vocab
from transformers import BertTokenizer
from transformers import BertForMaskedLM

# get GloVe 400k list as Row
cache_dir = 'GloVe6B5429'
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
R = glove.itos
# get BERT 30k Dict as Column
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
v = tokenizer.get_vocab()
k = tokenizer.get_vocab().keys()
# choose the word_piece
C = {}
for word_piece in k.__iter__():
    if not word_piece.isidentifier() and word_piece.isascii() and not word_piece.isdigit() and word_piece.isprintable():
        if "##" in word_piece:
            C[word_piece[2:]] = v[word_piece]

# create a 400000 x 30522 zeros matrix
# fill the matrix, 1 as the word_piece is in word, 0 not
T = torch.zeros(len(R),len(v))
for key in C:
    for i in range(len(R)):
        if key in R[i]:
            T[i,C[key]] = +1
            print("%dth row, %dth col = %d"%(i,C[key],T[i,C[key]]))

torch.save(T)