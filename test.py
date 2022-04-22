import torch
import torch.utils.data as tud
import torch.nn as nn
import numpy as np
import torchtext.vocab as vocab
from transformers import BertTokenizer
from transformers import BertForMaskedLM


Tw = torch.randint(high=2,low=0,size=[30,10],dtype=torch.long)
Ew  = torch.zeros([10,5],dtype=torch.long)
E  = torch.rand([30,5])

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,word_embedding,co_occurrence):
        super().__init__()
        self.word_embedding = word_embedding
        self.co_occurrence = co_occurrence
    def __len__(self):
        return len(self.word_embedding)

    def __getitem__(self, idx):
        return self.word_embedding[idx],self.co_occurrence[idx]

dt =  WordEmbeddingDataset(E,Tw)
BATCH_SIZE = 1
dataloader = tud.DataLoader(dt, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size  # 10
        self.embed_size = embed_size  # 5
        # [10, 5] matrix
        self.in_embed = torch.rand(size=[self.vocab_size, self.embed_size])


    def forward(self, word_embedding, co_occurrence):
        log1 = torch.bmm(co_occurrence,self.in_embed)


