from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import BertModel
import stanza
import torchtext.vocab as vocab
import sys
import getopt
import random
import torch
import copy

txt = "fireman working for ten years."
text = [txt]*3
tokenizer = BertTokenizer.from_pretrained('./Bert/vocabulary')
model = BertForMaskedLM.from_pretrained("./Bert/model/maskedLM")
word_piece_embeddings = torch.load('word-piece-embedding.txt').t()
tokens = tokenizer(text,add_special_tokens=True,return_tensors="pt")


c = word_piece_embeddings[tokens["input_ids"]]
c = torch.sum(c,dim=1)
print("c shape:{}".format(c.shape))
Rx = torch.ones(3,word_piece_embeddings.size(0),word_piece_embeddings.size(1))
print("Rx shape:{}".format(Rx.shape))


tokens["input_ids"][:,1] = tokenizer.mask_token_id
model.eval()
with torch.no_grad():
    predictions = model(**tokens)

#print(predictions.logits.shape)
predicted_id = torch.argmax(predictions.logits,dim=-1)
sample_idx = torch.topk(predictions.logits[0][1],k=3,dim=-1).indices
predicted_id[:,1] = sample_idx

# print(txt)
# print([tokenizer.decode(s[1:len(s)-1]) for s in predicted_id])
# #print(tokenizer.decode(s[1:len(s)-1]))










