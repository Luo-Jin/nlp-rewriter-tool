import random
import torch
import copy
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from torch.nn import functional as F
import torchtext.vocab as vocab

text = "Language Modeling is the task of predicting the best word to follow or continue a sentence given all the words already in the sentence."
# all the punctuations will not be replaced
punctuations = ["[CLS]","[UNK]","[MASK]","[SEP]","[PAD]","'",'"',";",":",",",".","?","/",">","<","{","}"]
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')
org_tokens = tokenizer(text, return_tensors="pt")
cache_dir = 'GloVe6B5429'
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
word_piece_embeddings = torch.load('word-piece-embedding.txt').t()




# track all replacable word's position
mask_pos = []
for i in range(len(org_tokens["input_ids"][0])):
    if tokenizer.convert_ids_to_tokens(org_tokens["input_ids"][0][i].item()) not in punctuations:
        mask_pos.append(i)

# iteratively replace the mask words
i = 1
tokens = copy.deepcopy(org_tokens)
while len(mask_pos) > 0 and i <= 2:
    i = i+1
    #pick a random word in k position and mask it
    pos = random.sample(mask_pos,1)
    mask_pos.remove(pos[0])
    tokens["input_ids"][0][pos[0]] = tokenizer.mask_token_id
    #calculate the P(lm), it is a token_size * vocal_size matrix
    logits = model(**tokens)
    logits = logits.logits
    softmax = F.softmax(logits, dim=-1)
    all_word_idx = torch.argmax(softmax[0],dim=-1)
    mask_word_idx = torch.argmax(softmax[0, pos[0], :])
    #calculate the P(enforce).
    c = word_piece_embeddings[tokens["input_ids"][0]]
    c = c[torch.arange(c.size(0)) != pos[0]]
    c = torch.sum(c, dim=0)
    word_piece_embeddings1 = word_piece_embeddings + c
    print(word_piece_embeddings1)

    #print ("'%s' was replaced by '%s'"%(tokens["input_ids"][0][pos[0]],mask_word_idx))
    tokens["input_ids"][0][pos[0]] = mask_word_idx

print(text)
print(tokenizer.decode(all_word_idx))