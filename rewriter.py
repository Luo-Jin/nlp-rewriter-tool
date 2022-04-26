import random
import torch
import copy
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from torch.nn import functional as F
import torchtext.vocab as vocab
import sys


text = 'A novel is a relatively long work of narrative fiction, typically written in prose and published as a book. The present English word for a long work of prose fiction derives from the Italian: novella for "new", "news", or "short story of something new", itself from the Latin: novella, a singular noun use of the neuter plural of novellus, diminutive of novus, meaning "new". Some novelists, including Nathaniel Hawthorne, Herman Melville, Ann Radcliffe, John Cowper Powys, preferred the term "romance" to describe their novels.'
# all the punctuations will not be replaced
punctuations = ["[CLS]","[UNK]","[MASK]","[SEP]","[PAD]","'",'"',";",":",",",".","?","/",">","<","{","}"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mapping = tokenizer(text, return_tensors="pt")
org_tokens = mapping["input_ids"][0]
cache_dir = 'GloVe6B5429'
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
word_piece_embeddings = torch.load('word-piece-embedding.txt').t()




# track all replacable word's position
mask_pos = []
for i in range(len(org_tokens)):
    if tokenizer.convert_ids_to_tokens(org_tokens[i].item()) not in punctuations:
        mask_pos.append(i)

# iteratively replace the mask words
i = 1
tokens = copy.deepcopy(org_tokens)
while len(mask_pos) > 0 and i <= len(tokens):
    i = i+1
    #pick a random word in k position and mask it
    pos = random.sample(mask_pos,1)
    mask_pos.remove(pos[0])
    #calculate the P(enforce).
    c = word_piece_embeddings[tokens]
    Rx = torch.ones([word_piece_embeddings.size(0),word_piece_embeddings.size(1)])
    Rx = Rx * torch.sum(c,dim=0)
    c = c[torch.arange(c.size(0)) != pos[0]]
    c = torch.sum(c, dim=0)
    Ru = word_piece_embeddings + c
    s = torch.cosine_similarity(Ru,Rx,dim=1)
    σ = 0.975
    k = 0.1
    Penforce = -k*torch.max(torch.zeros(s.size(0)),(σ-s))
    #calculate the P(lm), it is a token_size * vocal_size matrix
    tokens[pos[0]] = tokenizer.mask_token_id
    mapping["input_ids"][0]=copy.deepcopy(tokens)
    logits = model(**mapping)
    logits = logits.logits
    softmax = torch.softmax(logits, dim=-1)
    Plm = softmax[0][pos[0]]
    # calculate Pproposal
    Pproposal = Plm + Penforce
    all_word_idx = torch.argmax(softmax[0],dim=-1)
    mask_word_idx = torch.argmax(Pproposal)
    print ("'%s' was replaced by '%s'"%(tokenizer.decode(tokens[pos[0]]),tokenizer.decode(mask_word_idx)))
    tokens[pos[0]] = mask_word_idx

print(text)
print(tokenizer.decode(all_word_idx))
