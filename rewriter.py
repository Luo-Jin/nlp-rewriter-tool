import random
import torch
import copy
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from torch.nn import functional as F
import torchtext.vocab as vocab

text = "The world we live in is disintegrating into a place of malice and hatred, where we need hope and find it harder."
# all the punctuations will not be replaced
punctuations = ["[CLS]","[UNK]","[MASK]","[SEP]","[PAD]","'",'"',";",":",",",".","?","/",">","<","{","}"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mapping = tokenizer(text, return_tensors="pt")
org_tokens = mapping["input_ids"][0]
#org_tokens = org_tokens[torch.arange(org_tokens.size) != org_tokens["SEP"]]
print(org_tokens)
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
while len(mask_pos) > 0 and i <= 6:
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
    σ = 0.974
    k = 0.5
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