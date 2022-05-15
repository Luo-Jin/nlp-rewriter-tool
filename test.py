import torch

import stanza
from transformers import BertTokenizer
import torchtext.vocab as vocab
import re
cache_dir = 'train/GloVe6B5429'
glove = vocab.GloVe(name='840B', dim=300, cache=cache_dir)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# stanza.download('en',model_dir='stanza')       # This downloads the English models for the neural pipeline
txt  = "Twenty state or territorial democratic parties intend to apply to hold early presidential nominating contests in 2024, " \
       "a DNC official told CNN Saturday, as the party reevaluates its process of selecting nominees."
tokens = tokenizer(txt, return_tensors="pt",return_token_type_ids=False,return_attention_mask=False,return_special_tokens_mask=False)
en_nlp = stanza.Pipeline('en',model_dir='stanza',processors='tokenize,ner') # This sets up a default neural pipeline in English
#doc = en_nlp()

# for t in tokens["input_ids"][0]:
#     w = tokenizer.ids_to_tokens[t.item()]
#     doc=en_nlp(w)
#     print(doc.entities)
doc = en_nlp(txt)
words = doc.to_dict()[0]
# ids = {'input_ids':torch.tensor([[tokenizer.convert_tokens_to_ids(w['text']) for w in words]])}
# ids['input_ids'] = torch.cat((torch.tensor([[101]]),ids['input_ids'],torch.tensor([[102]])),dim=-1)
# print(tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
mask_pos = []
for i in range(len(tokens["input_ids"][0])):
        id = tokens["input_ids"][0][i]
        w = tokenizer.ids_to_tokens[id.item()]
        re.fullmatch('##[0-9]*', w)  # determine if it is a number
        doc = en_nlp(w)  # determine if it is an entity
        if  len(doc.entities) == 0:
               mask_pos.append(i)
print(tokenizer.convert_tokens_to_ids("I"))
print(tokenizer.convert_tokens_to_ids("i"))



# print(ids)
# print(tokenizer.decode(ids['input_ids'][0]))
# print(tokens)
# print(tokenizer.decode(tokens['input_ids'][0]))
# print([w['text'] for w in words])
# print(tokenizer.convert_tokens_to_ids('cnn'.lower()))