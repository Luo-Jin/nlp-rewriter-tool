from transformers import BertTokenizer
from transformers import BertForMaskedLM
import stanza
import torchtext.vocab as vocab
import sys
import getopt
import random
import torch
import copy

text = ["fireman working for ul for ten years."]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
print(tokenizer.tokenize(text[0]))
tokens = tokenizer(text,add_special_tokens=True,return_tensors="pt")
tokens["input_ids"][:,5] = tokenizer.mask_token_id
print(tokens["input_ids"][0])
model.eval()
with torch.no_grad():
    predictions = model(**tokens)

predicted_id = torch.argmax(predictions.logits,dim=-1)
#print(tokenizer.convert_ids_to_tokens(t) for t in predicted_id[:])
for t in predicted_id[:]:
    print(t)
    print(tokenizer.decode(t.numpy(),skip_special_tokens=True))






