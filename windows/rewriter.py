###############################################
#   File    : rewriter.py
#   Author  : Jin luo
#   Date    : 2022-05-17
#   Input   : None
#   Output  : None
###############################################
'''
a GUI for the rewriter tool
'''


import copy
import re
import os,io
import random
from tkinter import Entry
from tkinter import Label
from tkinter import Tk
from tkinter import SUNKEN
from tkinter import RAISED
from tkinter import END
from tkinter import Text
from tkinter import Button
from tkinter import ttk
from configparser import ConfigParser
import numpy as np
import torch
import stanza
from transformers import BertTokenizer
from transformers import BertForMaskedLM

config_file = os.path.join(os.path.abspath("."),"rewriter.ini")
config = ConfigParser()
with io.open(config_file, 'r', encoding='utf_8_sig') as fp:
    config.readfp(fp)


# load embeddings
tokenizer = BertTokenizer.from_pretrained(config.get('EMBEDDINGS','bert'))
model = BertForMaskedLM.from_pretrained(config.get('EMBEDDINGS','bert'))
word_piece_embeddings = torch.from_numpy(torch.load(config.get('EMBEDDINGS','word_piece'))).t()
en_nlp = stanza.Pipeline('en',processors='tokenize,ner')

# prepare windows
win = Tk()
win.title("Sentence Rewriter")
win.geometry('566x350')

lbr0 = Label(text="Number of alternative:")
lbr1 = Label(text="Sentence to be rewritten：")
lbr2 = Label(text="Suggested alternatives:")
lbr3 = Label(text="Similarity rate:")
lbr4 = Label(text="Smooth rate:")
lbr5 = Label(text="Number of mask:")

cbox_num = ttk.Combobox(win,width=2)
cbox_num['value'] = (1,2,3)
cbox_num.current(2)
ent_similarity = Entry(win,width=5,bd=1,relief=SUNKEN)
ent_smooth = Entry(win,width=5,bd=1,relief=SUNKEN)
ent_mask = Entry(win,width=5,bd=1,relief=SUNKEN)
txt_edit = Text(win, width = 70, height = 5,bd=1,relief=SUNKEN)
txt_rephrase = Text(win, width = 70, height = 10,bd=1,relief=SUNKEN)
btn_rephrase = Button(win, text="Rephrase",relief=RAISED, width=10)

lbr0.grid(row=0,column=0,sticky="w")
cbox_num.grid(row=0,column=1,sticky="w")
lbr3.grid(row=0,column=2,sticky="e")
ent_similarity.grid(row=0,column=3,sticky="w")
lbr4.grid(row=0,column=4,sticky="e")
ent_smooth.grid(row=0,column=5,sticky="w")
lbr5.grid(row=0,column=6,sticky="e")
ent_mask.grid(row=0,column=7,sticky="w")

lbr1.grid(row=1,column=0,columnspan=8,sticky="w")
txt_edit.grid(row=2,column=0,columnspan=8)
lbr2.grid(row=3,column=0,columnspan=8,sticky="w")
txt_rephrase.grid(row=4,column=0,columnspan=8)
btn_rephrase.grid(row=5,column=0,columnspan=8)
ent_similarity.insert(0,config.getfloat('PARAMS','σ'))
ent_smooth.insert(0,config.getint('PARAMS','k'))


def button_click(event):
    txt = txt_edit.get("1.0",END)
    σ   = float(ent_similarity.get())
    k   = float(ent_smooth.get())
    b   = int(cbox_num.get())
    m   = int(ent_mask.get()) if ent_mask.get() != "" else 0
    tokens = rephrase(txt,σ,k,b,m)
    txt_rephrase.delete("1.0",END)
    for i in torch.arange(len(tokens["input_ids"])):
        txt_rephrase.insert(END
                            ,"{}{}{}\n\n".format(i + 1
                                             , "."
                                             , tokenizer.decode(tokens["input_ids"][i]
                                                                , skip_special_tokens=True)))


btn_rephrase.bind('<ButtonPress-1>', button_click)


def penforce(batch,pos,org_tokens,tokens,k,σ):
    # calculate the Rx for original sentence 3 x 30k x 300
    Rx = word_piece_embeddings[org_tokens["input_ids"][0]]
    Rx = torch.sum(Rx, dim=0)
    Rx = Rx.expand(batch, word_piece_embeddings.size(0)
                   , word_piece_embeddings.size(1))
    # calculate the Ru for changed sentence in 3 x 30k x 300
    Ru = word_piece_embeddings[tokens["input_ids"]]
    # remove masked word from Ru
    Ru = torch.cat([t[torch.arange(t.size(0)) != pos[0]] for t in Ru[:]])
    Ru = Ru.view(batch, int(Ru.size(0) / batch), word_piece_embeddings.size(1))
    # sum rest words prob
    Ru = torch.sum(Ru, dim=1)
    Ru = torch.cat([t.expand(word_piece_embeddings.size(0)
                             , word_piece_embeddings.size(1)) for t in Ru[:]])
    # add all other possible words prob
    Ru = Ru.view(batch, word_piece_embeddings.size(0)
                 , word_piece_embeddings.size(1))
    Ru = Ru + word_piece_embeddings
    # compute similarities between original sentence with all possible sentences (30k)
    s = torch.cosine_similarity(Ru, Rx, dim=2)
    Penforce = torch.exp(-k * torch.max(torch.zeros(batch, word_piece_embeddings.size(0)), (float(σ) - s)))
    return Penforce

def plm(pos,tokens):
    # replace selected word with [MASK]
    tokens["input_ids"][:, pos[0]] = tokenizer.mask_token_id
    logits = model(**tokens)
    logits = logits.logits
    softmax = torch.softmax(logits, dim=-1)
    Plm = softmax[:, pos[0]]
    return Plm

def rephrase(txt,σ=0.975,k=0.1,batch=1,m=0):
    # set minibatch size of this task, determine how many sentences will be created in one call.
    text = [txt] * batch
    org_tokens = tokenizer(text, return_tensors="pt"
                           ,return_token_type_ids=False
                           ,return_attention_mask=False
                           )
    special_tokens = {"[CLS]": 0, "[UNK]": 0, "[MASK]": 0, "[SEP]": 0, "[PAD]": 0
        , "'": 0, '"': 0, ";": 0, ":": 0, ",": 0,".": 0, "?": 0
        , "/": 0, ">": 0, "<": 0, "{": 0, "}": 0,"-":0,"+":0,"=":0,"_":0
        , "!":0,"@":0,"#":0,"$":0,"%":0,"^":0,"&":0,"*":0,"(":0,")":0}
    special_tokens = {k:tokenizer.convert_tokens_to_ids(k) for k,v in special_tokens.items()}
    # determine all replaceable positions in the sentence.
    mask_pos = []
    for i in range(len(org_tokens["input_ids"][0])):
        id = org_tokens["input_ids"][0][i]
        w = tokenizer.ids_to_tokens[id.item()]
        re.fullmatch('##[0-9]*', w)  # determine if it is a number
        doc = en_nlp(w)  # determine if it is an entity
        if id not in special_tokens.values() \
                and not w.isdigit() \
                and re.fullmatch('##[0-9]*', w) == None \
                and len(doc.entities) == 0:
            mask_pos.append(i)

    # iteratively replace the mask words
    i = 1
    tokens = copy.deepcopy(org_tokens)
    m = len(tokens["input_ids"][0]) if m == 0 else m
    while len(mask_pos) > 0 and i <= m:
        i = i+1
        # pick a random word in k position and mask it
        pos = random.sample(mask_pos,1)
        mask_pos.remove(pos[0])
        # calculate the prob of enforcement
        Penforce = penforce(batch,pos,org_tokens,tokens,k,σ)
        # calculate the P(lm), it is a token_size * vocal_size matrix
        Plm = plm(pos,tokens)
        # calculate the proposal prob
        Pproposal = Plm + Penforce
        ppl_word_idx = torch.topk(Pproposal,k=batch,dim=-1)
        y = torch.diag_embed(torch.ones(batch,dtype=torch.long))
        sample_idx = torch.sum(ppl_word_idx.indices * y,dim=-1)
        tokens["input_ids"][:,pos[0]] = sample_idx
    return tokens

win.mainloop()
