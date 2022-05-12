#import stanza
#stanza.download('en',model_dir='stanza')       # This downloads the English models for the neural pipeline
# nlp = stanza.Pipeline('en',model_dir='stanza',processors='tokenize,ner') # This sets up a default neural pipeline in English
# doc = nlp("Auckland")
# print(*[f'token: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')


# from matplotlib  import pyplot as plt
# import torch
# l = {}
# #x = torch.tensor([[torch.from_numpy(t) for t  in torch.load('test/sig_l1_e1000_b5000_l0.5.pt')]])
# l['L1Loss, Sigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/sig_l1_e1000_b5000_l0.5.pt')])
# l['L1Loss, noSigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/nosig_l1_e1000_b5000_l0.5.pt')])
# l['SmoothL1Loss, Sigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/sig_sml1_e1000_b5000_l0.5.pt')])
# l['SmoothL1Loss, noSigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/nosig_sml1_e1000_b5000_l0.5.pt')])
# i = 0
# for k,v in l.items():
#     i = i + 1
#     v = v.view(int(v.size(0) / 80), 80)
#     ax = plt.subplot(220+i)
#     ax.set_title(k)
#     plt.plot(torch.arange(v.size(0)),torch.mean(v,dim=-1))
#     plt.xlabel('epoch')
#     plt.ylabel('loss (mean)')
# plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
import torch
import numpy as np
# line_size = 90
# row_size = 80
# s_pos = 0
# y = 0
# x = 0
# pos = 1
# nlp = spacy.load("en_core_web_sm")
# f = open('sample.txt', mode='r')
# texts = f.readlines()
# txt = []
# for i in np.arange(len(texts)):
#     p = texts[i]
#     p = p[0:len(p)-1]
#     doc = nlp(p)
#     sents = []
#     for sent in doc.sents:
#         sents.append([sent,0,0,0])
#     txt.append(sents)
# f.close()
#
# s_pos = s_pos + pos
# print(txt)
# txt1=[]
# for p in txt:
#     for s in p:
#         txt1.append(s)
# print(txt1)
# if s_pos > len(sents[:][:]) - 1:
#     s_pos = 0
# if s_pos < 0:
#     s_pos = 0
# sents[0][s_pos][1] = 1
# sents[0][s_pos][1] = 2
# offset_x = 4
# offset_y = 1
# for p in txt:
#     for i in np.arange(len(p)):
#         total_size = x + len(p[i][0])
#         l = [line_size] * int(total_size / line_size)
#         l.append(total_size % line_size)
#         str_len = len(p[i][0])
#         st = 0
#         for j in np.arange(len(l)):
#             et = min((l[j] - x), str_len) + st
#             if j == 0:
#                 p[i][2] = y
#                 p[i][3] = x
#             print("string :{},st:{},et:{},y:{},x:{},color:{}".format(p[i][0],st,et,y+offset_y,x+offset_x,p[i][1]))
#             #txt_box.addstr(y + offset_y, x + offset_x, p[i][0][st:et], curses.color_pair(p[i][1]))
#             x = et - st + x if et - st + x < line_size else 0
#             if x == 0:
#                 y = y + 1
#             str_len = str_len - et + st
#             st = et
#     y = y+1
#     x = 0

# Curses modules

# !/usr/bin/env python

"""
Lyle Scott, III
lyle@digitalfoo.net

A simple demo that uses curses to scroll the terminal.
"""
from transformers import BertTokenizer
import torchtext.vocab as vocab
_glove = vocab.GloVe(name='6B', dim=300, cache='train/GloVe6B5429')
x = _glove.vectors[4]
print (type(x))