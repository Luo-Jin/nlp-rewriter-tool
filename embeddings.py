# -*- coding: utf-8 -*-
###############################################
#   File    : nlp.py
#   Author  : Jin luo
#   Date    : 2022-05-18
#   Input   :
#   Output  :
###############################################
from transformers import BertTokenizer
from transformers import BertForMaskedLM
import torch
from configparser import ConfigParser
import os
import spacy
import stanza

# load the embeddings
file_path = os.path.join(os.path.abspath("."), "refrazer.ini")
config = ConfigParser()
config.read(file_path)
bert_tokenizer = BertTokenizer.from_pretrained(config.get("PRE_TRAINED","bert_vocal"))
bert_mlm_model = BertForMaskedLM.from_pretrained(config.get("PRE_TRAINED","bert_model"))
word_piece_embeddings = torch.from_numpy(torch.load(config.get("PRE_TRAINED","embeddings"))).t()
stanza.download('en',model_dir='stanza')
stanza_model = stanza.Pipeline('en',dir=config.get("PRE_TRAINED","stanza_model"),processors='tokenize,ner')
spacy_model = spacy.load("en_core_web_sm")