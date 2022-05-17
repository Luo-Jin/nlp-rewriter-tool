# -*- coding: utf-8 -*-
###############################################
#   File    : generator.py
#   Author  : Jin luo
#   Date    : 2022-04-23
#   Input   : generator.py -t <text> -s <similarity> -e <enforcement>
#             s=0.972 and e=0.01 if not specified
#   Output  : three revised copies of the original sentences
###############################################
'''
Description:
this script makes three copies of original sentence.
Loop in each replaceable positions
  1. mask a random position in those copies.
  2. calculates:
     a.the word prob Plm by using BERT MASK Language Model, get tensor[3,30k]
     b.the enforcement prob Penforce by using pre-trained "word-piece-embeddings", get tensor[3,30k]
     c.the proposed prob for word Pproposal = element-wise add Plm and Penforce, get tensor[3,30k]
  3. sample words with top 3 probs in Pproposal, (we could just use the top1)
     and fill the rank 1 word id in masked position in the first copy
         fill the rank 2 word id  in masked position in the second copy
         fill the rank 3 word id  in masked position in the third copy
  4. decode the three copies
End loop
'''
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from os import system, name
import re
import sys
import getopt
import random
import torch
import copy
from configparser import ConfigParser
import os
import curses
import spacy
import stanza


# render the screen
screen = curses.initscr()
t = screen.getmaxyx()
progress_bar = curses.newwin(1,40,0,0)
progress_bar.mvwin(int(t[0]/2)-1,int(t[1]/2)-20)
progress_bar.addstr(0,0,"Loading embeddings...")
progress_bar.refresh()
# load the embeddings
file_path = os.path.join(os.path.abspath("."), "rewriter.ini")
config = ConfigParser()
config.read(file_path)
tokenizer = BertTokenizer.from_pretrained(config.get("PRE_TRAINED","bert_vocal"))
model = BertForMaskedLM.from_pretrained(config.get("PRE_TRAINED","bert_model"))
word_piece_embeddings = torch.from_numpy(torch.load(config.get("PRE_TRAINED","embeddings"))).t()
#stanza.download('en',model_dir='stanza')
en_nlp = stanza.Pipeline('en',dir=config.get("PRE_TRAINED","stanza_model"),processors='tokenize,ner')
nlp = spacy.load("en_core_web_sm")
progress_bar.clear()
progress_bar.mvwin(config.getint('OUTPUT_BOX','y') + int(config.getint('OUTPUT_BOX','row')/2)-1
                   ,config.getint('OUTPUT_BOX','x') + int(config.getint('OUTPUT_BOX','col')/2)-4)
progress_counter = 0

def main():
    global tokenizer,model,word_piece_embeddings
    arg_sim = None
    arg_enf = None
    arg_txt = None
    arg_batch = 0
    arg_help = "{0} -t <text> -s <similarity> -e <enforcement> -b <batch>".format(sys.argv[0])

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:s:e:b:", ["help", "text=","similar=", "enforce=","batch="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-t", "--text"):
            arg_txt = arg
        elif opt in ("-s", "--similar"):
            arg_sim = arg
        elif opt in ("-e", "--enforce"):
            arg_enf = arg
        elif opt in ("-b", "--batch"):
            arg_batch = arg
    σ = 0.975 if arg_sim is None else float(arg_sim)
    k = 0.1 if arg_enf is None else float(arg_enf)
    batch = 1 if arg_batch is None else int(arg_batch)

    # read text from specific txt file
    texts = readTXT(arg_txt)
    print(texts)
    clearScreen()
    print("\033[1;33mThe original sentence is :\033[0m")
    print("\033[7m\n{}\n\033[0m".format(texts))
    tokens = getSentence(texts, σ, k,batch)
    print("\033[1;33mThe sentence is revised with smooth parameter k={} "
          "and similarity rate σ={} :\033[0m".format(k,σ))
    for i in torch.arange(len(tokens["input_ids"])):
        print("{}{}{}"
              .format(i + 1
                      , "."
                      , tokenizer.decode(tokens["input_ids"][i], skip_special_tokens=True)))


def readTXT(txt):
    f = open(txt, mode='r')
    texts = f.readline()
    f.close()
    return texts

# define our clear function
def clearScreen():
    if name == 'nt':
        _ = system('cls')
        # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
        # print out some text

def updateProgress(progress):
    rangex = (50 / float(100)) * progress
    pos = int(rangex)
    display = '#'
    if pos != 0:
        progress_bar.addstr(0, pos, "{}".format(display))
        progress_bar.refresh()

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


def getSentence(txt,σ=0.975,k=0.1,batch=1):
    global progress_counter
    # set minibatch size of this task, determine how many sentences will be created in one call.
    progress_bar.clear()
    progress_bar.box()
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
    #progress_bar.erase()
    while len(mask_pos) > 0 and i <= len(tokens["input_ids"][0]):
        i = i+1
        progress_counter = progress_counter + 1
        # pick a random word in k position and mask it
        pos = random.sample(mask_pos,1)
        mask_pos.remove(pos[0])
        # calculate the prob of enforcement
        Penforce = penforce(batch,pos,org_tokens,tokens,k,σ)

        # calculate the P(lm), it is a token_size * vocal_size matrix
        replaced_word_idx = copy.deepcopy(tokens["input_ids"][:,pos[0]])
        Plm = plm(pos,tokens)

        # calculate the proposal prob
        Pproposal = Plm + Penforce
        ppl_word_idx = torch.topk(Pproposal,k=batch,dim=-1)
        y = torch.diag_embed(torch.ones(batch,dtype=torch.long))
        sample_idx = torch.sum(ppl_word_idx.indices * y,dim=-1)
        index = [i for i in range(len(sample_idx))]
        random.shuffle(index)

        tokens["input_ids"][:,pos[0]] = sample_idx
        updateProgress(progress_counter)
    progress_counter = 0
    return tokens

if __name__ == "__main__":
    main()

