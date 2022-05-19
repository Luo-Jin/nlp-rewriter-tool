# -*- coding: utf-8 -*-
###############################################
#   File    : cbs.py
#   Author  : Jin luo
#   Date    : 2022-04-23
#   Input   : generator.py -t <text> -s <similarity> -e <enforcement>
#             s=0.972 and e=0.01 if not specified
#   Output  : three revised copies of the original sentences
###############################################
'''
Description:
a flexible conditional bert sampling framework
Loop in each replaceable positions
  1. mask a random position in those copies.
  2. calculates:
     a.the word prob Plm by using BERT MASK Language bert_mlm_model, get tensor[3,30k]
     b.the enforcement prob Penforce by using pre-trained "word-piece-embeddings", get tensor[3,30k]
     c.the proposed prob for word Pproposal = element-wise add Plm and Penforce, get tensor[3,30k]
  3. sample words with top 3 probs in Pproposal, (we could just use the top1)
     and fill the rank 1 word id in masked position in the first copy
         fill the rank 2 word id  in masked position in the second copy
         fill the rank 3 word id  in masked position in the third copy
  4. decode the three copies
End loop
'''


import re
import sys
import getopt
import random
import torch
import copy
from os import system, name
from embeddings import bert_tokenizer,bert_mlm_model,word_piece_embeddings,stanza_model

def main():
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
    texts = read_text(arg_txt)
    print(texts)
    clear_screen()
    print("\033[1;33mThe original sentence is :\033[0m")
    print("\033[7m\n{}\n\033[0m".format(texts))
    tokens = get_sentence(texts, σ, k,batch)
    print("\033[1;33mThe sentence is revised with smooth parameter k={} "
          "and similarity rate σ={} :\033[0m".format(k,σ))
    for i in torch.arange(len(tokens["input_ids"])):
        print("{}{}{}"
              .format(i + 1
                      , "."
                      , bert_tokenizer.decode(tokens["input_ids"][i], skip_special_tokens=True)))


def read_text(txt):
    f = open(txt, mode='r')
    texts = f.readline()
    f.close()
    return texts

# define our clear function
def clear_screen():
    if name == 'nt':
        _ = system('cls')
        # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
        # print out some text



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
    tokens["input_ids"][:, pos[0]] = bert_tokenizer.mask_token_id
    logits = bert_mlm_model(**tokens)
    logits = logits.logits
    softmax = torch.softmax(logits, dim=-1)
    Plm = softmax[:, pos[0]]
    return Plm


def get_sentence(txt,σ=0.975,k=0.1,batch=1):
    # set minibatch size of this task, determine how many sentences will be created in one call.
    text = [txt] * batch
    org_tokens = bert_tokenizer(text, return_tensors="pt"
                           ,return_token_type_ids=False
                           ,return_attention_mask=False
                           )
    special_tokens = {"[CLS]": 0, "[UNK]": 0, "[MASK]": 0, "[SEP]": 0, "[PAD]": 0
        , "'": 0, '"': 0, ";": 0, ":": 0, ",": 0,".": 0, "?": 0
        , "/": 0, ">": 0, "<": 0, "{": 0, "}": 0,"-":0,"+":0,"=":0,"_":0
        , "!":0,"@":0,"#":0,"$":0,"%":0,"^":0,"&":0,"*":0,"(":0,")":0}
    special_tokens = {k:bert_tokenizer.convert_tokens_to_ids(k) for k,v in special_tokens.items()}
    # determine all replaceable positions in the sentence.
    mask_pos = []
    for i in range(len(org_tokens["input_ids"][0])):
        id = org_tokens["input_ids"][0][i]
        w = bert_tokenizer.ids_to_tokens[id.item()]
        re.fullmatch('##[0-9]*', w)  # determine if it is a number
        doc = stanza_model(w)  # determine if it is an entity
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
        #progress_counter = progress_counter + 1
        # pick a random word in k position and mask it
        pos = random.sample(mask_pos,1)
        mask_pos.remove(pos[0])
        # calculate the prob of enforcement
        Penforce = penforce(batch,pos,org_tokens,tokens,k,σ)

        # calculate the P(lm), it is a token_size * vocal_size matrix
        # replaced_word_idx = copy.deepcopy(tokens["input_ids"][:,pos[0]])
        Plm = plm(pos,tokens)

        # calculate the proposal prob
        Pproposal = Plm + Penforce
        ppl_word_idx = torch.topk(Pproposal,k=batch,dim=-1)
        y = torch.diag_embed(torch.ones(batch,dtype=torch.long))
        sample_idx = torch.sum(ppl_word_idx.indices * y,dim=-1)
        index = [i for i in range(len(sample_idx))]
        random.shuffle(index)
        tokens["input_ids"][:,pos[0]] = sample_idx
    return tokens

if __name__ == "__main__":
    main()

