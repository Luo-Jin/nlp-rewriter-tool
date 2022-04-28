from transformers import BertTokenizer
from transformers import BertForMaskedLM
import stanza
import torchtext.vocab as vocab
import sys
import getopt
import random
import torch
import copy

# cache_dir = 'GloVe6B5429'
# glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
# download bert embeddings
tokenizer = BertTokenizer.from_pretrained('./Bert/vocabulary')
model = BertForMaskedLM.from_pretrained("./Bert/model/maskedLM")
word_piece_embeddings = torch.load('word-piece-embedding.txt').t()
# download English model
# stanza.download('en')


def main():
    arg_sim = None
    arg_enf = None
    arg_txt = None
    arg_help = "{0} -t <text> -s <similarity> -e <enforcement>".format(sys.argv[0])

    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:s:e:", ["help", "text=","similar=", "enforce="])
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
    σ = 0.975 if arg_sim is None else float(arg_sim)
    k = 0.1 if arg_enf is None else float(arg_enf)
    text = readtxt(arg_txt)

    rewriter(text,σ,k)


def readtxt(txt):
    f = open(txt, mode='r')
    txt = f.readline()
    return txt


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
    # replace choosen word with [MASK]
    tokens["input_ids"][:, pos[0]] = tokenizer.mask_token_id
    logits = model(**tokens)
    logits = logits.logits
    softmax = torch.softmax(logits, dim=-1)
    Plm = softmax[:, pos[0]]
    return Plm

def rewriter(txt,σ=0.975,k=0.1,batch=3):
    text = [txt] * batch
    org_tokens = tokenizer(text, return_tensors="pt")
    punctuations = {"[CLS]": 0, "[UNK]": 0, "[MASK]": 0, "[SEP]": 0, "[PAD]": 0, "'": 0, '"': 0, ";": 0, ":": 0, ",": 0,
                    ".": 0, "?": 0, "/": 0, ">": 0, "<": 0, "{": 0, "}": 0}
    punctuations = {k:tokenizer.convert_tokens_to_ids(k) for k,v in punctuations.items()}
    # track all replaceable word's position
    mask_pos = []
    for i in range(org_tokens["input_ids"][0].size(0)):
        if org_tokens["input_ids"][0][i] not in punctuations.values():
            mask_pos.append(i)
    # iteratively replace the mask words
    i = 1
    tokens = copy.deepcopy(org_tokens)
    while len(mask_pos) > 0 and i <= 3:#len(tokens["input_ids"][0]):
        i = i+1
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
        # print ("origin:{},ppl:{}".format(tokenizer.decode(replaced_word_idx)
        #                                         ,tokenizer.decode(sample_idx)))
        # replace the MASK with proposed words
        tokens["input_ids"][:,pos[0]] = sample_idx


    print("The original sentence is :\n{}".format(txt))
    print("The alternative sentences are :")
    for i in torch.arange(len(tokens["input_ids"])):
        print("{}{}{}"
               .format(i+1
                       ,"."
                       ,tokenizer.decode(tokens["input_ids"][i],skip_special_tokens=True)))

    #print(tokenizer.decode(tokens["input_ids"],skip_special_tokens=True))


if __name__ == "__main__":
    sys.exit(main())

