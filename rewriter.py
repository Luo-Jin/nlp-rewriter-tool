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
    txt = readtxt(arg_txt)

    rewriter(txt,σ,k)


def readtxt(txt):
    f = open(txt, mode='r')
    txt = f.readline()
    return txt


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
    while len(mask_pos) > 0 and i <= len(tokens["input_ids"][0]):
        i = i+1
        # [pick a random word in k position and mask it
        pos = random.sample(mask_pos,1)
        mask_pos.remove(pos[0])
        # calculate the P(enforce).
        c = word_piece_embeddings[tokens["input_ids"]]
        x = word_piece_embeddings[org_tokens["input_ids"]]
        Rx = torch.ones(batch,word_piece_embeddings.size(0),word_piece_embeddings.size(1))
        Rx = Rx * torch.sum(x,dim=1)
        c = c[torch.arange(c.size(0)) != pos[0]]
        c = torch.sum(c, dim=0)
        Ru = word_piece_embeddings + c
        s = torch.cosine_similarity(Ru,Rx,dim=1)

        Penforce = torch.exp(-k*torch.max(torch.zeros(s.size(0)),(float(σ)-s)))

        # calculate the P(lm), it is a token_size * vocal_size matrix
        replaced_word_idx = copy.deepcopy(tokens["input_ids"][0][pos[0]])
        tokens["input_ids"][0][pos[0]] = tokenizer.mask_token_id
        logits = model(**tokens)
        logits = logits.logits
        softmax = torch.softmax(logits, dim=-1)
        Plm = softmax[0][pos[0]]

        # calculate Pproposal
        Pproposal = Plm + Penforce
        # print("{},{}".format(sum(s),s.shape))
        # print("max of s:{}, min of s:{}".format(s.max(),s.min()))
        # print("σ :%s"%σ)
        print("tokens:{},Plm:{},Pen:{},Pro:{}".format(replaced_word_idx
                                                      ,torch.argmax(Plm)
                                                      ,torch.argmax(Penforce)
                                                      ,torch.argmax(Pproposal)))
        plm_word_idx = torch.argmax(Plm)
        ppl_word_idx = torch.argmax(Pproposal)
        # print ("origin:{},plm:{},ppl:{}".format(tokenizer.decode(replaced_word_idx)
        #                                         ,tokenizer.decode(plm_word_idx)
        #                                         ,tokenizer.decode(ppl_word_idx)))
        tokens["input_ids"][0][pos[0]] = ppl_word_idx


    # print(text)
    # print(tokenizer.decode(tokens["input_ids"][0],skip_special_tokens=True))


if __name__ == "__main__":
    sys.exit(main())

