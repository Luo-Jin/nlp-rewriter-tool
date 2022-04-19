from datetime import time
import glovetrain
import torch.utils.data as tud
import torch
import numpy as np
import scipy.spatial
import torchtext.vocab as vocab
'''
EMBEDDING_SIZE = 50		#50个特征
MAX_VOCAB_SIZE = 1000	#词汇表大小为2000个词语
WINDOW_SIZE = 5			#窗口大小为5
WEIGHT_FILE = "weight.txt"
NUM_EPOCHS = 10			#迭代10次
BATCH_SIZE = 10			#一批有10个样本
LEARNING_RATE = 0.05	#初始学习率
TEXT_SIZE = 20000000	#控制从语料库读取语料的规模

text, idx_to_word, word_to_idx, word_counts, word_freqs = glovetrain.getCorpus('train', size=TEXT_SIZE,MAX_VOCAB_SIZE=MAX_VOCAB_SIZE)    #加载语料及预处理
co_matrix = glovetrain.buildCooccuranceMatrix(text, word_to_idx,WINDOW_SIZE)    #构建共现矩阵
weight_matrix = glovetrain.buildWeightMatrix(co_matrix)             #构建权重矩阵
dataset = glovetrain.WordEmbeddingDataset(co_matrix, weight_matrix) #创建dataset
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
model = glovetrain.GloveModelForBGD(MAX_VOCAB_SIZE, EMBEDDING_SIZE) #创建模型
optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE) #选择Adagrad优化器

print_every = 10000
save_every = 50000
epochs = NUM_EPOCHS
iters_per_epoch = int(dataset.__len__() / BATCH_SIZE)
total_iterations = iters_per_epoch * epochs
print("Iterations: %d per one epoch, Total iterations: %d " % (iters_per_epoch, total_iterations))
#start = time.time()
for epoch in range(epochs):
    loss_print_avg = 0
    iteration = iters_per_epoch * epoch
    for i, j, co_occur, weight in dataloader:
        iteration += 1
        optimizer.zero_grad()   #每一批样本训练前重置缓存的梯度
        loss = model(i, j, co_occur, weight)    #前向传播
        loss.backward()     #反向传播
        optimizer.step()    #更新梯度
        loss_print_avg += loss.item()
torch.save(model.state_dict(), WEIGHT_FILE)


def find_nearest(word, embedding_weights):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return  [idx_to_word[i] for i in (cos_dis.argsort()[:10])]

glove_matrix = model.gloveMatrix()
print(find_nearest("minimal",glove_matrix))

print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])
cache_dir = "./GloVe6B5429"
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
print("一共包含%d个词。" % len(glove.stoi))
print(glove.stoi['beautiful'], glove.itos[3366])
print(glove.get_vecs_by_tokens("word",False))

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("You're a Ostrich.")
encoding_tokens = tokenizer.encode(tokens)
decoded_tokens = tokenizer.convert_ids_to_tokens(encoding_tokens)
input2 = {'encoding':encoding_tokens,'decoded':decoded_tokens,'original_tokens':tokens}
print(input2)

from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
print(unmasker("The man worked as a [MASK]."))
'''
import random
import copy
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from torch.nn import functional as F

text = "Language Modeling is the task of predicting the best word to follow or continue a sentence given all the words already in the sentence."
punctuations = ["[CLS]","[UNK]","[MASK]","[SEP]","[PAD]","'",'"',";",":",",",".","?","/",">","<","{","}"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
org_tokens = tokenizer(text, return_tensors="pt")
mask_pos = []
for i in range(len(org_tokens["input_ids"][0])):
    if tokenizer.convert_ids_to_tokens(org_tokens["input_ids"][0][i].item()) not in punctuations:
        mask_pos.append(i)
i = 1
tokens = copy.deepcopy(org_tokens)
while len(mask_pos) > 0 and i <= 8:
    i = i+1
    pos = random.sample(mask_pos,1)
    mask_pos.remove(pos[0])
    tokens["input_ids"][0][pos[0]] = tokenizer.mask_token_id
    logits = model(**tokens)
    logits = logits.logits
    softmax = F.softmax(logits, dim=-1)
    all_word_idx = torch.argmax(softmax[0],dim=-1)
    mask_word_idx = torch.argmax(softmax[0, pos[0], :])

print(text)
print(tokenizer.decode(all_word_idx))






#model = BertModel.from_pretrained("bert-base-uncased")


