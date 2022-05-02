import stanza
#stanza.download('en',model_dir='stanza')       # This downloads the English models for the neural pipeline
# nlp = stanza.Pipeline('en',model_dir='stanza',processors='tokenize,ner') # This sets up a default neural pipeline in English
# doc = nlp("Auckland")
# print(*[f'token: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')



from matplotlib  import pyplot as plt
import torch
l = {}
#x = torch.tensor([[torch.from_numpy(t) for t  in torch.load('test/sig_l1_e1000_b5000_l0.5.pt')]])
l['L1Loss, Sigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/sig_l1_e1000_b5000_l0.5.pt')])
l['L1Loss, noSigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/nosig_l1_e1000_b5000_l0.5.pt')])
l['SmoothL1Loss, Sigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/sig_sml1_e1000_b5000_l0.5.pt')])
l['SmoothL1Loss, noSigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/nosig_sml1_e1000_b5000_l0.5.pt')])
i = 0
for k,v in l.items():
    i = i + 1
    v = v.view(int(v.size(0) / 80), 80)
    ax = plt.subplot(220+i)
    ax.set_title(k)
    plt.plot(torch.arange(v.size(0)),torch.mean(v,dim=-1))
    plt.xlabel('epoch')
    plt.ylabel('loss (mean)')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()





# plt.subplot(221)
# plt.subplot(222)
# plt.subplot(223)
# plt.subplot(224)
#
# plt.show()

#
# plt.plot(torch.arange(x.size(0)),torch.mean(x,dim=-1))
# plt.title('Sigmoid, loss=L1Loss, epoch=1000, batch_size=5000, lr=0.5')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

