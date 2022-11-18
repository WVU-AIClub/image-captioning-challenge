from network import Model, TransformerDecoder
from network import initialize_model
# from pytorch_pretrained_vit import ViT
from utils import plot_multiple_lists
from datetime import datetime
import logging
import torch
from datasets import load_dataset
from spacy.vocab import Vocab
import spacy
from collections import Counter
from data import BloomData
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F


mode = 1
vocab_len = 1000
imgs = torch.rand([32, 3, 224, 224])
root = "D:/Big Data/Bloom"

if mode == 0:
    captions = torch.randint(high=1000, size=(32, 196))
else:
    # iso639_3_letter_code = "hau"
    iso639_3_letter_code = "tha"
    # iso639_3_letter_code = "kir"
    dataset = BloomData(root=root, img_size=224, lang=iso639_3_letter_code, seq_len=196, priority='space')
    dl = DataLoader(dataset, batch_size=32)
    imgs, captions = next(iter(dl))
    vocab_len = len(dataset.word2ind)


model = Model(i_dim=224,
                p_dim=16,
                dim=768,
                n_heads=4,
                n_encoder_layers=3,
                n_decoder_layers=3,
                dropout=0,
                vocab_len=vocab_len,
                mode='train',
                device='cpu')

out = model(imgs, captions)

# item = out[0]
# cap = []
# for i in range(len(item)):
#     choice = torch.argmax(item[i])
#     cap.append(choice.item())

# cap = dataset.untokenize(cap)
# print(cap)
onehot = F.one_hot(captions.to(torch.int64), vocab_len).float()

# KL = torch.nn.KLDivLoss(reduction='batchmean')

loss = F.kl_div(torch.log(out), onehot, reduction='batchmean')

print(type(loss))
print(loss.item())

