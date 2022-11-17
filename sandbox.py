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

mode = 1
vocab_len = 1000
noise = torch.rand([8, 3, 224, 224])

if mode == 0:
    captions = torch.randint(high=1000, size=(32, 196))
else:
    # iso639_3_letter_code = "hau"
    iso639_3_letter_code = "tha"
    # iso639_3_letter_code = "kir"
    dataset = BloomData(img_size=128, lang=iso639_3_letter_code)
    dl = DataLoader(dataset, batch_size=8)
    captions = next(iter(dl))
    vocab_len = len(dataset.vocab)

model = Model(i_dim=224,
                p_dim=16,
                dim=768,
                n_heads=4,
                n_encoder_layers=3,
                n_decoder_layers=3,
                dropout=0,
                vocab_len=vocab_len,
                mode='eval',
                device='cpu')

out = model(noise, captions)
print(out)

print()

print(out[0])

