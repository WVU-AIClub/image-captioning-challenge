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


# pretrained_vit = ViT('B_32', pretrained=True)
# model = Model(i_dim=224,
#                 p_dim=16,
#                 dim=768,
#                 n_heads=4,
#                 n_encoder_layers=6,
#                 n_decoder_layers=6,
#                 dropout=0,
#                 device='cuda')
# # model = initialize_model(model, pretrained_vit, n=12)

# noise = torch.rand([32, 3, 224, 224])
# caption = torch.randint(high=1000, size=(32, 196))

# out = model(noise, caption)

# print(out)
# print(out.size())

# Add the relevant ISO code for the language you want to work with.
# iso639_3_letter_code = "hau"
iso639_3_letter_code = "tha"
# iso639_3_letter_code = "kir"

dataset = BloomData(img_size=128, lang=iso639_3_letter_code)

train = dataset.dataset['train']
caption = train[123]['caption']

tokenized = dataset.tokenize(caption)
print(caption)
print(tokenized)

