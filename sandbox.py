from network import Model, TransformerDecoder
from network import initialize_model
# from pytorch_pretrained_vit import ViT
from utils import plot_multiple_lists
from datetime import datetime
import logging
import torch


# pretrained_vit = ViT('B_32', pretrained=True)
model = Model(i_dim=224,
                p_dim=16,
                dim=768,
                n_heads=4,
                n_encoder_layers=6,
                n_decoder_layers=6,
                dropout=0,
                device='cuda')
# model = initialize_model(model, pretrained_vit, n=12)

noise = torch.rand([32, 3, 224, 224])
caption = torch.randint(high=1000, size=(32, 196))

out = model(noise, caption)

print(out)
print(out.size())


