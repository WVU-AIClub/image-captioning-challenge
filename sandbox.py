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
                n_transformer_layers=12,
                dropout=0,
                device='cuda')
# model = initialize_model(model, pretrained_vit, n=12)

noise = torch.rand([32, 3, 224, 224])

out = model(noise)

print(out)
print(out.size())

decoder = TransformerDecoder(dim=768, n_heads=4, n_layers=4, dropout=0)
out = decoder(out, out, padding_mask=None, shifted_output_mask=None)

print(out)
print(out.size())