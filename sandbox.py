from network import Model
from network import initialize_model
from pytorch_pretrained_vit import ViT
from utils import count_parameters
from datetime import datetime
import logging

# vit = ViT('B_32', pretrained=True)
# model = Model(i_dim=(224, 224), p_dim=(16, 16), dim=768, n_heads=12, n_transformer_layers=12, device='cuda', dropout=0)

# model = initialize_model(model=model, pretrained=vit, model_type=1)

# num_params = count_parameters(model)
# print(num_params)

# date, time = datetime.now().split(' ')
logger = logging.getLogger('test')
