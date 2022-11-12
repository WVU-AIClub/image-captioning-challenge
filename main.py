import json
import logging
import torch
from torch.optim import Adam, SGD
from pytorch_pretrained_vit import ViT
from utils import create_logfile, count_parameters
from network import Model
from network import initialize_model

with open('config.json', 'r') as f:
    config = json.load(f)

create_logfile(config['log_path'])

logging.info(f'Starting training with config:')
for key in config:
    logging.info(f'\t{key}: {config[key]}')

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'\tDevice: {device}')

# Load Dataset and DataLoader(s) here

# Initialize model
pretrained_vit = ViT('B_32', pretrained=True)
model = Model(i_dim=config['img_dim'],
                p_dim=config['patch_dim'],
                dim=config['hid_dim'],
                n_heads=config['n_heads'],
                n_transformer_layers=config['n_transformer_layers'],
                dropout=config['dropout'],
                device=device)
model = initialize_model(model, pretrained_vit, n=config['n_layer_init'])

num_params = count_parameters(model)
logging.info(f'\nModel successfully loaded with {num_params} trainable parametrs\n')

# Define optimizers
if config['optim']['type'] == 'Adam':
    optim = Adam(model.parameters(), lr=config['optim']['lr'], 
                 beta1=config['optim']['beta1'], beta2=config['optim']['beta2'])

logging.info('Beginning training loop...')
for epoch in range(config['epochs']):
    logging.info(f'Epoch {epoch} -----------------')
    # Run train step
    print(epoch)


