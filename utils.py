import os
import logging
from datetime import datetime
import sys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_logfile(dir):
    now = datetime.now().strftime("%m%d%Y_%H%M")
    filename = f'{now}.log'
    path = os.path.join(dir, filename)

    logging.basicConfig(filename=path, format ='%(message)s', encoding='utf-8', level=logging.INFO)