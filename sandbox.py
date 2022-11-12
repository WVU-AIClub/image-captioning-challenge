from network import Model
from network import initialize_model
from pytorch_pretrained_vit import ViT
from utils import plot_multiple_lists
from datetime import datetime
import logging

l1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
l2 = [2, 4, 6, 8, 2, 4, 6, 8, 2]
l3 = [9, 8, 7, 6, 5, 4, 3, 2, 1]

plot_multiple_lists('graphs', 'Test', 'x axis', 'y axis', [l1, l2, l3], ['l1', 'l2', 'l3'])
