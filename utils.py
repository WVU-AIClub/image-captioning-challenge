import os
import logging
from datetime import datetime
import sys
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_logfile(dir):
    now = datetime.now().strftime("%m%d%Y_%H%M")
    filename = f'{now}.log'
    path = os.path.join(dir, filename)

    logging.basicConfig(filename=path, format ='%(message)s', encoding='utf-8', level=logging.INFO)

def plot_list(dir, title, x_label, y_label, list_, label):
    plt.figure(figsize=(10, 5)) 
    plt.title(title)
    
    plt.plot(list_, label=label)

        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    filename = os.path.join(dir, f'{title}.png')
    filename = "metrics/" + title + ".png"
    
    plt.savefig(filename)
    plt.close()

def plot_multiple_lists(dir, title, x_label, y_label, lists, labels):
    plt.figure(figsize=(10, 5)) 
    plt.title(title)
    
    for list_, label in zip(lists, labels):
        plt.plot(list_, label=label)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    filename = os.path.join(dir, f'{title}.png')
    
    plt.savefig(filename)
    plt.close()