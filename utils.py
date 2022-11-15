import os
import logging
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import torch
import numbers

def load_access_token():
    with open('secrets.txt', 'r') as f:
        token = f.read()

    return token


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

def create_shifted_output_mask(seq, n_heads=4):
    """Creates a mask that prevents the decoder to attend future outputs.
    
    For each sample in the provided batch, the created mask is a square matrix that contains one row for every
    position in the output sequence. Each of these rows indicates those parts of the sequence that may be considered in
    order to compute the respective output, i.e., those output values that have been computed earlier.
    
    Args:
        seq (torch.Tensor): The output sequence that the padding is mask is created for. ``seq`` has to be a tensor of
            shape (batch-size x seq-len x ...), i.e., it has to have at least two dimensions.
    
    Returns:
        torch.ByteTensor: A binary mask where ``1``s represent tokens that should be considered for the respective
            position and ``0``s indicate future outputs. The provided mask has shape (batch-size x seq-len x seq-len).

    From: https://github.com/phohenecker/pytorch-transformer/blob/master/src/main/python/transformer/util.py
    """
    # sanitize args
    if not isinstance(seq, torch.Tensor):
        raise TypeError("<seq> has to be a Tensor!")
    if seq.dim() < 2:
        raise ValueError("<seq> has to be at least a 2-dimensional tensor!")
    
    batch_size = seq.size(0)
    seq_len = seq.size(1)
    
    # create a mask for one sample
    mask = 1 - seq.new(seq_len, seq_len).fill_(1).triu(diagonal=1).byte()
    
    # copy the mask for all samples in the batch
    mask = mask.unsqueeze(0).expand(batch_size * n_heads, -1, -1)
    
    return mask

def shift_output_sequence(seq, zero_range=1e-22):
    """Shifts the provided output sequence one position to the right.
    
    To shift the sequence, this function truncates the last element of and prepends a zero-entry to every sample of
    the provided batch. However, to prevent ``nan`` values in the gradients of tensors created by means of
    ``torch.std``, the prepended tensors are not actually set to 0, but sampled uniformly from a tiny interval around 0,
    which may be adjusted via the arg ``zero_range``.
    
    Args:
        seq (torch.Tensor): The sequence to shift as (batch-size x seq-length x dim-model)-tensor.
        zero_range (numbers.Real, optional): Specifies the range to sample zero-entries from as closed interval
            [``zero_range``, ``-zero_range``].
    
    Returns:
        torch.Tensor: The shifted sequence, which, just like ``seq``, is a (batch-size x seq-length x dim-model)-tensor.

    From: https://github.com/phohenecker/pytorch-transformer/blob/master/src/main/python/transformer/util.py
    """
    # sanitize args
    if not isinstance(seq, torch.Tensor):
        raise TypeError("<seq> has to be a tensor!")
    if seq.dim() != 3:
        raise ValueError("Expected <seq> to be 3D, but {} dimensions were encountered!".format(seq.dim()))
    if not isinstance(zero_range, numbers.Real):
        raise TypeError("The <zero_range> has to be a real number!")
    zero_range = float(zero_range)
    if zero_range <= 0:
        raise ValueError("The <zero_range> has to be a positive number!")
    
    return torch.cat(
            [
                    seq.new(seq.size(0), 1, seq.size(2)).uniform_(-zero_range, zero_range),
                    seq[:, :-1, :]
            ],
            dim=1
    )