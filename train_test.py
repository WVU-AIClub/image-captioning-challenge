import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
import logging
import time
from utils import plot_list

def train(model, dataloader, optim, device, config, vocab_len):
    logging.info('Beginning training loop...')

    total_loss = []
    for epoch in range(config['epochs']):
        logging.info(f'Epoch {epoch} -----------------')

        start = time.time()
        epoch_loss = []
        for batch, (images, captions) in enumerate(dataloader):

            # Send data to cpu or cuda
            images, captions = images.to(device), captions.to(device)

            # Make predication
            pred = model(images, captions)

            # One-hot encode caption and compute loss
            onehot = F.one_hot(captions.to(torch.int64), vocab_len)
            loss = KLDivLoss(torch.log(pred), onehot)
            epoch_loss.append(loss)

            # Backprop
            optim.zero_grad()
            loss.backward()
            optim.step()

        end = time.time()
        total_time = end - start
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        logging.info(f'Avg Loss: {avg_loss}\nTime Elapsed: {total_time}s')

        total_loss += epoch_loss
        plot_list(config['graph__path'], "Loss over time", x_label="Epoch",
                     y_label="KL_Div Loss", list=total_loss, label=None)
