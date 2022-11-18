import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
import logging
import time
from utils import plot_list
import os

def train(model, dataloader, optim, device, config, vocab_len):
    logging.info('Beginning training loop...')

    KL = KLDivLoss(reduction='batchmean')

    total_loss = []
    for epoch in range(config['epochs']):
        logging.info(f'Epoch {epoch} -----------------')
        print(f"Starting epoch {epoch}.........")

        start = time.time()
        epoch_loss = []
        for batch, (images, captions) in enumerate(dataloader):
            optim.zero_grad()

            # Send data to cpu or cuda
            images, captions = images.to(device), captions.to(device)

            # Make predication
            pred = model(images, captions)

            # One-hot encode caption and compute loss
            onehot = F.one_hot(captions.to(torch.int64), vocab_len).float()
            loss = F.kl_div(pred, onehot, reduction='batchmean')
            epoch_loss.append(loss)

            if batch % 1 == 0:
                print(f"Loss [{batch}/{len(dataloader)}]: {loss} ")

            # Backprop
            loss.backward()
            optim.step()

        end = time.time()
        total_time = end - start
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        logging.info(f'Avg Loss: {avg_loss}\nTime Elapsed: {total_time}s\n')

        total_loss += epoch_loss
        plot_list(config['graph_path'], "Loss over time", x_label="Epoch",
                     y_label="KL_Div Loss", list_=total_loss, label=None)

        print(f"Avg loss  : {avg_loss}\nTotal Time: {total_time}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config['model_path'], config['model_name']))
