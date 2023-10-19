import torch

import pandas as pd
import torch.nn as nn

from typing import Iterable
from modules.loss import PHOSCLoss
from utils import get_map_dict


"""
NOTE: You need not change any part of the code in this file for the assignement.
"""
def train_one_epoch(model: torch.nn.Module, criterion: PHOSCLoss,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):

    model.train(True)

    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0
    for samples, targets, _ in dataloader:
        # Putting images and targets on given device
        samples = samples.to(device)
        targets = targets.to(device)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(samples)

        # calculating loss and backpropagation the loss through the network
        loss = criterion(outputs, targets)
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        print(f'loss: {loss.item()}, step progression: {batch}/{n_batches}, epoch: {epoch}')

        batch += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def accuracy_test(model, dataloader: Iterable, device: torch.device):
    model.eval()
    df = dataloader.dataset.df_all
    word_map = get_map_dict(list(set(df['Word'])))
    word_map = {w: torch.tensor(word_map[w]).to(device) for w in word_map}
    word_map_keys = list(word_map.keys())
    word_map_values = torch.stack(list(word_map.values()))
    n_correct = 0
    no_of_images = len(df)
    acc_by_len = dict()
    word_count_by_len = dict()

    for w in df['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    Predictions = []

    for samples, targets, words in dataloader:
        samples = samples.to(device)
        vector_dict = model(samples)
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        for i in range(len(words)):
            target_word = words[i]
            pred_vector = vectors[i].unsqueeze(0)
            cos_sim = torch.cosine_similarity(pred_vector, word_map_values, dim=1)
            pred_word = word_map_keys[torch.argmax(cos_sim).item()]
            Predictions.append((samples[i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])
    acc = n_correct / no_of_images

    return acc, df, acc_by_len


