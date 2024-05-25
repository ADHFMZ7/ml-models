import torch
import torch.nn as nn
from util import get_cifar100_data as get_data
from model import ResNet

epochs = 100
criterion = nn.CrossEntropyLoss()

X, Y = get_data()

for epoch in range(epochs):

    for batch, (inputs, labels) in enumerate(X):

        # zero gradients

        # compute logits

        # compute loss

        # backward pass

        # update weights

        # logging
        

        break
    break


