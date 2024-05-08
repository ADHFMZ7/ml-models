import torch
from torch import nn

"""
Model from Bengio et al. (2003) - A Neural Probabilistic Language Model

The model is a simple feedforward neural network with a single hidden layer.
It takes in a sequence of words and predicts the next word in the sequence.

"""

class Model(nn.Module):
    def __init__(self, vocab_size, context_len=8, feature_len=2):
        super().__init__()
        self.C = torch.randn(vocab_size, feature_len)
        self.W1 = torch.randn(context_len * feature_len, 100)
        self.b1 = torch.randn(100)
        self.W2 = torch.randn(100, vocab_size)
        self.b2 = torch.randn(vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y):
        emb = self.C[x].view(-1, 16)
        h = torch.tanh(emb @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        preds = torch.softmax(logits, dim=1)
        loss = self.criterion(preds, y)
        
        return preds, loss