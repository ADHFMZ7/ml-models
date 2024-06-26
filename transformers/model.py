import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        


    def forward(self):
        ...


class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention
    """

    def __init__(self, head_size, embdim, kdim, vdim) -> None: 
        super().__init__()

        self.key = nn.Linear(embdim, kdim)
        self.query = nn.Linear(embdim, kdim)
        self.value = nn.Linear(embdim, vdim)

        self.mask = self.register_buffer("Mask", F.softmax(torch.zeros().masked_fill(torch.tril(torch.ones()) == 0, -float('inf'))))

        self.softmax = nn.Softmax()
        
    def forward(self, x):

        B, T, C = x.shape
        # B - batch dimension; batch size
        # T - Time dimension; the sequence length
        # C - Channel dimension; the embedding size

        key = self.key(x) 
        query = self.query(x)
        value = self.value(x)

        out = (query @ key.T) / (C ** -0.5) 


        out = self.softmax(out) @ value
         

class MultiHeadAttention(nn.Module):
    """ 
    multiple heads of self-attention in parallel 
    """

    def __init__(self, num_heads, head_size):
        ...

    def forward(self, x):
        ...
