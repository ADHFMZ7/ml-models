"""
Defines a singular neuron.



"""
import torch
from torch import nn

class Neuron:
    def __init__(self):
        self.w = torch.rand(2, 1)
        self.b = torch.rand(1)

    def forward(self, x):
        return torch.max((x @ self.w) + self.b, 0)

    def loss(self, x, y):
        loss = nn.CrossEntropyLoss()
        logits = self.forward(x)
        return loss(logits, y) 


data_x = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=torch.float32)

data_y = torch.tensor([
    0,
    0,
    0,
    1
], dtype=torch.float32)

def main():
    neuron = Neuron()


    for ix in range(len(data_x)):
        # print(neuron(data_x[ix]))
        # print(data_y[ix])
        print(neuron.loss(data_x[ix], data_y[ix]))


if __name__ == "__main__":
    main()
