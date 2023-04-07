# This script simulates binary oporation (AND, OR, NOT) using a single perceptron neuron.
# This is not a self learning model.


# OR:
#     0, 0 -> 0
#     0, 1 -> 1
#     1, 0 -> 1
#     1, 1 -> 1
#
# AND:
#     0, 0 -> 0
#     0, 1 -> 0
#     1, 0 -> 0
#     1, 1 -> 1


import numpy as np

class Perceptron:

    def __init__(self):
        self.W = np.array([1/2, 1/2])

    def forward(self, x):
        return x @ self.W > 0.45


if __name__ == "__main__":

    por = Perceptron()


    print("0 v 0 -> ", por.forward(np.array([0, 0]).T))
    print("0 v 1 -> ", por.forward(np.array([0, 1]).T))
    print("1 v 0 -> ", por.forward(np.array([1, 0]).T))
    print("1 v 1 -> ", por.forward(np.array([1, 1]).T))
