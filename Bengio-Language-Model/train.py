import torch

path = "../data/russian-novels/WAP.txt"


def load_data(context_len: int = 10):

    X, Y = [], []

    with open(path, 'r') as file:
        text = file.read()

    for i in range(len(text[:len(text) - context_len - 1])):

        x = text[i:context_len + i]
        y = text[i + 1:context_len + 1 + i]

        for t in range(context_len):
            context = x[:t+1]
            target = y[t] 
            X.append(context)
            Y.append(target)

    return X, Y

data = load_data()
print(data[0][:100])

