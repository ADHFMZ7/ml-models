"""
Generates a dataset of random arithmetic expressions

To be used in a language model that outputs the answer given an expression
"""
import torch
import random

ops = '*/+-'


def gen_expr(max_len=100):
    res = []
    res.append(str(random.randint(-10000, 10000)))

    for i in range(1, random.randint(2, max_len)):
        res.append(random.choice(ops))
        res.append(str(random.randint(-10000, 10000)))

    return res


def encode(expr):
    seq = []
    for char in "".join(expr):
        if char in ops:
            seq.append(10000+ops.index(char))
        else:
            seq.append(int(char))
    seq += [-1] * (2048 - len(seq))
    return seq 

def decode(seq):
    ...

def next_batch(batch_size: int, seq_len: int):
    xs, ys = [], []
    for example in range(batch_size):
        ex = gen_expr(seq_len)
        xs.append(encode(ex))
        ys.append(eval("".join(ex)))
        print(ex)
    print(xs)
    return torch.tensor(xs), torch.tensor(ys)


next_batch(1, 10)
