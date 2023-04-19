# Transformers

This Folder contains my notes on the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. It also will contain my implementation of a transformer.

My implementation will then be used to create an elementary language model using data.txt, a concatenation of several books into one txt file.

## Notes

### The Transformer

The Transformer is a model that uses a mechanism called multi-headed self-attention to enconde one sequence to an intermediate vector, and decode it into a different sequence.

An example of this is language translation

[English text] -> encode -> intermediate vector -> decode -> [french text]

Before transformers, this problem was solved using [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network). This was later replaced

The Transformer has been used with good result in many different areas of machine learning.
