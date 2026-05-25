import torch.nn as nn


def TokenEmbedding(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim)
