import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(
        self,
        x,
        mask=None
    ):

        # x shape:
        # [batch_size, sequence_length, embedding_dim]

        Q = self.query_projection(x)

        K = self.key_projection(x)

        V = self.value_projection(x)

        # Attention Scores

        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention_scores = attention_scores / (
            self.embedding_dim ** 0.5
        )

        # Apply causal mask

        if mask is not None:

            attention_scores = attention_scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        # Softmax

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        # Context vectors

        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights