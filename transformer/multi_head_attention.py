import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):

        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim

        self.num_heads = num_heads

        self.head_dim = (
            embedding_dim // num_heads
        )

        # Combined QKV projections

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

        # Final output projection

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def split_heads(
        self,
        x
    ):

        batch_size, seq_len, embedding_dim = x.size()

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        x = x.transpose(1, 2)

        return x

    def combine_heads(
        self,
        x
    ):

        batch_size, num_heads, seq_len, head_dim = x.size()

        x = x.transpose(1, 2)

        x = x.contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        return x

    def forward(
        self,
        x,
        mask=None
    ):

        Q = self.query_projection(x)

        K = self.key_projection(x)

        V = self.value_projection(x)

        # Split into heads

        Q = self.split_heads(Q)

        K = self.split_heads(K)

        V = self.split_heads(V)

        # Attention scores

        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention_scores = attention_scores / (
            self.head_dim ** 0.5
        )

        # Apply mask

        if mask is not None:

            mask = mask.unsqueeze(0).unsqueeze(0)

            attention_scores = attention_scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        # Softmax

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        # Weighted values

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        # Combine heads

        output = self.combine_heads(
            attention_output
        )

        # Final projection

        output = self.output_projection(
            output
        )

        return output, attention_weights