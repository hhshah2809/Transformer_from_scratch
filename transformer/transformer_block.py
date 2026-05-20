import torch.nn as nn

from transformer.multi_head_attention import (
    MultiHeadSelfAttention
)

from transformer.feed_forward import (
    FeedForwardNetwork
)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1
    ):

        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim
        )

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask=None
    ):

        # Multi-head attention

        attention_output, _ = self.attention(
            x,
            mask
        )

        # Residual connection + norm

        x = self.layer_norm_1(
            x + self.dropout(attention_output)
        )

        # Feed forward

        ff_output = self.feed_forward(
            x
        )

        # Residual connection + norm

        output = self.layer_norm_2(
            x + self.dropout(ff_output)
        )

        return output