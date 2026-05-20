import torch
import torch.nn as nn

from transformer.transformer_block import (
    TransformerBlock
)


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_sequence_length,
        num_heads,
        num_layers,
        dropout=0.1
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.transformer_blocks = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )

                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        input_ids,
        mask=None
    ):

        batch_size, sequence_length = (
            input_ids.shape
        )

        positions = torch.arange(
            0,
            sequence_length,
            device=input_ids.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:

            x = transformer_block(
                x,
                mask
            )

        x = self.final_layer_norm(x)

        logits = self.output_projection(
            x
        )

        return logits