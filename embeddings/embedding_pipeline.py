import torch.nn as nn

from embeddings.embedding import TokenEmbedding
from embeddings.positional_encoding import PositionalEncoding


class EmbeddingPipeline(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim
        )

        self.position_encoding = PositionalEncoding(
            embedding_dim=config.embedding_dim,
            max_len=config.max_sequence_length
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(self, token_ids):

        x = self.token_embedding(token_ids)

        x = self.position_encoding(x)

        x = self.dropout(x)

        return x