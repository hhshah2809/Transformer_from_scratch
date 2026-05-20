from dataclasses import dataclass


@dataclass
class EmbeddingConfig:

    vocab_size: int = 50000

    embedding_dim: int = 128

    max_sequence_length: int = 128

    dropout: float = 0.1