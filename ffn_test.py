import torch

from embeddings.embedding_config import EmbeddingConfig
from embeddings.embedding_pipeline import EmbeddingPipeline

from transformer.feed_forward import (
    FeedForwardNetwork
)


def main():

    config = EmbeddingConfig()

    embedding_pipeline = EmbeddingPipeline(
        config
    )

    ffn = FeedForwardNetwork(
        embedding_dim=config.embedding_dim
    )

    input_ids = torch.tensor([
        [1, 5, 8, 10]
    ])

    embeddings = embedding_pipeline(
        input_ids
    )

    output = ffn(
        embeddings
    )

    print("\nFFN OUTPUT SHAPE:\n")

    print(output.shape)


if __name__ == "__main__":
    main()