import torch

from embeddings.embedding_config import EmbeddingConfig
from embeddings.embedding_pipeline import EmbeddingPipeline

from transformer.multi_head_attention import (
    MultiHeadSelfAttention
)

from transformer.causal_mask import (
    generate_causal_mask
)


def main():

    config = EmbeddingConfig()

    embedding_pipeline = EmbeddingPipeline(
        config
    )

    attention = MultiHeadSelfAttention(
        embedding_dim=config.embedding_dim,
        num_heads=8
    )

    input_ids = torch.tensor([
        [1, 5, 8, 10]
    ])

    embeddings = embedding_pipeline(
        input_ids
    )

    sequence_length = input_ids.size(1)

    mask = generate_causal_mask(
        sequence_length
    )

    output, weights = attention(
        embeddings,
        mask
    )

    print("\nOUTPUT SHAPE:\n")

    print(output.shape)

    print("\nATTENTION WEIGHTS SHAPE:\n")

    print(weights.shape)


if __name__ == "__main__":
    main()