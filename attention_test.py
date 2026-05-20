import torch
from embeddings.embedding_config import EmbeddingConfig
from embeddings.embedding_pipeline import EmbeddingPipeline
from transformer.attention import SelfAttention
from transformer.causal_mask import generate_causal_mask
from transformer.attention_utils import print_attention_info


def main():
    config = EmbeddingConfig()
    embedding_pipeline = EmbeddingPipeline(
        config
    )
    attention = SelfAttention(
        embedding_dim=config.embedding_dim
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

    print_attention_info(
        output,
        weights
    )


if __name__ == "__main__":
    main()