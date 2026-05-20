import torch

from embeddings.embedding_config import EmbeddingConfig
from embeddings.embedding_pipeline import EmbeddingPipeline
from embeddings.embedding_utils import print_tensor_info


def main():

    config = EmbeddingConfig()

    embedding_pipeline = EmbeddingPipeline(
        config
    )

    input_ids = torch.tensor([
        [1, 5, 8, 10, 22]
    ])

    embeddings = embedding_pipeline(
        input_ids
    )

    print("\nFINAL EMBEDDINGS:\n")

    print(embeddings)

    print("\nTENSOR INFO:\n")

    print_tensor_info(embeddings)


if __name__ == "__main__":
    main()