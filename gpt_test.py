import torch

from gpt.gpt_model import GPTModel

from transformer.causal_mask import (
    generate_causal_mask
)


def main():

    model = GPTModel(
        vocab_size=5000,
        embedding_dim=128,
        max_sequence_length=128,
        num_heads=8,
        num_layers=4
    )

    input_ids = torch.tensor([
        [1, 5, 8, 10]
    ])

    sequence_length = input_ids.size(1)

    mask = generate_causal_mask(
        sequence_length
    )

    logits = model(
        input_ids,
        mask
    )

    print("\nLOGITS SHAPE:\n")

    print(logits.shape)


if __name__ == "__main__":
    main()