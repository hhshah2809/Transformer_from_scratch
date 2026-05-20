import torch


def generate_causal_mask(sequence_length):

    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length
        )
    )

    return mask