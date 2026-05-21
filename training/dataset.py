# training/dataset.py
import torch

from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(
        self,
        token_ids,
        sequence_length
    ):

        self.token_ids = token_ids

        self.sequence_length = sequence_length

    def __len__(self):

        return len(
            self.token_ids
        ) - self.sequence_length

    def __getitem__(
        self,
        index
    ):

        input_sequence = self.token_ids[
            index : index + self.sequence_length
        ]

        target_sequence = self.token_ids[
            index + 1 : index + self.sequence_length + 1
        ]

        input_tensor = torch.tensor(
            input_sequence,
            dtype=torch.long
        )

        target_tensor = torch.tensor(
            target_sequence,
            dtype=torch.long
        )

        return (
            input_tensor,
            target_tensor
        )
