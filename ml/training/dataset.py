import numpy as np
import torch
from torch.utils.data import Dataset
from ml.config.model_config import MAX_SEQUENCE_LENGTH


class GPTDataset(Dataset):
    """Dataset that returns autoregressive input-target pairs from a flattened token id array."""

    def __init__(self, token_ids: np.ndarray, seq_len: int = None):
        if seq_len is None:
            seq_len = MAX_SEQUENCE_LENGTH

        self.seq_len = seq_len

        if isinstance(token_ids, str):
            token_ids = np.load(token_ids)

        self.tokens = token_ids

        if len(self.tokens) < 2:
            raise ValueError("token_ids must contain at least two tokens")

    def __len__(self):
        # number of possible windows of size seq_len+1
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        start = idx
        end = start + self.seq_len + 1
        window = self.tokens[start:end]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        target_ids = torch.tensor(window[1:], dtype=torch.long)
        return input_ids, target_ids
