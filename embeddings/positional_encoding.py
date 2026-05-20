import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embedding_dim,
        max_len=5000
    ):

        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2
            ) * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]