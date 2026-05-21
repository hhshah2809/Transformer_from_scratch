# root/train.py
import os

import torch

import numpy as np

from torch.utils.data import DataLoader

from training.dataset import GPTDataset

from training.loss import (
    get_loss_function
)

from training.trainer import (
    Trainer
)

from gpt.gpt_model import GPTModel

from transformer.causal_mask import (
    generate_causal_mask
)

from model_config import *

from training_config import *


def main():

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    device = torch.device(

        DEVICE

        if torch.cuda.is_available()

        else "cpu"
    )

    print(f"\nUsing device: {device}")

    token_ids = np.load(
        "data/token_ids.npy"
    ).tolist()

    dataset = GPTDataset(

        token_ids=token_ids,

        sequence_length=MAX_SEQUENCE_LENGTH
    )

    dataloader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    model = GPTModel(

        vocab_size=VOCAB_SIZE,

        embedding_dim=EMBEDDING_DIM,

        max_sequence_length=MAX_SEQUENCE_LENGTH,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        dropout=DROPOUT

    ).to(device)

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE
    )

    loss_function = get_loss_function()

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        loss_function=loss_function,

        device=device
    )

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}")

        average_loss = trainer.train_epoch(

            dataloader,

            generate_causal_mask
        )

        print(
            f"Average Loss: {average_loss:.4f}"
        )

        checkpoint_path = (

            f"checkpoints/model_epoch_{epoch+1}.pt"
        )

        torch.save(

            model.state_dict(),

            checkpoint_path
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )


if __name__ == "__main__":

    main()
