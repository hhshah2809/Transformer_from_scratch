import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ml.config.model_config import *
from ml.config.training_config import *

from ml.training.dataset import GPTDataset
from ml.training.loss import get_loss_function
from ml.training.trainer import Trainer
from ml.utils.checkpoint import save_checkpoint

from gpt.gpt_model import GPTModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = os.path.join(DATA_DIR, "token_ids.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Token ids not found at {data_path}. Run scripts/tokenize_dataset.py first.")

    token_ids = np.load(data_path)

    dataset = GPTDataset(token_ids, seq_len=MAX_SEQUENCE_LENGTH)

    # split
    n = len(dataset)
    train_n = int(n * TRAIN_SPLIT)
    indices = list(range(n))

    train_dataset = Subset(dataset, indices[:train_n])
    val_dataset = Subset(dataset, indices[train_n:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    loss_fn = get_loss_function()

    trainer = Trainer(model, optimizer, loss_fn, device=device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, trainer.scaler, epoch, train_loss=train_loss, val_loss=val_loss)


if __name__ == "__main__":
    main()
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
