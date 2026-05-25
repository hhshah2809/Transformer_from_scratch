
import os
import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    Subset
)

from ml.config.model_config import *

from ml.config.training_config import *

from ml.training.dataset import GPTDataset

from ml.training.loss import (
    get_loss_function
)

from ml.training.trainer import (
    Trainer
)

from ml.utils.checkpoint import (
    save_checkpoint
)

from gpt.gpt_model import (
    GPTModel
)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    data_path = os.path.join(
        DATA_DIR,
        "token_ids.npy"
    )

    if not os.path.exists(data_path):

        raise FileNotFoundError(
            f"Token ids not found at {data_path}"
        )

    token_ids = np.load(data_path)

    dataset = GPTDataset(
        token_ids=token_ids,
        seq_len=MAX_SEQUENCE_LENGTH
    )

    n = len(dataset)

    train_n = int(
        n * TRAIN_SPLIT
    )

    indices = list(range(n))

    train_dataset = Subset(
        dataset,
        indices[:train_n]
    )

    val_dataset = Subset(
        dataset,
        indices[train_n:]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATASET,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
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
        loss_fn=loss_function,
        device=device
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = trainer.train_epoch(
            train_loader
        )

        val_loss = trainer.validate(
            val_loader
        )

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if SAVE_EVERY_EPOCH:

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"ckpt_epoch_{epoch}.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                trainer.scaler,
                epoch,
                train_loss,
                val_loss
            )


if __name__ == "__main__":
    main()

