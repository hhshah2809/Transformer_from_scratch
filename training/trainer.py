# training/trainer.py

import torch

from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        device
    ):

        self.model = model

        self.optimizer = optimizer

        self.loss_function = loss_function

        self.device = device

    def train_epoch(
        self,
        dataloader,
        causal_mask_function
    ):

        self.model.train()

        total_loss = 0

        progress_bar = tqdm(
            dataloader,
            desc="Training"
        )

        for inputs, targets in progress_bar:

            inputs = inputs.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            sequence_length = inputs.size(1)

            mask = causal_mask_function(
                sequence_length
            ).to(self.device)

            logits = self.model(
                inputs,
                mask
            )

            loss = self.loss_function(

                logits.view(
                    -1,
                    logits.size(-1)
                ),

                targets.view(-1)
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=loss.item()
            )

        average_loss = (
            total_loss / len(dataloader)
        )

        return average_loss
