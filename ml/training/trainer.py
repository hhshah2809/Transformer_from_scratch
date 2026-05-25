import os
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from ml.config.training_config import DEVICE, GRADIENT_CLIP_VALUE


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or DEVICE
        self.scaler = GradScaler()

        if self.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        iters = 0

        pbar = tqdm(dataloader, desc="train", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                logits = self.model(inputs)
                # logits: (B, S, V) -> reshape for loss
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            self.scaler.scale(loss).backward()

            # gradient clipping
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_VALUE)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            iters += 1
            pbar.set_postfix(loss=batch_loss)

        avg_loss = total_loss / max(1, iters)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        iters = 0

        pbar = tqdm(dataloader, desc="valid", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with autocast():
                logits = self.model(inputs)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            iters += 1

        avg_loss = total_loss / max(1, iters)
        return avg_loss
