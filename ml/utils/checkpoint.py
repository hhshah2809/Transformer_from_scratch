import os
import torch


def save_checkpoint(path, model, optimizer, scaler, epoch, train_loss=None, val_loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    state = torch.load(path, map_location=device)

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])

    return state
