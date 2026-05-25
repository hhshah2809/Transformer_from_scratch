import os
import glob
import torch
from ml.config.model_config import *
from ml.config.training_config import CHECKPOINT_DIR, DATA_DIR, DEVICE
from ml.tokenizer import SentencePieceTokenizer
from ml.inference.generator import generate
from gpt.gpt_model import GPTModel


def find_latest_checkpoint(dir_path):
    files = glob.glob(os.path.join(dir_path, "*.pt"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def main():
    ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found. Train the model first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    tokenizer = SentencePieceTokenizer(os.path.join(DATA_DIR, "tokenizer.model"))

    prompt = input("Enter prompt: ")
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [tokenizer.bos_id() or 1]

    input_ids = torch.tensor([ids], dtype=torch.long)

    out = generate(model, input_ids, device=device)

    out_ids = out[0].cpu().tolist()
    text = tokenizer.decode(out_ids)
    print("\nGenerated:\n", text)


if __name__ == "__main__":
    main()
