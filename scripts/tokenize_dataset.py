import os
import numpy as np
from tqdm import tqdm
from ml.tokenizer import SentencePieceTokenizer
from ml.config.training_config import DATA_DIR


def main():
    tokenizer_model = os.path.join(DATA_DIR, "tokenizer.model")
    txt_path = os.path.join(DATA_DIR, "tinystories.txt")
    out_path = os.path.join(DATA_DIR, "token_ids.npy")

    if not os.path.exists(tokenizer_model):
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_model}")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text dataset not found: {txt_path}")

    tokenizer = SentencePieceTokenizer(tokenizer_model)

    all_ids = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            if len(ids) == 0:
                continue
            all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.int64)
    np.save(out_path, arr)

    print(f"Saved {arr.shape[0]} token ids to {out_path}")


if __name__ == "__main__":
    main()
import os
import numpy as np
import sentencepiece as spm


def main(
    model_path: str = "data/tokenizer.model",
    input_path: str = "data/tinystories.txt",
    out_path: str = "data/token_ids.npy",
):

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    all_ids = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.EncodeAsIds(line)
            all_ids.extend(ids + [sp.eos_id()])

    arr = np.array(all_ids, dtype=np.int32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)

    print("Saved token ids to", out_path)


if __name__ == "__main__":
    main()
