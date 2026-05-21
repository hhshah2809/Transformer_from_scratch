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
