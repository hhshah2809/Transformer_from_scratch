"""Download TinyStories dataset subset and save to data/tinystories.txt"""
import os
from datasets import load_dataset
from ml.config.training_config import TINYSTORIES_SUBSET_SIZE, DATA_DIR


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Try known dataset ids; fall back to 'tiny_shakespeare' if not found
    candidates = [
        "tiny_stories",
        "tinystories",
        "tiny-stories",
        "tiny_shakespeare",
    ]

    dataset = None

    for name in candidates:
        try:
            dataset = load_dataset(name, split="train")
            print(f"Loaded dataset: {name}")
            break
        except Exception:
            continue

    if dataset is None:
        raise RuntimeError("Could not find TinyStories dataset automatically. Please provide dataset manually.")

    texts = []
    for i, example in enumerate(dataset):
        if i >= TINYSTORIES_SUBSET_SIZE:
            break
        # Try several field names for text
        if isinstance(example, dict):
            for key in ("text", "story", "content", "sentence"):
                if key in example:
                    texts.append(example[key].replace("\n", " "))
                    break
        else:
            texts.append(str(example))

    out_path = os.path.join(DATA_DIR, "tinystories.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.strip() + "\n")

    print(f"Saved {len(texts)} examples to {out_path}")


if __name__ == "__main__":
    main()
from datasets import load_dataset


def main():

    dataset = load_dataset(
        "roneneldan/TinyStories"
    )

    train_dataset = dataset["train"]

    with open(
        "data/tinystories.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for item in train_dataset:

            text = item["text"]

            file.write(text + "\n")

    print("TinyStories downloaded.")


if __name__ == "__main__":
    main()