import os

from datasets import load_dataset

from ml.config.training_config import (
    TINYSTORIES_SUBSET_SIZE,
    DATA_DIR
)


def main():

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    print("Loading TinyStories dataset...")

    dataset = load_dataset(
        "roneneldan/TinyStories"
    )

    train_dataset = dataset["train"]

    print(
        f"Using subset size: {TINYSTORIES_SUBSET_SIZE}"
    )

    subset = train_dataset.select(
        range(TINYSTORIES_SUBSET_SIZE)
    )

    output_path = (
        f"{DATA_DIR}/tinystories.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in subset:

            text = item["text"]

            file.write(
                text + "\n"
            )

    print(
        f"Saved dataset to: {output_path}"
    )


if __name__ == "__main__":
    main()