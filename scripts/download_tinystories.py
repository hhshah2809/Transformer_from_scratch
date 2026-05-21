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