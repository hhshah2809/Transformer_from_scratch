from tokenizer.config import TokenizerConfig
from tokenizer.vocabulary import Vocabulary
from tokenizer.tokenizer import Tokenizer
from tokenizer.preprocessor import TextPreprocessor
from tokenizer.dataset_loader import DatasetLoader


def main():

    config = TokenizerConfig()

    text = DatasetLoader.load_tiny_shakespeare()

    preprocessor = TextPreprocessor()

    tokens = preprocessor.clean(text).split()

    vocab = Vocabulary(config)

    vocab.build(tokens)

    tokenizer = Tokenizer(
        config=config,
        vocabulary=vocab,
        preprocessor=preprocessor
    )

    encoded = tokenizer.encode(
        "Transformers are powerful"
    )

    print("Encoded:")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded:")
    print(decoded)


if __name__ == "__main__":
    main()