import os
import sentencepiece as spm
from ml.config.model_config import VOCAB_SIZE
from ml.config.training_config import DATA_DIR


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    input_file = os.path.join(DATA_DIR, "tinystories.txt")
    model_prefix = os.path.join(DATA_DIR, "tokenizer")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )

    print(f"Trained tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    main()
import os
import sentencepiece as spm


def main(
    input_path: str = "data/tinystories.txt",
    model_prefix: str = "data/tokenizer",
    vocab_size: int = 8000,
    model_type: str = "bpe",
):

    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    args = (
        f"--input={input_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        "--character_coverage=1.0 "
        "--unk_id=0 "
    )

    print("Training SentencePiece tokenizer with:", args)

    spm.SentencePieceTrainer.Train(args)

    print("Tokenizer trained:", model_prefix + ".model")


if __name__ == "__main__":
    main()
