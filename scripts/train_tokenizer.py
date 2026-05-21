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
