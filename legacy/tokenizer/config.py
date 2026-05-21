from dataclasses import dataclass

@dataclass
class TokenizerConfig:

    lowercase: bool = True

    max_vocab_size: int = 50000

    min_frequency: int = 2

    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"

    max_sequence_length: int = 128
