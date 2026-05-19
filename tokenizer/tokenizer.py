from typing import List

class Tokenizer:

    def __init__(
        self,
        config,
        vocabulary,
        preprocessor
    ):

        self.config = config
        self.vocab = vocabulary
        self.preprocessor = preprocessor

    def tokenize(self, text: str) -> List[str]:

        text = self.preprocessor.clean(text)

        return text.split()

    def encode(
        self,
        text: str,
        add_special_tokens=True,
        padding=True,
        truncation=True
    ) -> List[int]:

        tokens = self.tokenize(text)

        ids = [
            self.vocab.token_to_id(token)
            for token in tokens
        ]

        if add_special_tokens:

            ids = (
                [self.vocab.token_to_id(self.config.sos_token)]
                + ids
                + [self.vocab.token_to_id(self.config.eos_token)]
            )

        if truncation:
            ids = ids[: self.config.max_sequence_length]

        if padding:
            ids = self.pad(ids)

        return ids

    def decode(self, ids: List[int]) -> str:

        tokens = [
            self.vocab.id_to_token(idx)
            for idx in ids
        ]

        filtered_tokens = [
            token
            for token in tokens
            if token not in {
                self.config.pad_token,
                self.config.sos_token,
                self.config.eos_token
            }
        ]

        return " ".join(filtered_tokens)

    def pad(self, ids: List[int]):

        pad_id = self.vocab.token_to_id(
            self.config.pad_token
        )

        padding_length = (
            self.config.max_sequence_length
            - len(ids)
        )

        if padding_length > 0:
            ids += [pad_id] * padding_length

        return ids

    def batch_encode(self, texts):

        return [
            self.encode(text)
            for text in texts
        ]