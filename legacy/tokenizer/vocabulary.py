from collections import Counter

class Vocabulary:

    def __init__(self, config):

        self.config = config

        self.word_to_id = {}
        self.id_to_word = {}

    def build(self, tokens):

        counter = Counter(tokens)

        vocab = []

        for token, freq in counter.items():

            if freq >= self.config.min_frequency:
                vocab.append(token)

        vocab = sorted(vocab)

        vocab = vocab[: self.config.max_vocab_size]

        all_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.sos_token,
            self.config.eos_token,
        ] + vocab

        self.word_to_id = {
            token: idx
            for idx, token in enumerate(all_tokens)
        }

        self.id_to_word = {
            idx: token
            for token, idx in self.word_to_id.items()
        }

    def token_to_id(self, token):

        return self.word_to_id.get(
            token,
            self.word_to_id[self.config.unk_token]
        )

    def id_to_token(self, idx):

        return self.id_to_word.get(
            idx,
            self.config.unk_token
        )
