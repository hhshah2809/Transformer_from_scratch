import pickle

class TokenizerSerializer:

    @staticmethod
    def save(tokenizer, path):

        with open(path, "wb") as f:
            pickle.dump(tokenizer, f)

    @staticmethod
    def load(path):

        with open(path, "rb") as f:
            return pickle.load(f)
