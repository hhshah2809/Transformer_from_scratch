import requests

class DatasetLoader:

    @staticmethod
    def load_tiny_shakespeare():

        url = (
            "https://raw.githubusercontent.com/"
            "karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt"
        )

        return requests.get(url).text