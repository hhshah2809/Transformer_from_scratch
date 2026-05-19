import re

class TextPreprocessor:

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def clean(self, text: str) -> str:

        if self.lowercase:
            text = text.lower()

        text = re.sub(r"\s+", " ", text)

        return text.strip()