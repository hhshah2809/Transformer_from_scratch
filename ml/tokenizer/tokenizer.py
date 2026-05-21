import os
from typing import List, Optional

import sentencepiece as spm


class SentencePieceTokenizer:
    """Simple wrapper around SentencePiece for BPE tokenization.

    Exposes: encode, decode, vocab_size, bos_id, eos_id, pad_id
    Loads model from `data/tokenizer.model` by default.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join("data", "tokenizer.model")

        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model_path)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id()] + ids
        if add_eos:
            ids = ids + [self.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.DecodeIds(ids)

    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def bos_id(self) -> int:
        try:
            return self.sp.bos_id()
        except Exception:
            return -1

    def eos_id(self) -> int:
        try:
            return self.sp.eos_id()
        except Exception:
            return -1

    def pad_id(self) -> int:
        try:
            return self.sp.pad_id()
        except Exception:
            return -1

    # convenience
    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False):
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
