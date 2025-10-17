from __future__ import annotations

import json
from collections import Counter
from typing import List, Iterable, Dict

PAD = "<PAD>"
UNK = "<UNK>"


def java_code_tokenize(text: str) -> List[str]:
    """Tokenize Java code:
    - group adjacent alphanumerics/underscore into one token
    - group adjacent whitespace characters into one token
    - every other character is a single-char token
    """
    tokens: List[str] = []
    if not text:
        return tokens
    buf = []

    def flush():
        nonlocal buf
        if buf:
            tokens.append("".join(buf))
            buf = []

    cur_kind = None  # 'alnum' | 'space' | None
    for ch in text:
        if ch.isalnum() or ch == "_":
            kind = "alnum"
        elif ch.isspace():
            kind = "space"
        else:
            kind = "other"
        if kind == "other":
            flush()
            tokens.append(ch)
            cur_kind = None
        else:
            if cur_kind == kind:
                buf.append(ch)
            else:
                flush()
                buf.append(ch)
                cur_kind = kind

    flush()
    return tokens


class SimpleVocab:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = {PAD: 0, UNK: 1, **{k: v for k, v in stoi.items() if k not in {PAD, UNK}}}
        # rebuild to ensure contiguous indices starting at 0
        items = [PAD, UNK] + [k for k in stoi.keys() if k not in {PAD, UNK}]
        self.stoi = {tok: i for i, tok in enumerate(items)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @classmethod
    def build(cls, token_seqs: Iterable[List[str]], max_size: int = 30000, min_freq: int = 1) -> "SimpleVocab":
        counter = Counter()
        for seq in token_seqs:
            counter.update(seq)
        most_common = [(tok, c) for tok, c in counter.items() if c >= min_freq]
        most_common.sort(key=lambda x: (-x[1], x[0]))
        stoi = {}
        for tok, _ in most_common[: max_size - 2]:  # reserve for PAD/UNK
            stoi.setdefault(tok, len(stoi) + 2)
        return cls(stoi)

    def encode(self, tokens: List[str], max_len: int = 512) -> List[int]:
        ids = [self.stoi.get(t, 1) for t in tokens]  # 1 -> UNK
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [0] * (max_len - len(ids))  # 0 -> PAD

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleVocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["stoi"])
