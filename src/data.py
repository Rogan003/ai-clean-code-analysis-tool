from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .tokenizer import java_code_tokenize, SimpleVocab
from .heuristics import method_heuristics, class_heuristics


@dataclass
class Split:
    X_train: List[str]
    y_train: List[int]
    X_test: List[str]
    y_test: List[int]


def load_csv_for_kind(kind: str) -> pd.DataFrame:
    assert kind in {"methods", "classes"}
    # primary expected locations
    expected = os.path.join("dataset", f"official_dataset_{kind}.csv")
    if os.path.exists(expected):
        df = pd.read_csv(expected)
    else:
        raise FileNotFoundError(f"Expected dataset at {expected} not found.")
    # drop rows with any nulls
    df = df.dropna(how="any").reset_index(drop=True)
    return df


def split_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Split:
    code_col = df.columns[0]
    score_col = df.columns[1]
    avg_method_score_col = df.columns[2] if len(df.columns) > 2 else None # TODO: Inject

    X = df[code_col].astype(str).tolist()
    y_raw = df[score_col].tolist()

    score_and_labels_map: Dict = {"good": 0, "Green": 0, "changes_recommended": 1, "Yellow": 1, "changes_required": 2, "Red": 2}
    y: List[int] = []
    for v in y_raw:
        if isinstance(v, str) and v in score_and_labels_map:
            y.append(score_and_labels_map[v])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )
    return Split(X_train, y_train, X_test, y_test)


# --- Lightweight helpers for signature extraction ---
import re


def guess_method_signature(code: str) -> Tuple[str, List[str]]:
    # naive: match 'returnType name(type p1, type p2) {'
    m = re.search(r"\b([a-zA-Z_][\w]*)\s*\(([^)]*)\)\s*\{", code)
    if not m:
        return "method", []
    name = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    # take only param names if present else types
    params = [pp.split()[-1] if " " in pp else pp for pp in params]
    return name, params


def guess_class_name(code: str) -> str:
    m = re.search(r"\bclass\s+([A-Z][A-Za-z0-9_]*)\b", code)
    return m.group(1) if m else "MyClass"


# --- Feature/heuristic wrappers ---

def compute_method_features(codes: List[str]) -> Tuple[np.ndarray, List[int], List[List[float]]]:
    feats: List[List[float]] = []
    labels_h: List[int] = []
    for src in codes:
        name, params = guess_method_signature(src)
        h = method_heuristics(src, name, params)
        feats.append(h.features)
        labels_h.append(h.label)

    return np.array(feats, dtype=float), labels_h, feats


def compute_class_features(codes: List[str]) -> Tuple[np.ndarray, List[int], List[List[float]]]:
    feats: List[List[float]] = []
    labels_h: List[int] = []
    for src in codes:
        cls = guess_class_name(src)
        h = class_heuristics(src, cls)
        feats.append(h.features)
        labels_h.append(h.label)

    return np.array(feats, dtype=float), labels_h, feats


# --- Tokenization helpers for CNN ---

def build_vocab_from_texts(texts: List[str], max_size: int = 30000, min_freq: int = 1) -> SimpleVocab:
    token_seqs = (java_code_tokenize(t) for t in texts)
    return SimpleVocab.build(token_seqs, max_size=max_size, min_freq=min_freq)


def encode_texts(texts: List[str], vocab: SimpleVocab, max_len: int = 512) -> np.ndarray:
    arr = np.stack([np.array(vocab.encode(java_code_tokenize(t), max_len=max_len), dtype=np.int64) for t in texts])
    return arr
