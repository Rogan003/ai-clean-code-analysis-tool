from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from javalang.tree import MethodDeclaration, ClassDeclaration
from sklearn.model_selection import train_test_split

from src.tokenizer import java_code_tokenize, SimpleVocab
from src.heuristics import method_heuristics, class_heuristics

"""
BASICALLY - LOADING THE DATA, SPLITTING AND EXTRACTING FEATURES FOR THE MODEL 
(some of which we could keep in the dataset honestly)
"""

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

    # drop rows with unparseable code snippets
    if kind == "methods":
        code_col = df.columns[0]
        def is_parseable(code):
            try:
                return get_method_object(str(code)) is not None
            except:
                return False
        valid_mask = df[code_col].apply(is_parseable)
        df = df[valid_mask].reset_index(drop=True)
    else:
        code_col = df.columns[0]

        def is_parseable(code):
            try:
                return get_class_obj(str(code)) is not None
            except:
                return False

        valid_mask = df[code_col].apply(is_parseable)
        df = df[valid_mask].reset_index(drop=True)

    return df


def split_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Split:
    code_col = df.columns[[0, 2]].tolist() if len(df.columns) > 2 else [df.columns[0]]
    score_col = df.columns[1]

    X = df[code_col].astype(str).values.tolist()
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


# --- Lightweight helpers for java code parsing ---
import javalang


def get_method_object(code: str) -> MethodDeclaration:
    try:
        # Wrap method in a class because this library won't work otherwise :)
        wrapped_code = f"class TempClass {{ {code} }}"
        tree = javalang.parse.parse(wrapped_code)
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            return node
    except Exception:
        # Return None if parsing fails
        return None


def get_class_obj(code: str) -> ClassDeclaration:
    try:
        tree = javalang.parse.parse(code)
        for _, node in tree.filter(javalang.tree.ClassDeclaration):
            return node
    except Exception:
        return None


# --- Feature/heuristic wrappers ---
# TODO: These 2 methods just calculate features for already existing code snippets. These methods won't need to exist like this once we save the method/class features in the dataset
def compute_method_features(codes: List[str]) -> np.ndarray:
    feats: List[List[float]] = []

    for src in codes:
        method_obj = get_method_object(str(src))
        if method_obj is None:
            continue
        h = method_heuristics(str(src), method_obj)
        feats.append(h.features)

    return np.array(feats, dtype=float)


def compute_class_features(codes: List[str], avg_method_scores: List[float]) -> np.ndarray:
    feats: List[List[float]] = []

    for src, score in zip(codes, avg_method_scores):
        class_obj = get_class_obj(src)
        h = class_heuristics(src, class_obj, score)
        feats.append(h.features)

    return np.array(feats, dtype=float)


# --- Tokenization helpers for CNN ---

def build_vocab_from_codes(codes: List[str], max_size: int = 30000, min_freq: int = 1) -> SimpleVocab:
    token_seqs = (java_code_tokenize(t) for t in codes)
    return SimpleVocab.build(token_seqs, max_size=max_size, min_freq=min_freq)


def encode_codes(codes: List[str], vocab: SimpleVocab, max_len: int = 512) -> np.ndarray:
    arr = np.stack([np.array(vocab.encode(java_code_tokenize(t), max_len=max_len), dtype=np.int64) for t in codes])
    return arr

if __name__ == "__main__":
    method_code = """
    public void something(int a, int b) {
        int c = a + b;
        System.out.println(c);
    }"""
    print(get_method_object(method_code))