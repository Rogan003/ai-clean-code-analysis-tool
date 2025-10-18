from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from javalang.tree import MethodDeclaration, ClassDeclaration
from sklearn.model_selection import train_test_split

from src.tokenizer import java_code_tokenize, SimpleVocab

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
    df_train: pd.DataFrame = None  # DataFrame with heuristic features for train set
    df_test: pd.DataFrame = None   # DataFrame with heuristic features for test set


def load_csv_for_kind(kind: str) -> pd.DataFrame:
    assert kind in {"methods", "classes"}
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
    """
    Split dataframe into train/test sets.
    Returns code snippets along with their metadata and the split dataframes with heuristics.
    """
    score_col = 'score'
    y_raw = df[score_col].tolist()

    score_and_labels_map: Dict = {"good": 0, "Green": 0, "changes_recommended": 1, "Yellow": 1, "changes_required": 2, "Red": 2}
    y: List[int] = []
    for v in y_raw:
        if isinstance(v, str) and v in score_and_labels_map:
            y.append(score_and_labels_map[v])

    # Split the dataframe
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Extract X data (code snippets + metadata)
    if 'average_method_score' in df.columns:
        # Classes dataset
        X_train = df_train[['code_snippet', 'average_method_score']].values.tolist()
        X_test = df_test[['code_snippet', 'average_method_score']].values.tolist()
    else:
        # Methods dataset
        X_train = df_train[['code_snippet']].values.tolist()
        X_test = df_test[['code_snippet']].values.tolist()

    return Split(X_train, y_train, X_test, y_test, df_train, df_test)


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
def compute_method_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract method heuristic features from DataFrame.
    Expected columns: h_name_len, h_name_special, h_name_camel, h_variable_len,
                     h_variable_special, h_variable_camel, h_method_length, h_n_params,
                     h_comment_chars, h_indent, h_long_line, h_cyclomatic, h_returns
    """
    feature_cols = [
        'h_name_len', 'h_name_special', 'h_name_camel', 'h_variable_len',
        'h_variable_special', 'h_variable_camel', 'h_method_length', 'h_n_params',
        'h_comment_chars', 'h_indent', 'h_long_line', 'h_cyclomatic', 'h_returns'
    ]

    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing heuristic columns: {missing_cols}. Run dataset/scripts/add_heuristics_to_csv.py first.")

    return df[feature_cols].values.astype(float)


def compute_class_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract class heuristic features from DataFrame.
    Expected columns: h_name_len, h_prop_name_len_avg, h_prop_name_special_avg,
                     h_public_non_gs, h_n_vars, h_comment_chars, h_avg_method_score,
                     h_lack_of_cohesion, h_name_camel
    """
    feature_cols = [
        'h_name_len', 'h_prop_name_len_avg', 'h_prop_name_special_avg',
        'h_public_non_gs', 'h_n_vars', 'h_comment_chars', 'h_avg_method_score',
        'h_lack_of_cohesion', 'h_name_camel'
    ]

    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing heuristic columns: {missing_cols}. Run dataset/scripts/add_heuristics_to_csv.py first.")

    return df[feature_cols].values.astype(float)


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