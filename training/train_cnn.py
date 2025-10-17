from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from src.data import load_csv_for_kind, split_df, build_vocab_from_codes, encode_codes
from src.models.cnn import TextCNN, ArrayDataset, TrainConfig, train_textcnn


def train_one(kind: str, out_dir: str = "training/checkpoints", max_len: int = 512):
    os.makedirs(out_dir, exist_ok=True)
    df = load_csv_for_kind(kind)
    split = split_df(df, test_size=0.2, seed=42)

    code_only = [x[0] if isinstance(x, list) else x for x in split.X_train]
    # further split train into train/val for early selection
    X_tr, X_val, y_tr, y_val = train_test_split(code_only, split.y_train, test_size=0.1, random_state=42, stratify=split.y_train)

    vocab = build_vocab_from_codes(X_tr)
    np.save(os.path.join(out_dir, f"{kind}_vocab_size.npy"), np.array([len(vocab.stoi)]))
    vocab.save(os.path.join(out_dir, f"{kind}_vocab.json"))

    Xtr_ids = encode_codes(X_tr, vocab, max_len=max_len)
    Xval_ids = encode_codes(X_val, vocab, max_len=max_len)

    ytr = np.array(y_tr, dtype=np.int64)
    yval = np.array(y_val, dtype=np.int64)

    train_ds = ArrayDataset(Xtr_ids, ytr)
    val_ds = ArrayDataset(Xval_ids, yval)

    model = TextCNN(vocab_size=len(vocab.stoi))
    cfg = TrainConfig(epochs=8, batch_size=64, lr=1e-3, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model, hist = train_textcnn(model, train_loader, val_loader, cfg)

    torch.save(model.state_dict(), os.path.join(out_dir, f"{kind}_textcnn.pt"))
    print(f"Saved {kind} model to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TextCNN for methods/classes")
    parser.add_argument("--kind", choices=["methods", "classes"], required=True)
    parser.add_argument("--out", default="training/checkpoints")
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()
    train_one(args.kind, args.out, args.max_len)
