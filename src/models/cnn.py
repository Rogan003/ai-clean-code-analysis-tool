from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
CNN MODEL TO RECOGNIZE CODE TEXT PATTERNS AND CLASSIFY CODE INTO ONE OF 3 CLASSES
"""

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_classes: int = 3, kernel_sizes=(3, 5, 7, 9), num_filters: int = 64, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes) * 2, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        t = emb.transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = self.act(conv(t))
            maxp = torch.amax(h, dim=-1)
            avgp = torch.mean(h, dim=-1)
            feats.append(torch.cat([maxp, avgp], dim=1))
        z = torch.cat(feats, dim=1)
        z = self.dropout(z)
        return self.fc(z)


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 64
    lr: float = 1e-3
    max_len: int = 512


def train_textcnn(model: TextCNN, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig, device: str = None) -> Tuple[TextCNN, List[float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += int((pred == y).sum())

        # val
        model.eval()
        v_total = 0
        v_correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                pred = torch.argmax(logits, dim=1)
                v_total += y.size(0)
                v_correct += int((pred == y).sum())
        val_acc = v_correct / max(1, v_total)
        history.append(val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
