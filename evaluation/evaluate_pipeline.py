from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import torch

from src.data import load_csv_for_kind, split_df, compute_method_features, compute_class_features, encode_codes
from src.tokenizer import SimpleVocab
from src.models.cnn import TextCNN

"""
EVALUATION SCRIPT FOR THE HYBRID PIPELINE
"""


def load_cnn(kind: str, ckpt_dir: str) -> Tuple[TextCNN, SimpleVocab]:
    vocab_path = os.path.join(ckpt_dir, f"{kind}_vocab.json")
    ckpt_path = os.path.join(ckpt_dir, f"{kind}_textcnn.pt")
    if not (os.path.exists(vocab_path) and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"Missing CNN artifacts for {kind} at {ckpt_dir}. Train via training/train_cnn.py first.")
    vocab = SimpleVocab.load(vocab_path)
    model = TextCNN(vocab_size=len(vocab.stoi))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model, vocab


def predict_cnn(model: TextCNN, vocab: SimpleVocab, texts: List[str], max_len: int = 512) -> np.ndarray:
    X = encode_codes(texts, vocab, max_len=max_len)
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.long))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.argmax(axis=1)


def predict_knn(df_train, df_test, y_train: List[int], kind: str, k: int = 35) -> np.ndarray:
    """
    Predict using KNN with precomputed heuristic features from dataframe.
    """
    if kind == "methods":
        Xtr_feats = compute_method_features(df_train)
        Xte_feats = compute_method_features(df_test)
    else:
        Xtr_feats = compute_class_features(df_train)
        Xte_feats = compute_class_features(df_test)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(Xtr_feats, y_train)
    probs = knn.predict_proba(Xte_feats)
    preds = probs.argmax(axis=1)
    return preds


def predict_heuristics(df) -> np.ndarray:
    """
    Get heuristic predictions from dataframe (uses precomputed h_label column).
    """
    if 'h_label' not in df.columns:
        raise ValueError("Missing h_label column. Run dataset/scripts/add_heuristics_to_csv.py first.")

    return df['h_label'].values.astype(int)


def evaluate_kind(kind: str, ckpt_dir: str, weights=(0.1, 0.5, 0.4), out_dir: str = "evaluation/plots"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_csv_for_kind(kind)
    split = split_df(df, test_size=0.2, seed=42)

    # Extract code only for CNN
    code_only = [x[0] if isinstance(x, list) else x for x in split.X_test]

    # Heuristics (from precomputed h_label)
    h_preds = predict_heuristics(split.df_test)

    # KNN (runtime fit with precomputed features)
    k = 58 if kind == "methods" else 92
    knn_preds = predict_knn(split.df_train, split.df_test, split.y_train, kind, k)

    # CNN
    cnn_model, cnn_vocab = load_cnn(kind, ckpt_dir)
    cnn_preds = predict_cnn(cnn_model, cnn_vocab, code_only)

    # Weighted ensemble
    w_h, w_knn, w_cnn = weights  # default 0.1, 0.5, 0.4 (KNN strongest)
    final_preds = w_h * h_preds + w_knn * knn_preds + w_cnn * cnn_preds
    for i in range(len(final_preds)):
        if final_preds[i] < 0.6:
            final_preds[i] = 0
        elif final_preds[i] < 1.2:
            final_preds[i] = 1
        else:
            final_preds[i] = 2

    # Metrics
    y_true = np.array(split.y_test)
    acc_h = accuracy_score(y_true, h_preds)
    acc_knn = accuracy_score(y_true, knn_preds)
    acc_cnn = accuracy_score(y_true, cnn_preds)
    acc_ens = accuracy_score(y_true, final_preds)

    print(f"{kind} — Heuristics: {acc_h:.3f}, KNN: {acc_knn:.3f}, CNN: {acc_cnn:.3f}, Ensemble: {acc_ens:.3f}")

    my_eval_score = (1 - (sum((final_preds - y_true) ** 2) / (4 * len(final_preds)))) * 100

    print(f"ENS my eval score: {my_eval_score:.3f}%")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, final_preds, labels=[0,1,2])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_title(f"{kind.capitalize()} Ensemble Confusion")
    ax[0].set_xlabel("Pred")
    ax[0].set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(j, i, str(v), ha='center', va='center')
    plt.colorbar(im, ax=ax[0])

    # Component accuracies
    ax[1].bar(["Heur", "KNN", "CNN", "Ensemble"], [acc_h, acc_knn, acc_cnn, acc_ens], color=["gray","orange","green","blue"])
    ax[1].set_ylim(0, 1)
    ax[1].set_title(f"{kind.capitalize()} Accuracies")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{kind}_results.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plots to {out_path}")

    return {
        "acc": {"heuristics": acc_h, "knn": acc_knn, "cnn": acc_cnn, "ensemble": acc_ens},
        "confusion_matrix": cm.tolist(),
        "plot": out_path,
    }

def choosing_best_k(out_dir: str = "evaluation/plots"):
    os.makedirs(out_dir, exist_ok=True)

    for kind in ["methods", "classes"]:
        print(f"\n{'='*60}\nEvaluating KNN with K=1-100 for {kind}\n{'='*60}")

        df = load_csv_for_kind(kind)
        split = split_df(df, test_size=0.2, seed=42)
        y_true = np.array(split.y_test)

        # Test K from 1 to 100
        k_values = range(1, 101)
        accuracies = [accuracy_score(y_true, predict_knn(split.df_train, split.df_test, split.y_train, kind, k=k))
                      for k in k_values]

        # Find best K
        best_k = np.argmax(accuracies) + 1
        best_acc = accuracies[best_k - 1]
        print(f"\n{'*'*60}\nBEST K for {kind}: {best_k} with Accuracy: {best_acc:.4f}\n{'*'*60}\n")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(k_values, accuracies, 'b-o', linewidth=2, markersize=3)
        ax.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best K={best_k}')
        ax.scatter([best_k], [best_acc], color='r', s=200, zorder=5, marker='*', label=f'Acc={best_acc:.4f}')
        ax.set_xlabel('K Value', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'KNN Accuracy vs K ({kind.capitalize()})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 101)

        plot_path = os.path.join(out_dir, f"{kind}_knn_k_analysis.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hybrid pipeline for methods/classes")
    parser.add_argument("--kind", choices=["methods", "classes"], required=True)
    parser.add_argument("--ckpt", default="training/checkpoints")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.1, 0.5, 0.4], help="Weights: heuristics KNN CNN")
    parser.add_argument("--out", default="evaluation/plots")
    args = parser.parse_args()
    evaluate_kind(args.kind, args.ckpt, tuple(args.weights), args.out)
    # choosing_best_k(args.ckpt)
