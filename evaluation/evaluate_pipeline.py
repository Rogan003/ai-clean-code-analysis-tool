from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import torch

from src.data import load_csv_for_kind, split_df, compute_method_features, compute_class_features, build_vocab_from_texts, encode_texts, get_method_object
from src.heuristics import method_heuristics, class_heuristics
from src.tokenizer import SimpleVocab, java_code_tokenize
from src.models.cnn import TextCNN


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


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
    X = encode_texts(texts, vocab, max_len=max_len)
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.long))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def predict_knn(kind: str, X_train_src: List[str], y_train: List[int], X_test_src: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if kind == "methods":
        Xtr_feats, feat_names, _ = compute_method_features(X_train_src)
        Xte_feats, _, _ = compute_method_features(X_test_src)
    else:
        Xtr_feats, feat_names, _ = compute_class_features(X_train_src)
        Xte_feats, _, _ = compute_class_features(X_test_src)

    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(Xtr_feats, y_train)
    proba = knn.predict_proba(Xte_feats)
    preds = proba.argmax(axis=1)
    return proba, preds


def predict_heuristics(kind: str, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    preds = []
    if kind == "methods":
        for src in texts:
            method_obj = get_method_object(src)
            h = method_heuristics(src, method_obj)
            probs.append(h.proba)
            preds.append(h.label)
    else:
        for src in texts:
            cname = "MyClass"
            h = class_heuristics(src, cname)
            probs.append(h.proba)
            preds.append(h.label)
    return np.array(probs), np.array(preds)


def extract_methods_from_class(src: str) -> List[str]:
    # lightweight extraction; may not be perfect
    bodies = []
    for m in src.split("{"):
        if ")" in m and "}" in m:
            body = m.split(")")[-1]
            if "}" in body:
                body = body.split("}")[0]
                bodies.append("{" + body + "}")
    return [b for b in bodies if len(b) > 2]


def average_method_probs_over_class(method_model: TextCNN, method_vocab: SimpleVocab, class_srcs: List[str]) -> List[np.ndarray]:
    results = []
    for cls in class_srcs:
        methods = extract_methods_from_class(cls)
        if not methods:
            results.append(np.array([1/3, 1/3, 1/3]))
            continue
        probs = predict_cnn(method_model, method_vocab, methods)
        results.append(probs.mean(axis=0))
    return results


def evaluate_kind(kind: str, ckpt_dir: str, weights=(0.1, 0.5, 0.4), out_dir: str = "evaluation/plots"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_csv_for_kind(kind)
    split = split_df(df, test_size=0.2, seed=42)

    # Heuristics
    h_probs, h_preds = predict_heuristics(kind, split.X_test)

    # KNN (runtime fit)
    knn_probs, knn_preds = predict_knn(kind, split.X_train, split.y_train, split.X_test)

    # CNN
    cnn_model, cnn_vocab = load_cnn(kind, ckpt_dir)
    cnn_probs = predict_cnn(cnn_model, cnn_vocab, split.X_test)
    cnn_preds = cnn_probs.argmax(axis=1)

    # If evaluating classes, optionally mix in average method probabilities into CNN probs for classes
    if kind == "classes":
        method_model, method_vocab = load_cnn("methods", ckpt_dir)
        method_avg_probs = np.stack(average_method_probs_over_class(method_model, method_vocab, split.X_test))
        cnn_probs = (cnn_probs + method_avg_probs) / 2.0
        cnn_preds = cnn_probs.argmax(axis=1)

    # Weighted ensemble
    w_h, w_knn, w_cnn = weights  # default 0.1, 0.5, 0.4 per README (KNN strongest)
    final_probs = w_h * h_probs + w_knn * knn_probs + w_cnn * cnn_probs
    final_preds = final_probs.argmax(axis=1)

    # Metrics
    y_true = np.array(split.y_test)
    acc_h = accuracy_score(y_true, h_preds)
    acc_knn = accuracy_score(y_true, knn_preds)
    acc_cnn = accuracy_score(y_true, cnn_preds)
    acc_ens = accuracy_score(y_true, final_preds)

    print(f"{kind} — Heuristics: {acc_h:.3f}, KNN: {acc_knn:.3f}, CNN: {acc_cnn:.3f}, Ensemble: {acc_ens:.3f}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hybrid pipeline for methods/classes")
    parser.add_argument("--kind", choices=["methods", "classes"], required=True)
    parser.add_argument("--ckpt", default="training/checkpoints")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.1, 0.5, 0.4], help="Weights: heuristics KNN CNN")
    parser.add_argument("--out", default="evaluation/plots")
    args = parser.parse_args()
    evaluate_kind(args.kind, args.ckpt, tuple(args.weights), args.out)
