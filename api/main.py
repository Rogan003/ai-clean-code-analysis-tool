from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import (
    load_csv_for_kind,
    split_df,
    compute_method_features,
    compute_class_features,
    encode_codes,
    get_method_object,
    get_class_obj
)
from src.tokenizer import SimpleVocab
from src.models.cnn import TextCNN
from src.heuristics import method_heuristics, class_heuristics
import pandas as pd

app = FastAPI(title="AI Clean Code Analysis API")

# Enable CORS for IntelliJ plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
MODELS = {
    "methods": {
        "cnn_model": None,
        "cnn_vocab": None,
        "knn_model": None,
        "train_data": None,
    },
    "classes": {
        "cnn_model": None,
        "cnn_vocab": None,
        "knn_model": None,
        "train_data": None,
    }
}

CKPT_DIR = "training/checkpoints"
WEIGHTS = (0.1, 0.3, 0.6)  # heuristics, KNN, CNN


class MethodRequest(BaseModel):
    code_snippet: str


class ClassRequest(BaseModel):
    code_snippet: str
    average_method_score: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int  # 0=green, 1=yellow, 2=red
    prediction_label: str  # "good", "changes_recommended", "changes_required"
    confidence: float


def load_model_for_kind(kind: str):
    """Load CNN and KNN models for the specified kind (methods/classes)."""
    if MODELS[kind]["cnn_model"] is not None:
        return  # Already loaded

    # Load CNN
    vocab_path = os.path.join(CKPT_DIR, f"{kind}_vocab.json")
    ckpt_path = os.path.join(CKPT_DIR, f"{kind}_textcnn.pt")

    if not (os.path.exists(vocab_path) and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"Missing CNN artifacts for {kind}. Train via training/train_cnn.py first.")

    vocab = SimpleVocab.load(vocab_path)
    model = TextCNN(vocab_size=len(vocab.stoi))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    MODELS[kind]["cnn_model"] = model
    MODELS[kind]["cnn_vocab"] = vocab

    # Load training data for KNN
    df = load_csv_for_kind(kind)
    split = split_df(df, test_size=0.2, seed=42)

    # Fit KNN
    k = 58 if kind == "methods" else 92
    if kind == "methods":
        X_train_feats = compute_method_features(split.df_train)
    else:
        X_train_feats = compute_class_features(split.df_train)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train_feats, split.y_train)

    MODELS[kind]["knn_model"] = knn
    MODELS[kind]["train_data"] = split.df_train


def predict_single_method(code_snippet: str) -> dict:
    """Predict quality for a single method."""
    kind = "methods"
    load_model_for_kind(kind)

    # Parse method
    method_obj = get_method_object(code_snippet)
    if method_obj is None:
        raise ValueError("Invalid method code snippet")

    # 1. Heuristic prediction
    h_result = method_heuristics(code_snippet, method_obj)
    h_pred = h_result.label

    # 2. KNN prediction
    df_temp = pd.DataFrame([{
        'h_name_len': h_result.features[0],
        'h_name_special': h_result.features[1],
        'h_name_camel': h_result.features[2],
        'h_variable_len': h_result.features[3],
        'h_variable_special': h_result.features[4],
        'h_variable_camel': h_result.features[5],
        'h_method_length': h_result.features[6],
        'h_n_params': h_result.features[7],
        'h_comment_chars': h_result.features[8],
        'h_indent': h_result.features[9],
        'h_long_line': h_result.features[10],
        'h_cyclomatic': h_result.features[11],
        'h_returns': h_result.features[12],
    }])
    X_feats = compute_method_features(df_temp)
    knn_probs = MODELS[kind]["knn_model"].predict_proba(X_feats)
    knn_pred = knn_probs.argmax(axis=1)[0]

    # 3. CNN prediction
    cnn_model = MODELS[kind]["cnn_model"]
    cnn_vocab = MODELS[kind]["cnn_vocab"]
    X = encode_codes([code_snippet], cnn_vocab, max_len=512)
    with torch.no_grad():
        logits = cnn_model(torch.as_tensor(X, dtype=torch.long))
        cnn_probs = torch.softmax(logits, dim=1).cpu().numpy()
    cnn_pred = cnn_probs.argmax(axis=1)[0]

    # 4. Ensemble
    w_h, w_knn, w_cnn = WEIGHTS
    final_pred = w_h * h_pred + w_knn * knn_pred + w_cnn * cnn_pred

    if final_pred < 0.4:
        final_label = 0
    elif final_pred < 1.0:
        final_label = 1
    else:
        final_label = 2

    # Calculate confidence (based on max probability from ensemble components)
    confidence = max(knn_probs[0][final_label], cnn_probs[0][final_label])

    return {
        "prediction": int(final_label),
        "confidence": float(confidence)
    }


def predict_single_class(code_snippet: str, avg_method_score: float = None) -> dict:
    """Predict quality for a single class."""
    kind = "classes"
    load_model_for_kind(kind)

    # Parse class
    class_obj = get_class_obj(code_snippet)
    if class_obj is None:
        raise ValueError("Invalid class code snippet")

    # Default average method score if not provided
    if avg_method_score is None:
        avg_method_score = 1.0  # Assume neutral

    # 1. Heuristic prediction
    h_result = class_heuristics(code_snippet, class_obj, avg_method_score)
    h_pred = h_result.label

    # 2. KNN prediction
    df_temp = pd.DataFrame([{
        'h_name_len': h_result.features[0],
        'h_prop_name_len_avg': h_result.features[1],
        'h_prop_name_special_avg': h_result.features[2],
        'h_public_non_gs': h_result.features[3],
        'h_n_vars': h_result.features[4],
        'h_comment_chars': h_result.features[5],
        'h_avg_method_score': h_result.features[6],
        'h_lack_of_cohesion': h_result.features[7],
        'h_name_camel': h_result.features[8],
    }])
    X_feats = compute_class_features(df_temp)
    knn_probs = MODELS[kind]["knn_model"].predict_proba(X_feats)
    knn_pred = knn_probs.argmax(axis=1)[0]

    # 3. CNN prediction
    cnn_model = MODELS[kind]["cnn_model"]
    cnn_vocab = MODELS[kind]["cnn_vocab"]
    X = encode_codes([code_snippet], cnn_vocab, max_len=512)
    with torch.no_grad():
        logits = cnn_model(torch.as_tensor(X, dtype=torch.long))
        cnn_probs = torch.softmax(logits, dim=1).cpu().numpy()
    cnn_pred = cnn_probs.argmax(axis=1)[0]

    # 4. Ensemble
    w_h, w_knn, w_cnn = WEIGHTS
    final_pred = w_h * h_pred + w_knn * knn_pred + w_cnn * cnn_pred

    if final_pred < 0.6:
        final_label = 0
    elif final_pred < 1.2:
        final_label = 1
    else:
        final_label = 2

    # Calculate confidence
    confidence = max(knn_probs[0][final_label], cnn_probs[0][final_label])

    return {
        "prediction": int(final_label),
        "confidence": float(confidence)
    }


@app.get("/")
def root():
    return {"message": "AI Clean Code Analysis API", "status": "running"}


@app.post("/predict/method", response_model=PredictionResponse)
def predict_method(request: MethodRequest):
    """Predict quality for a method."""
    try:
        result = predict_single_method(request.code_snippet)
        label_map = {0: "good", 1: "changes_recommended", 2: "changes_required"}
        return PredictionResponse(
            prediction=result["prediction"],
            prediction_label=label_map[result["prediction"]],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/class", response_model=PredictionResponse)
def predict_class(request: ClassRequest):
    """Predict quality for a class."""
    try:
        result = predict_single_class(request.code_snippet, request.average_method_score)
        label_map = {0: "good", 1: "changes_recommended", 2: "changes_required"}
        return PredictionResponse(
            prediction=result["prediction"],
            prediction_label=label_map[result["prediction"]],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
