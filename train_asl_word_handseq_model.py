"""
Sign-Bridge — Train Dynamic Hand-Only Word Model (Sequence)

Trains a classifier for dynamic hand-only words from:
- data/asl_words_handseq.npz (X: [N, seq_len, 63], y: [N])

It converts each sequence into summary features:
- mean (63), std (63), mean abs delta (63) => 189 dims

Outputs:
- models/asl_word_handseq_model.pkl

This is CPU-friendly and works well for YES/NO-style gestures.
"""

from __future__ import annotations

import os
import json
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class TrainConfig:
    """Runtime training config."""
    npz_path: str
    label_map_path: str
    out_dir: str
    test_size: float
    seed: int


def summarize_sequence(X_seq: np.ndarray) -> np.ndarray:
    """Sequence -> 189D summary."""
    X_seq = X_seq.astype(np.float32)
    mu = np.mean(X_seq, axis=0)
    sd = np.std(X_seq, axis=0)
    if X_seq.shape[0] >= 2:
        delta = np.abs(X_seq[1:, :] - X_seq[:-1, :])
        dmu = np.mean(delta, axis=0)
    else:
        dmu = np.zeros_like(mu)
    return np.concatenate([mu, sd, dmu], axis=0).astype(np.float32)


def load_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing dataset: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    X = data["X"].astype(np.float32)  # (N, seq_len, 63)
    y = data["y"].astype(np.int64)    # (N,)

    if X.ndim != 3 or X.shape[2] != 63:
        raise ValueError("Expected X shape (N, seq_len, 63).")

    X_sum = np.stack([summarize_sequence(X[i]) for i in range(X.shape[0])], axis=0)
    return X_sum, y


def load_label_names(label_map_path: str) -> Optional[list]:
    try:
        if not os.path.exists(label_map_path):
            return None
        with open(label_map_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if "label_to_idx" not in m:
            return None
        inv = {int(v): str(k) for k, v in m["label_to_idx"].items()}
        return [inv[i] for i in sorted(inv.keys())]
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join("data", "asl_words_handseq.npz"))
    ap.add_argument("--label_map", default=os.path.join("data", "asl_words_handseq_labels.json"))
    ap.add_argument("--out_dir", default=os.path.join("models"))
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = TrainConfig(
        npz_path=args.npz,
        label_map_path=args.label_map,
        out_dir=args.out_dir,
        test_size=float(args.test_size),
        seed=int(args.seed),
    )

    X, y = load_data(cfg.npz_path)
    label_names = load_label_names(cfg.label_map_path)

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 sequence samples to train the hand-sequence word model.")

    class_counts = Counter(y.tolist())
    print("[handseq-train] samples per class:", {int(k): int(v) for k, v in sorted(class_counts.items())})

    can_hold_out = X.shape[0] >= 5 and len(class_counts) >= 2 and min(class_counts.values()) >= 2
    if can_hold_out:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y,
        )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y
        print("[handseq-train] dataset is too small for a reliable stratified holdout; training and reporting on all samples.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("[handseq-train] report:")
    if label_names is not None:
        print(classification_report(y_val, y_pred, target_names=label_names))
    else:
        print(classification_report(y_val, y_pred))

    print("[handseq-train] confusion:")
    print(confusion_matrix(y_val, y_pred))

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, "asl_word_handseq_model.pkl")
    joblib.dump(model, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
