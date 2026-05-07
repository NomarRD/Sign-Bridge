from __future__ import annotations

import os
import json
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    npz_path: str
    label_map_path: str
    out_dir: str
    test_size: float
    seed: int
    augment_factor: int = 2
    use_cv: bool = True


# ============================================================
# Features and augmentation
# ============================================================


def summarize_sequence(X_seq: np.ndarray) -> np.ndarray:
    """Sequence -> richer summary feature vector.

    Features per dimension:
    - mean
    - std
    - min
    - max
    - range
    - mean absolute delta
    - final frame
    Total: 63 * 7 = 441 dims
    """
    X_seq = X_seq.astype(np.float32)
    mu = np.mean(X_seq, axis=0)
    sd = np.std(X_seq, axis=0)
    mn = np.min(X_seq, axis=0)
    mx = np.max(X_seq, axis=0)
    rg = mx - mn
    if X_seq.shape[0] >= 2:
        delta = np.abs(X_seq[1:, :] - X_seq[:-1, :])
        dmu = np.mean(delta, axis=0)
    else:
        dmu = np.zeros_like(mu)
    last = X_seq[-1]
    return np.concatenate([mu, sd, mn, mx, rg, dmu, last], axis=0).astype(np.float32)


def jitter_sequence(X_seq: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0.0, noise_std, size=X_seq.shape).astype(np.float32)
    return (X_seq.astype(np.float32) + noise).astype(np.float32)


def time_warp_sequence(X_seq: np.ndarray) -> np.ndarray:
    """Lightweight temporal resampling that preserves the original length."""
    seq = X_seq.astype(np.float32)
    T = seq.shape[0]
    if T < 3:
        return seq.copy()

    stretch = np.random.uniform(0.88, 1.12)
    target_positions = np.linspace(0, T - 1, T, dtype=np.float32)
    source_positions = np.clip(target_positions / stretch, 0.0, T - 1)

    out = np.zeros_like(seq)
    left_idx = np.floor(source_positions).astype(int)
    right_idx = np.clip(left_idx + 1, 0, T - 1)
    alpha = (source_positions - left_idx).astype(np.float32)

    for t in range(T):
        out[t] = (1.0 - alpha[t]) * seq[left_idx[t]] + alpha[t] * seq[right_idx[t]]
    return out


def maybe_flip_sign_sequence(X_seq: np.ndarray, prob: float = 0.35) -> np.ndarray:
    """Horizontal flip in normalized landmark space by negating x coordinates."""
    if np.random.rand() > prob:
        return X_seq.astype(np.float32).copy()
    seq = X_seq.astype(np.float32).copy().reshape(X_seq.shape[0], 21, 3)
    seq[:, :, 0] *= -1.0
    return seq.reshape(X_seq.shape[0], 63).astype(np.float32)


def build_augmented_training_set(X_raw: np.ndarray, y: np.ndarray, augment_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    if augment_factor <= 0:
        return X_raw.astype(np.float32), y.astype(np.int64)

    X_list = [X_raw.astype(np.float32)]
    y_list = [y.astype(np.int64)]
    for _ in range(augment_factor):
        aug = maybe_flip_sign_sequence(time_warp_sequence(jitter_sequence(X_raw)))
        X_list.append(aug.astype(np.float32))
        y_list.append(y.astype(np.int64))
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ============================================================
# I/O
# ============================================================


def load_raw_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing dataset: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    X = data["X"].astype(np.float32)  # (N, seq_len, 63)
    y = data["y"].astype(np.int64)    # (N,)

    if X.ndim != 3 or X.shape[2] != 63:
        raise ValueError("Expected X shape (N, seq_len, 63).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Feature/sample count mismatch between X and y.")
    return X, y


def load_label_names(label_map_path: str) -> Optional[List[str]]:
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


# ============================================================
# Model selection
# ============================================================


def build_candidate_models(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "svc_rbf",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=8.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
            ]),
        ),
        (
            "svc_linear",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=1.2, probability=True, class_weight="balanced", random_state=seed)),
            ]),
        ),
        (
            "extra_trees",
            Pipeline([
                ("clf", ExtraTreesClassifier(
                    n_estimators=450,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                    min_samples_leaf=1,
                )),
            ]),
        ),
        (
            "random_forest",
            Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=450,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                    min_samples_leaf=1,
                )),
            ]),
        ),
        (
            "logreg",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=seed,
                    multi_class="auto",
                )),
            ]),
        ),
    ]


def pick_best_model(X_train: np.ndarray, y_train: np.ndarray, seed: int, use_cv: bool) -> tuple[str, object, dict[str, float]]:
    models = build_candidate_models(seed)
    scores: dict[str, float] = {}

    unique_classes = np.unique(y_train)
    class_counts = Counter(y_train.tolist())
    can_cv = use_cv and len(unique_classes) >= 2 and min(class_counts.values()) >= 2 and X_train.shape[0] >= 8

    if can_cv:
        n_splits = min(5, int(min(class_counts.values())))
        n_splits = max(2, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for name, model in models:
            try:
                vals = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
                scores[name] = float(np.mean(vals))
            except Exception:
                scores[name] = -1.0
    else:
        # small dataset fallback: training accuracy is optimistic, but better than blind selection
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_train)
                scores[name] = float(f1_score(y_train, pred, average="macro"))
            except Exception:
                scores[name] = -1.0

    best_name = max(scores, key=scores.get)
    best_model = dict(models)[best_name]
    return best_name, best_model, scores


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join("data", "asl_words_handseq.npz"))
    ap.add_argument("--label_map", default=os.path.join("data", "asl_words_handseq_labels.json"))
    ap.add_argument("--out_dir", default=os.path.join("models"))
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--augment_factor", type=int, default=2, help="Number of augmented training copies to add.")
    ap.add_argument("--no_cv", action="store_true", help="Disable model-selection CV.")
    args = ap.parse_args()

    cfg = TrainConfig(
        npz_path=args.npz,
        label_map_path=args.label_map,
        out_dir=args.out_dir,
        test_size=float(args.test_size),
        seed=int(args.seed),
        augment_factor=int(max(0, args.augment_factor)),
        use_cv=not bool(args.no_cv),
    )

    X_raw, y = load_raw_data(cfg.npz_path)
    label_names = load_label_names(cfg.label_map_path)

    if X_raw.shape[0] < 2:
        raise ValueError("Need at least 2 sequence samples to train the hand-sequence word model.")

    class_counts = Counter(y.tolist())
    print("[handseq-train] samples per class:", {int(k): int(v) for k, v in sorted(class_counts.items())})
    if len(class_counts) < 2:
        raise ValueError("Need at least 2 distinct labels to train a classifier.")

    can_hold_out = X_raw.shape[0] >= 8 and min(class_counts.values()) >= 2
    if can_hold_out:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw,
            y,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y,
        )
    else:
        X_train_raw, X_val_raw, y_train, y_val = X_raw, X_raw, y, y
        print("[handseq-train] dataset is too small for a reliable stratified holdout; training and reporting on all samples.")

    X_train_aug_raw, y_train_aug = build_augmented_training_set(X_train_raw, y_train, cfg.augment_factor)

    X_train = np.stack([summarize_sequence(seq) for seq in X_train_aug_raw], axis=0)
    X_val = np.stack([summarize_sequence(seq) for seq in X_val_raw], axis=0)

    print(f"[handseq-train] raw train samples: {X_train_raw.shape[0]}")
    print(f"[handseq-train] augmented train samples: {X_train.shape[0]}")
    print(f"[handseq-train] validation samples: {X_val.shape[0]}")
    print(f"[handseq-train] feature dim: {X_train.shape[1]}")

    best_name, best_model, model_scores = pick_best_model(X_train, y_train_aug, cfg.seed, cfg.use_cv)
    print("[handseq-train] candidate scores:")
    for name, score in sorted(model_scores.items()):
        print(f"  - {name}: {score:.4f}")
    print(f"[handseq-train] selected model: {best_name}")

    best_model.fit(X_train, y_train_aug)
    y_pred = best_model.predict(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    f1m = float(f1_score(y_val, y_pred, average="macro"))
    print(f"[handseq-train] accuracy: {acc:.4f}")
    print(f"[handseq-train] macro_f1: {f1m:.4f}")

    print("[handseq-train] report:")
    if label_names is not None:
        print(classification_report(y_val, y_pred, target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_val, y_pred, zero_division=0))

    print("[handseq-train] confusion:")
    print(confusion_matrix(y_val, y_pred))

    os.makedirs(cfg.out_dir, exist_ok=True)
    model_path = os.path.join(cfg.out_dir, "asl_word_handseq_model.pkl")
    meta_path = os.path.join(cfg.out_dir, "asl_word_handseq_model_meta.json")

    joblib.dump(best_model, model_path)
    meta = {
        "selected_model": best_name,
        "candidate_scores": model_scores,
        "feature_dim": int(X_train.shape[1]),
        "seq_len": int(X_raw.shape[1]),
        "train_samples_raw": int(X_train_raw.shape[0]),
        "train_samples_augmented": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "accuracy": acc,
        "macro_f1": f1m,
        "class_counts": {str(int(k)): int(v) for k, v in sorted(class_counts.items())},
        "labels": label_names,
        "config": asdict(cfg),
        "summary_features": ["mean", "std", "min", "max", "range", "mean_abs_delta", "last"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved -> {model_path}")
    print(f"Saved -> {meta_path}")


if __name__ == "__main__":
    main()
