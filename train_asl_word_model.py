from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Dataset
# ============================================================

def list_npz_files(data_dir: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith(".npz"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


class WordSeqDataset(Dataset):
    """
    Loads .npz samples produced by updated_collect_asl_words.py.

    Each file contains:
    - hand_seq (T,63)
    - face_seq (T,face_dim)
    - rel_seq  (T,63)
    - label (string)
    """

    def __init__(self, files: List[str], label_to_idx: Dict[str, int]) -> None:
        self.files = files
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[i]
        d = np.load(path, allow_pickle=True)
        hand = d["hand_seq"].astype(np.float32)
        face = d["face_seq"].astype(np.float32)
        rel = d["rel_seq"].astype(np.float32)

        x = np.concatenate([hand, face, rel], axis=1)  # (T,F)
        label = str(d["label"][0]) if "label" in d else os.path.basename(os.path.dirname(path))
        y = self.label_to_idx[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def split_files(files: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    n_val = int(round(len(files) * val_ratio))
    val = files[:n_val]
    train = files[n_val:]
    return train, val


# ============================================================
# Model: lightweight Temporal CNN
# ============================================================

class TemporalCNN(nn.Module):
    """
    1D CNN over time.
    Input: (B,T,F) -> transpose -> (B,F,T)

    Notes:
    - This is a strong baseline for word sequences.
    - If you later want even better results, swap to BiGRU/Transformer.
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B,128,1)
            nn.Flatten(),             # (B,128)
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        x = x.transpose(1, 2)  # (B,F,T)
        z = self.net(x)
        return self.classifier(z)


# ============================================================
# Train / Eval
# ============================================================

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.15
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    conf_threshold: float = 0.55  # useful for demo defaults


def resolve_device(requested: str) -> str:
    """Resolve a safe torch device.

    RTX 50-series (sm_120) may not be supported by stable PyTorch builds yet.
    If CUDA is requested but unsupported, fall back to CPU to avoid runtime crashes.

    @param requested: 'cuda' or 'cpu' (case-insensitive).
    @return: 'cuda' if usable, otherwise 'cpu'.
    """
    req = (requested or "").lower().strip()
    if req.startswith("cpu"):
        return "cpu"

    # Treat anything else as 'cuda' preference
    if not torch.cuda.is_available():
        return "cpu"

    # Check whether the installed torch build supports the current GPU arch.
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
        if supported and arch not in supported:
            return "cpu"
    except Exception:
        # If capability probing fails, try a tiny CUDA op; if it fails, fall back to CPU.
        pass

    try:
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * int(x.size(0))

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train(cfg: TrainConfig) -> None:
    # Resolve CUDA/CPU safely (RTX 50xx may require CPU fallback)
    cfg.device = resolve_device(cfg.device)
    files = list_npz_files(cfg.data_dir)
    if not files:
        raise RuntimeError(f"No .npz files found under: {cfg.data_dir}")

    # Labels are folder names under data_dir
    labels = sorted({os.path.basename(os.path.dirname(p)) for p in files})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    train_files, val_files = split_files(files, cfg.val_ratio, cfg.seed)

    train_ds = WordSeqDataset(train_files, label_to_idx)
    val_ds = WordSeqDataset(val_files, label_to_idx)

    # Infer feature_dim, seq_len from first sample
    x0, _y0 = train_ds[0]
    seq_len = int(x0.shape[0])
    feature_dim = int(x0.shape[1])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = TemporalCNN(feature_dim=feature_dim, num_classes=len(labels)).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "asl_word_seq_model.pt")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save metadata for the demo to stay in sync
    meta = {
        "labels": labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "model_type": "TemporalCNN",
        "conf_threshold_suggested": cfg.conf_threshold,
    }
    with open(os.path.join(cfg.out_dir, "asl_word_seq_model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[train] device={cfg.device}")
    print(f"[train] samples={len(files)} | train={len(train_files)} | val={len(val_files)}")
    print(f"[train] labels={labels}")
    print(f"[train] seq_len={seq_len} | feature_dim={feature_dim}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            loss_sum += float(loss.item()) * int(x.size(0))
            seen += int(x.size(0))

        train_loss = loss_sum / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "meta": meta,
            }
            torch.save(ckpt, best_path)

    print(f"[done] best_val_acc={best_val_acc:.3f}")
    print(f"[done] saved: {best_path}")
    print(f"[done] meta : {os.path.join(cfg.out_dir, 'asl_word_seq_model_meta.json')}")


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train ASL word sequence model from .npz samples.")
    ap.add_argument("--data_dir", default=os.path.join("data", "words_seq"), help="Directory containing label folders with .npz files.")
    ap.add_argument("--out_dir", default=os.path.join("models"), help="Output directory for model + meta.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training device preference. Falls back to CPU if CUDA is unsupported.")
    args = ap.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)