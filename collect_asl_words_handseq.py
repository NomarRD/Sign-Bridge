from __future__ import annotations

import os
import time
import json
import math
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Config
# ============================================================

@dataclass
class CollectorConfig:
    labels: List[str]
    seq_len: int
    out_path: str
    labels_out_path: str
    hand_model_path: str
    max_hands: int = 1
    record_key: str = "r"
    cancel_key: str = "c"
    mirror_preview: bool = True
    ready_hold_frames: int = 10
    cooldown_ms: int = 250
    motion_min: float = 0.004
    motion_max: float = 0.28
    no_hand_reset_frames: int = 12
    save_preview_delay_sec: float = 0.20
    min_brightness: float = 25.0
    max_brightness: float = 245.0
    min_hand_box_size: float = 0.08


# ============================================================
# Feature helpers
# ============================================================


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Normalize landmarks for translation and scale invariance.

    @param lm: Raw landmarks array of shape (21, 3).
    @return: Normalized landmarks array of shape (21, 3).
    """
    pts = lm.astype(np.float32).copy()
    wrist = pts[0, :].copy()

    pts[:, 0] -= wrist[0]
    pts[:, 1] -= wrist[1]
    pts[:, 2] -= wrist[2]

    xy_scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)) + 1e-6)
    pts[:, :2] /= xy_scale
    pts[:, 2] /= xy_scale
    return pts


def flatten_features(lm_norm: np.ndarray) -> np.ndarray:
    """Flatten normalized landmarks to (63,)."""
    return lm_norm.reshape(-1).astype(np.float32)


def hand_box_size(lm: np.ndarray) -> float:
    """Approximate hand size using normalized image coordinates."""
    x_span = float(np.max(lm[:, 0]) - np.min(lm[:, 0]))
    y_span = float(np.max(lm[:, 1]) - np.min(lm[:, 1]))
    return max(x_span, y_span)


def frame_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def sequence_motion_score(seq: List[np.ndarray]) -> float:
    if len(seq) < 2:
        return 0.0
    vals = []
    for i in range(1, len(seq)):
        vals.append(float(np.linalg.norm(seq[i] - seq[i - 1])))
    return float(np.mean(vals)) if vals else 0.0


def draw_outlined_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.65,
    fg: Tuple[int, int, int] = (255, 255, 255),
    outline: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    outline_thickness: int = 4,
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        fg,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# MediaPipe Tasks HandLandmarker wrapper
# ============================================================

class HandLandmarkExtractor:
    """MediaPipe Tasks HandLandmarker wrapper."""

    def __init__(self, model_path: str, max_hands: int = 1) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing Tasks model file: {model_path}. Place 'hand_landmarker.task' inside models/."
            )
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[object]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None, None
        lms = result.hand_landmarks[0]
        lm_array = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
        return lm_array, result

    def draw(self, frame_bgr: np.ndarray, result: object) -> None:
        if not getattr(result, "hand_landmarks", None):
            return
        h, w = frame_bgr.shape[:2]
        for hand in result.hand_landmarks:
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)


# ============================================================
# Dataset I/O
# ============================================================


def load_npz(npz_path: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset if present; otherwise return empty arrays."""
    if not os.path.exists(npz_path):
        return np.zeros((0, seq_len, 63), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    data = np.load(npz_path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if X.ndim != 3 or X.shape[2] != 63:
        raise ValueError("Expected X shape (N, seq_len, 63).")
    if X.shape[1] != seq_len:
        raise ValueError(
            f"Existing dataset seq_len={X.shape[1]} != requested {seq_len}. Use same --seq_len."
        )
    return X, y


def save_npz(npz_path: str, X: np.ndarray, y: np.ndarray) -> None:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez_compressed(npz_path, X=X.astype(np.float32), y=y.astype(np.int64))


def save_label_map(labels_out_path: str, labels: List[str], label_to_idx: dict[str, int]) -> None:
    os.makedirs(os.path.dirname(labels_out_path) or ".", exist_ok=True)
    with open(labels_out_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label_to_idx": label_to_idx}, f, indent=2)


# ============================================================
# Quality checks
# ============================================================


def validate_frame_quality(frame_bgr: np.ndarray, hand_lm: Optional[np.ndarray], cfg: CollectorConfig) -> Tuple[bool, str]:
    brightness = frame_brightness(frame_bgr)
    if brightness < cfg.min_brightness:
        return False, "Scene too dark"
    if brightness > cfg.max_brightness:
        return False, "Scene too bright"
    if hand_lm is None:
        return False, "No hand detected"
    if hand_box_size(hand_lm) < cfg.min_hand_box_size:
        return False, "Hand appears too small / too far"
    return True, "Ready"


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="YES,NO", help="Comma-separated labels.")
    ap.add_argument("--seq_len", type=int, default=25, help="Frames per sample.")
    ap.add_argument("--out", default=os.path.join("data", "asl_words_handseq.npz"))
    ap.add_argument("--labels_out", default=os.path.join("data", "asl_words_handseq_labels.json"))
    ap.add_argument("--hand_model", default=os.path.join("models", "hand_landmarker.task"))
    ap.add_argument("--motion_min", type=float, default=0.004, help="Minimum mean motion needed for a usable dynamic sample.")
    ap.add_argument("--motion_max", type=float, default=0.28, help="Maximum mean motion allowed before a sample is considered too noisy.")
    ap.add_argument("--ready_hold_frames", type=int, default=10, help="Frames with a good hand before recording starts cleanly.")
    ap.add_argument("--mirror", action="store_true", help="Mirror preview for a more natural webcam experience.")
    args = ap.parse_args()

    labels: List[str] = []
    seen_labels: set[str] = set()
    for raw_label in args.labels.split(","):
        label = raw_label.strip().upper()
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        labels.append(label)
    if not labels:
        raise ValueError("No labels provided. Use --labels YES,NO,...")

    cfg = CollectorConfig(
        labels=labels,
        seq_len=int(max(2, args.seq_len)),
        out_path=args.out,
        labels_out_path=args.labels_out,
        hand_model_path=args.hand_model,
        motion_min=float(args.motion_min),
        motion_max=float(args.motion_max),
        ready_hold_frames=int(max(1, args.ready_hold_frames)),
        mirror_preview=bool(args.mirror),
    )

    label_to_idx = {lab: i for i, lab in enumerate(cfg.labels)}
    save_label_map(cfg.labels_out_path, cfg.labels, label_to_idx)

    hand_extractor = HandLandmarkExtractor(cfg.hand_model_path, max_hands=cfg.max_hands)
    X, y = load_npz(cfg.out_path, cfg.seq_len)
    saved_total = int(y.shape[0])
    saved_counts = Counter(y.tolist())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    current_idx = 0
    current_label = cfg.labels[current_idx]

    recording = False
    ready_count = 0
    no_hand_streak = 0
    seq_buf: List[np.ndarray] = []
    last_save_msg = ""
    save_msg_until = 0.0

    window_name = "collect_asl_words_handseq"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if cfg.mirror_preview:
            frame = cv2.flip(frame, 1)

        now = time.time()
        hand_lm, hand_result = hand_extractor.extract(frame)
        feats: Optional[np.ndarray] = None
        quality_ok, quality_msg = validate_frame_quality(frame, hand_lm, cfg)

        if hand_lm is not None:
            hand_extractor.draw(frame, hand_result)
            feats = flatten_features(normalize_landmarks(hand_lm))
            no_hand_streak = 0
        else:
            no_hand_streak += 1

        if not quality_ok:
            ready_count = 0
            if recording and no_hand_streak >= cfg.no_hand_reset_frames:
                recording = False
                seq_buf = []
                last_save_msg = "Recording canceled: hand lost"
                save_msg_until = now + 1.6
        else:
            ready_count += 1

        draw_outlined_text(frame, "Dynamic Word Collector (hand-only, sequence)", (20, 35), 0.75)
        draw_outlined_text(frame, f"Label: {current_label} (press 1-{min(9, len(cfg.labels))})", (20, 68), 0.67)
        draw_outlined_text(
            frame,
            f"{cfg.record_key}=start  {cfg.cancel_key}=cancel  ESC=quit  seq_len={cfg.seq_len}",
            (20, 101),
            0.62,
        )
        draw_outlined_text(frame, f"Saved total: {saved_total}", (20, 134), 0.62)

        label_count = int(saved_counts.get(label_to_idx[current_label], 0))
        min_count = min(saved_counts.values()) if saved_counts else 0
        draw_outlined_text(frame, f"Saved for {current_label}: {label_count}", (20, 167), 0.62)
        draw_outlined_text(frame, f"Balance target (current min): {min_count}", (20, 200), 0.62)

        brightness = frame_brightness(frame)
        box_size = hand_box_size(hand_lm) if hand_lm is not None else 0.0
        motion_preview = sequence_motion_score(seq_buf) if seq_buf else 0.0
        draw_outlined_text(frame, f"Brightness: {brightness:.1f}", (20, 233), 0.56)
        draw_outlined_text(frame, f"Hand size: {box_size:.3f}", (20, 262), 0.56)
        draw_outlined_text(frame, f"Motion(avg): {motion_preview:.4f}", (20, 291), 0.56)

        status_color = (0, 220, 0) if quality_ok else (0, 0, 255)
        draw_outlined_text(frame, f"Status: {quality_msg}", (20, 324), 0.62, fg=status_color)

        if quality_ok and not recording:
            hold_left = max(0, cfg.ready_hold_frames - ready_count)
            draw_outlined_text(frame, f"Steady hold before start: {hold_left}", (20, 357), 0.58)

        if recording:
            draw_outlined_text(frame, f"RECORDING: {len(seq_buf)}/{cfg.seq_len}", (20, 390), 0.72, fg=(0, 255, 255))

        if now < save_msg_until and last_save_msg:
            draw_outlined_text(frame, last_save_msg, (20, 423), 0.60, fg=(255, 220, 0))

        # draw quick per-label stats along the right side
        y0 = 36
        for idx, label in enumerate(cfg.labels[:12]):
            cnt = int(saved_counts.get(idx, 0))
            draw_outlined_text(frame, f"{idx + 1}. {label}: {cnt}", (frame.shape[1] - 220, y0), 0.54)
            y0 += 26

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if ord("1") <= key <= ord("9"):
            idx = int(chr(key)) - 1
            if idx < len(cfg.labels):
                current_idx = idx
                current_label = cfg.labels[current_idx]
                ready_count = 0

        if key in (ord(cfg.record_key.lower()), ord(cfg.record_key.upper())) and not recording:
            if not quality_ok or ready_count < cfg.ready_hold_frames:
                last_save_msg = "Not ready yet: wait for stable hand and good lighting"
                save_msg_until = now + 1.6
            else:
                recording = True
                seq_buf = []
                last_save_msg = ""

        if key in (ord(cfg.cancel_key.lower()), ord(cfg.cancel_key.upper())) and recording:
            recording = False
            seq_buf = []
            last_save_msg = "Recording canceled"
            save_msg_until = now + 1.2

        if recording and feats is not None:
            seq_buf.append(feats)
            if len(seq_buf) >= cfg.seq_len:
                motion = sequence_motion_score(seq_buf)
                if motion < cfg.motion_min:
                    last_save_msg = f"Sample rejected: motion too low ({motion:.4f})"
                    save_msg_until = now + 1.8
                elif motion > cfg.motion_max:
                    last_save_msg = f"Sample rejected: motion too noisy ({motion:.4f})"
                    save_msg_until = now + 1.8
                else:
                    x_seq = np.stack(seq_buf[: cfg.seq_len], axis=0).astype(np.float32)
                    y_idx = int(label_to_idx[current_label])

                    X = np.concatenate([X, x_seq[None, :, :]], axis=0)
                    y = np.concatenate([y, np.array([y_idx], dtype=np.int64)], axis=0)

                    save_npz(cfg.out_path, X, y)
                    saved_total = int(y.shape[0])
                    saved_counts[y_idx] += 1
                    last_save_msg = f"Saved {current_label} sample #{saved_counts[y_idx]} (motion={motion:.4f})"
                    save_msg_until = now + 1.8
                    time.sleep(cfg.save_preview_delay_sec)

                recording = False
                seq_buf = []

    cap.release()
    cv2.destroyAllWindows()
    save_npz(cfg.out_path, X, y)
    save_label_map(cfg.labels_out_path, cfg.labels, label_to_idx)
    print(f"Wrote {int(y.shape[0])} samples -> {cfg.out_path}")


if __name__ == "__main__":
    main()
