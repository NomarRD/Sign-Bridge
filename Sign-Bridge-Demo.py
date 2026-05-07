"""
Sign-Bridge Demo (Scenarios) - MediaPipe Tasks Hand + Face Landmarkers + OpenCV

Implements:
- Still image processing (letters)
- Real-time webcam processing
- Confidence threshold + Unknown output
- No-sign present handling
- Letter smoothing (majority vote)
- Fast letter stabilizer with release-based cooldown
- Word mode toggle using a special key (default '`')
- Word mode toggle ALSO using a special SIGN (predicted letter label, default '`')
- Word mode toggle ALSO using a RECORDED custom gesture (template matching; press 'r' to record)
- Sequence buffering for dynamic words (real model or stub)
- Explanation output (rule-based stub)
- Motion debug overlay (optional)
- Hybrid autocorrect for fingerspelled words:
  - local fuzzy correction first
  - optional OpenAI fallback in background when local correction is weak
- OpenAI debug status in terminal and overlay
- Trained actual space character label ' ' support for word separation
- Transcript clear hotkey (default 'c')
- Corrected transcript display now prefers OpenAI output when available
- Explanation line now better preserves OpenAI output when available

Notes:
- Letters use HAND landmarks only.
- Words use HAND + FACE landmarks.
- The actual space label ' ' is treated as a real word separator in the transcript.
- The '=' label is also transcribed as a real space so fingerspelled words can form sentences.
"""

from __future__ import annotations

import os
import re
import csv
import time
from datetime import datetime
import unicodedata
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Optional, Tuple, Callable
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# Configuration
# ============================================================

@dataclass
class DemoConfig:
    """
    Runtime configuration for the demo.
    """
    letter_conf_threshold: float = 0.48
    word_conf_threshold: float = 0.70

    smoothing_window: int = 4
    seq_len: int = 25

    stable_letter_frames: int = 2
    letter_release_frames: int = 3

    max_hands: int = 1

    hand_model_path: str = os.path.join("models", "hand_landmarker.task")
    face_model_path: str = os.path.join("models", "face_landmarker.task")

    letter_model_path: str = os.path.join("models", "asl_model.pkl")
    letter_scaler_path: str = ""
    letter_label_encoder_path: str = os.path.join("models", "asl_label_encoder.pkl")

    word_model_path: str = os.path.join("models", "asl_word_model.pkl")
    word_label_encoder_path: str = os.path.join("models", "asl_word_label_encoder.pkl")

    word_handseq_model_path: str = os.path.join("models", "asl_word_handseq_model.pkl")
    word_handseq_labels_path: str = os.path.join("models", "asl_word_handseq_labels.json")
    word_handseq_seq_len: int = 20
    word_handseq_conf_threshold: float = 0.70
    word_handseq_margin: float = 0.05

    word_motion_window: int = 8
    word_motion_threshold: float = 0.020

    word_static_model_path: str = os.path.join("models", "asl_word_static_model.pkl")
    word_static_label_encoder_path: str = os.path.join("models", "asl_word_static_label_encoder.pkl")

    static_motion_window: int = 6
    static_motion_threshold: float = 0.015

    word_toggle_label: str = "`"
    toggle_streak_required: int = 6
    toggle_cooldown_sec: float = 1.25

    word_toggle_key: str = "`"

    toggle_gesture_enabled: bool = True
    toggle_gesture_template_path: str = os.path.join("models", "toggle_gesture.npy")
    toggle_gesture_record_key: str = "r"
    toggle_gesture_record_frames: int = 30
    toggle_gesture_threshold: float = 0.92
    toggle_gesture_streak_required: int = 8
    toggle_gesture_cooldown_sec: float = 1.25

    clear_transcript_key: str = "c"
    performance_toggle_key: str = "p"
    performance_log_path: str = os.path.join("logs", "sign_bridge_performance.csv")

    ai_context_enabled: bool = True
    ai_backend: str = "local_gemma"
    ai_context_model: str = "gpt-5.4"
    ai_context_history: int = 12
    ai_context_max_chars: int = 140
    ai_context_min_tokens: int = 3

    local_llm_enabled: bool = True
    local_llm_download_via: str = "kagglehub"
    local_llm_model_ref: str = "google/gemma-3/transformers/gemma-3-1b-it"
    local_llm_model_path: str = ""
    local_llm_device_map: str = "auto"
    local_llm_dtype: str = "auto"
    local_llm_max_new_tokens: int = 18
    local_llm_explain_max_new_tokens: int = 72
    local_llm_do_sample: bool = False
    local_llm_temperature: float = 0.0
    local_llm_top_p: float = 1.0
    local_llm_repetition_penalty: float = 1.02
    local_llm_trust_remote_code: bool = True
    openai_fallback_enabled: bool = True

    fuzzy_correction_enabled: bool = True
    fuzzy_min_word_len: int = 3
    fuzzy_match_cutoff: float = 0.72
    fuzzy_letter_commit_len: int = 12

    repeat_letter_cooldown_sec: float = 0.75
    min_letter_commit_gap_sec: float = 0.12

    # Word mode uses whole-word commits, so keep a larger delay
    # between accepted words without slowing down letter mode.
    word_commit_cooldown_sec: float = 3.0

    ai_candidate_shortlist_size: int = 8
    ai_candidate_shortlist_cutoff: float = 0.55
    ai_accept_close_cutoff: float = 0.82
    ai_accept_max_len_delta: int = 1
    ai_accept_max_edit_ratio: float = 0.34
    ai_accept_min_shared_chars: int = 2

    fuzzy_accept_close_cutoff: float = 0.84
    fuzzy_accept_max_len_delta: int = 1
    fuzzy_accept_max_edit_ratio: float = 0.34
    fuzzy_accept_min_shared_chars: int = 2


# ============================================================
# MediaPipe Tasks Hand Landmarker Wrapper
# ============================================================

class HandLandmarkExtractor:
    """
    Extract 21 hand landmarks (x,y,z) using MediaPipe Tasks Hand Landmarker.
    """

    def __init__(self, model_path: str, max_hands: int = 1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing Tasks model file: {model_path}. "
                "Download a real 'hand_landmarker.task' into the models/ folder."
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
# MediaPipe Tasks Face Landmarker Wrapper
# ============================================================

class FaceLandmarkExtractor:
    """
    Extract face landmarks (x,y,z) using MediaPipe Tasks Face Landmarker.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing Tasks model file: {model_path}. "
                "Download a real 'face_landmarker.task' into the models/ folder."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[object]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None, None

        lms = result.face_landmarks[0]
        lm_array = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
        return lm_array, result

    def draw(self, frame_bgr: np.ndarray, result: object, step: int = 6) -> None:
        if not getattr(result, "face_landmarks", None):
            return

        h, w = frame_bgr.shape[:2]
        for face in result.face_landmarks:
            for i, lm in enumerate(face):
                if i % step != 0:
                    continue
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 1, (255, 0, 0), -1)


# ============================================================
# Feature Engineering
# ============================================================

def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    anchor = lm[0, :2].copy()
    xy = lm[:, :2] - anchor
    scale = np.max(np.linalg.norm(xy, axis=1)) + 1e-6
    xy = xy / scale
    z = lm[:, 2:3]
    return np.concatenate([xy, z], axis=1)


def flatten_features(lm_norm: np.ndarray) -> np.ndarray:
    return lm_norm.reshape(-1).astype(np.float32)


def face_center_and_scale(face_lm: np.ndarray) -> Tuple[np.ndarray, float]:
    center = np.mean(face_lm, axis=0).astype(np.float32)
    d = np.linalg.norm(face_lm[:, :2] - center[:2], axis=1)
    scale = float(np.max(d) + 1e-6)
    return center, scale


def hand_relative_to_face(hand_lm: np.ndarray, face_center: np.ndarray, face_scale: float) -> np.ndarray:
    rel = (hand_lm - face_center[None, :]) / float(face_scale)
    return rel.reshape(-1).astype(np.float32)


def make_word_features(
    hand_feats_seq: np.ndarray,
    face_feats_seq: np.ndarray,
    rel_feats_seq: np.ndarray
) -> np.ndarray:
    seq = np.concatenate([hand_feats_seq, face_feats_seq, rel_feats_seq], axis=1)
    mu = np.mean(seq, axis=0)
    sd = np.std(seq, axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)


def summarize_hand_sequence(hand_feats_seq: np.ndarray) -> np.ndarray:
    mu = np.mean(hand_feats_seq, axis=0)
    sd = np.std(hand_feats_seq, axis=0)
    last = hand_feats_seq[-1]
    return np.concatenate([mu, sd, last], axis=0).astype(np.float32)


def fit_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    d = int(vec.size)
    if d == target_dim:
        return vec.astype(np.float32)
    if d > target_dim:
        return vec[:target_dim].astype(np.float32)
    pad = np.zeros((target_dim - d,), dtype=np.float32)
    return np.concatenate([vec.astype(np.float32), pad], axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = float(np.linalg.norm(a)) + 1e-8
    nb = float(np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def load_toggle_template(path: str) -> Optional[np.ndarray]:
    try:
        if not os.path.exists(path):
            return None
        t = np.load(path)
        if t is None:
            return None
        t = np.array(t, dtype=np.float32).reshape(-1)
        if t.size == 0:
            return None
        return t
    except Exception:
        return None


def save_toggle_template(path: str, template: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, template.astype(np.float32))
        return True
    except Exception:
        return False


def display_label_text(label: str) -> str:
    if label == " ":
        return "[space]"
    return label


def format_csv_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def compute_total_time_ms(*values: float) -> float:
    total = 0.0
    for value in values:
        try:
            total += float(value or 0.0)
        except Exception:
            pass
    return total


def transcript_to_string(tokens: list[str]) -> str:
    out: list[str] = []
    prev_space = True

    for tok in tokens:
        if tok == " ":
            if not prev_space:
                out.append(" ")
            prev_space = True
            continue

        out.append(tok)
        prev_space = False

    return "".join(out).strip()


def replace_trailing_candidate_in_transcript(
    tokens: list[str],
    candidate_letters: str,
    corrected_word: str
) -> str:
    """
    Build a corrected transcript by replacing the trailing fingerspelled
    candidate with the corrected word.
    """
    raw_text = transcript_to_string(tokens)

    if not corrected_word:
        return raw_text

    if candidate_letters and raw_text.endswith(candidate_letters):
        return (raw_text[:-len(candidate_letters)] + corrected_word).strip()

    return raw_text


def sanitize_text_for_display(text: str, max_len: Optional[int] = None) -> str:
    """
    Convert model output into OpenCV-safe printable ASCII so the overlay does not
    show unsupported Unicode glyphs as ???.
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[^A-Za-z0-9 .,!?:;'\"\-_/()\[\]{}@#%&*+=<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_len is not None and max_len > 0 and len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."

    return text


def sanitize_token_text(text: str, fallback: str = "") -> str:
    cleaned = normalize_token_for_match(sanitize_text_for_display(text))
    return cleaned or fallback


def strip_known_llm_artifacts(text: str) -> str:
    cleaned = sanitize_text_for_display(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?i)\b(?:apex|explained|candidate|final|source|mode|transcript|corrected transcript|local gemma|local|gemma|openai)\b", " ", cleaned)
    cleaned = re.sub(r"(?i)(?:apex\s*){1,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    return cleaned


def extract_best_candidate_word(raw_text: str, original: str, vocabulary: list[str]) -> str:
    raw = strip_known_llm_artifacts(raw_text)
    if not raw:
        return original

    tokens = re.findall(r"[A-Za-z]+", raw.lower())
    banned = {
        "apex", "explained", "candidate", "final", "source", "mode", "word", "input",
        "output", "transcript", "corrected", "local", "gemma", "openai", "original",
        "unchanged", "english", "lowercase", "return"
    }
    original_norm = normalize_token_for_match(original)
    vocab_map = {normalize_token_for_match(word): word for word in vocabulary}
    vocab_norm = list(vocab_map.keys())

    filtered: list[str] = []
    for tok in tokens:
        norm_tok = normalize_token_for_match(tok)
        if norm_tok in banned:
            continue
        if len(norm_tok) < 2:
            continue
        filtered.append(norm_tok)

    if not filtered:
        return original_norm or original

    for tok in filtered:
        if tok in vocab_map:
            return vocab_map[tok]

    close_source = filtered + ([original_norm] if original_norm else [])
    for candidate in close_source:
        close = get_close_matches(candidate, vocab_norm, n=1, cutoff=0.60)
        if close:
            return vocab_map[close[0]]

    return filtered[0]


def clean_overlay_reason(text: str, max_len: Optional[int] = None) -> str:
    cleaned = strip_known_llm_artifacts(text)
    if not cleaned:
        return ""

    cleaned = re.sub(r"^(?:OpenAI|Local|AI)\s*:\s*", "", cleaned, count=1).strip()
    cleaned = re.sub(r"^(?:Why|Explanation|Summary)\s*:\s*", "", cleaned, count=1).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    if "." in cleaned:
        cleaned = cleaned.split(".", 1)[0].strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    if max_len is not None and max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."

    return cleaned


def build_corrected_transcript_from_decision(
    tokens: list[str],
    correction_decision: "CorrectionDecision",
    fallback_text: str = ""
) -> str:
    raw_text = transcript_to_string(tokens)

    if correction_decision.changed and correction_decision.final_suggestion:
        candidate = sanitize_token_text(correction_decision.raw_candidate)
        corrected = sanitize_token_text(correction_decision.final_suggestion)
        if candidate and corrected:
            updated = replace_trailing_candidate_in_transcript(tokens, candidate, corrected)
            if updated != raw_text:
                return updated
            return corrected

    return fallback_text or raw_text


# ============================================================
# Overlay helpers
# ============================================================

def wrap_text_to_width(
    text: str,
    max_width_px: int,
    font_face: int,
    font_scale: float,
    thickness: int
) -> list[str]:
    def _fits(s: str) -> bool:
        (tw, _), _ = cv2.getTextSize(s, font_face, font_scale, thickness)
        return tw <= max_width_px

    def _split_long_token(tok: str) -> list[str]:
        chunks: list[str] = []
        cur = ""
        for ch in tok:
            test = cur + ch
            if cur == "" or _fits(test):
                cur = test
            else:
                chunks.append(cur)
                cur = ch
        if cur:
            chunks.append(cur)
        return chunks
    paragraphs = [p.strip() for p in str(text).replace("\r", "\n").split("\n")]
    lines: list[str] = []

    for paragraph in paragraphs:
        if not paragraph:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        words = paragraph.split()
        cur = ""
        for w in words:
            if not _fits(w):
                parts = _split_long_token(w)
                for p in parts:
                    test = (cur + " " + p).strip()
                    if cur and _fits(test):
                        cur = test
                    else:
                        if cur:
                            lines.append(cur)
                        cur = p
                continue

            test = (cur + " " + w).strip()
            if _fits(test):
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w

        if cur:
            lines.append(cur)

    return lines or [""]


def ellipsize_line_to_width(
    text: str,
    max_width_px: int,
    font_face: int,
    font_scale: float,
    thickness: int
) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    ellipsis = "..."
    (tw, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    if tw <= max_width_px:
        return text

    while text:
        candidate = text.rstrip() + ellipsis
        (cw, _), _ = cv2.getTextSize(candidate, font_face, font_scale, thickness)
        if cw <= max_width_px:
            return candidate
        text = text[:-1]

    return ellipsis


def trim_lines_to_fit(
    lines: list[str],
    max_lines: Optional[int],
    max_width_px: int,
    font_face: int,
    font_scale: float,
    thickness: int,
) -> list[str]:
    if max_lines is None or max_lines <= 0 or len(lines) <= max_lines:
        return lines

    trimmed = list(lines[:max_lines])
    trimmed[-1] = ellipsize_line_to_width(
        trimmed[-1],
        max_width_px,
        font_face,
        font_scale,
        thickness,
    )
    return trimmed


def line_height_px(font_face: int, font_scale: float, thickness: int, line_gap: int = 6) -> int:
    (_, th), baseline = cv2.getTextSize("Ag", font_face, font_scale, thickness)
    return int(th + baseline + line_gap)


def max_lines_for_space(
    frame: np.ndarray,
    y: int,
    font_face: int,
    font_scale: float,
    thickness: int,
    line_gap: int = 6,
    bottom_margin: int = 16,
) -> int:
    available = int(frame.shape[0]) - int(y) - int(bottom_margin)
    lh = max(1, line_height_px(font_face, font_scale, thickness, line_gap=line_gap))
    return max(0, available // lh)


def draw_text_lines(
    frame: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    font_face: int,
    font_scale: float,
    thickness_fg: int,
    thickness_bg: int,
    line_gap: int = 6,
    bottom_margin: int = 16
) -> int:
    h = int(frame.shape[0])

    for line in lines:
        (_, th), baseline = cv2.getTextSize(line or " ", font_face, font_scale, thickness_fg)
        if y + th + baseline >= (h - bottom_margin):
            break

        cv2.putText(frame, line, (x, y), font_face, font_scale, (0, 0, 0), thickness_bg)
        cv2.putText(frame, line, (x, y), font_face, font_scale, (255, 255, 255), thickness_fg)
        y += th + baseline + line_gap

    return y


def draw_wrapped_block(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_width_px: int,
    font_face: int,
    font_scale: float,
    thickness_fg: int,
    thickness_bg: int,
    max_lines: Optional[int] = None,
    line_gap: int = 6,
    bottom_margin: int = 16,
) -> int:
    available_lines = max_lines_for_space(
        frame,
        y,
        font_face,
        font_scale,
        thickness_fg,
        line_gap=line_gap,
        bottom_margin=bottom_margin,
    )
    if max_lines is None:
        block_max_lines = available_lines
    else:
        block_max_lines = min(max_lines, available_lines)

    if block_max_lines <= 0:
        return y

    lines = wrap_text_to_width(text, max_width_px, font_face, font_scale, thickness_fg)
    lines = trim_lines_to_fit(
        lines,
        block_max_lines,
        max_width_px,
        font_face,
        font_scale,
        thickness_fg,
    )
    return draw_text_lines(
        frame,
        lines,
        x,
        y,
        font_face,
        font_scale,
        thickness_fg,
        thickness_bg,
        line_gap=line_gap,
        bottom_margin=bottom_margin,
    )


def draw_overlay_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    alpha: float = 0.42,
) -> None:
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x))
    y1 = max(0, min(h - 1, y))
    x2 = max(x1 + 1, min(w, x + width))
    y2 = max(y1 + 1, min(h, y + height))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)



def block_height_px(
    lines: list[str],
    font_face: int,
    font_scale: float,
    thickness: int,
    line_gap: int = 6,
) -> int:
    if not lines:
        return 0
    return len(lines) * line_height_px(font_face, font_scale, thickness, line_gap=line_gap)


def fit_overlay_layout(
    frame: np.ndarray,
    blocks: list[dict[str, Any]],
    max_width_px: int,
    available_height_px: int,
    font_face: int,
    min_scale: float = 0.30,
    scale_step: float = 0.02,
) -> list[dict[str, Any]]:
    """
    Wrap every overlay block and shrink body/title font scales until all text fits.
    This keeps the overlay readable on different webcam resolutions while avoiding
    hard clipping when Why/Summary become longer than expected.
    """
    prepared: list[dict[str, Any]] = []

    for shrink in range(0, 16):
        prepared = []
        total_height = 0

        for block in blocks:
            scale = max(min_scale, float(block["scale"]) - shrink * scale_step)
            gap = int(block.get("line_gap", 4))
            lines = wrap_text_to_width(
                block["text"],
                max_width_px,
                font_face,
                scale,
                int(block["thickness_fg"]),
            )
            prepared.append({
                "text": block["text"],
                "lines": lines,
                "scale": scale,
                "thickness_fg": int(block["thickness_fg"]),
                "thickness_bg": int(block["thickness_bg"]),
                "line_gap": gap,
            })
            total_height += block_height_px(
                lines,
                font_face,
                scale,
                int(block["thickness_fg"]),
                line_gap=gap,
            ) + 1

        if total_height <= available_height_px:
            return prepared

    if not prepared:
        return prepared

    remaining = max(0, int(available_height_px))
    fallback: list[dict[str, Any]] = []

    for idx, block in enumerate(prepared):
        scale = float(block["scale"])
        gap = int(block["line_gap"])
        lh = line_height_px(font_face, scale, int(block["thickness_fg"]), line_gap=gap)
        max_lines = max(1, remaining // max(1, lh))
        lines = trim_lines_to_fit(
            block["lines"],
            max_lines,
            max_width_px,
            font_face,
            scale,
            int(block["thickness_fg"]),
        )
        fallback.append({
            **block,
            "lines": lines,
        })
        remaining -= block_height_px(
            lines,
            font_face,
            scale,
            int(block["thickness_fg"]),
            line_gap=gap,
        ) + 1
        if remaining <= 0:
            break

    return fallback


def draw_fitted_overlay_blocks(
    frame: np.ndarray,
    x: int,
    y: int,
    max_width_px: int,
    available_height_px: int,
    blocks: list[dict[str, Any]],
    font_face: int,
    bottom_margin: int = 16,
) -> int:
    prepared = fit_overlay_layout(
        frame,
        blocks,
        max_width_px=max_width_px,
        available_height_px=available_height_px,
        font_face=font_face,
    )

    for block in prepared:
        y = draw_text_lines(
            frame,
            block["lines"],
            x,
            y,
            font_face,
            float(block["scale"]),
            int(block["thickness_fg"]),
            int(block["thickness_bg"]),
            line_gap=int(block["line_gap"]),
            bottom_margin=bottom_margin,
        )
        y += 1

    return y


# ============================================================
# Recognition explanation structures
# ============================================================

@dataclass
class LetterExplanationTrace:
    """
    Structured explanation for a letter prediction.
    """
    raw_label: str
    final_label: str
    confidence: float
    threshold: float
    fingers_extended: list[str] = field(default_factory=list)
    geometric_cues: dict[str, float] = field(default_factory=dict)
    hand_state: str = ""
    label_reason: str = ""

    def short_text(self) -> str:
        shown_raw = display_label_text(self.raw_label)
        shown_final = display_label_text(self.final_label)
        finger_text = ", ".join(self.fingers_extended) if self.fingers_extended else "none"
        return (
            f"Raw {shown_raw}, final {shown_final}, conf {self.confidence:.2f} vs threshold {self.threshold:.2f}. "
            f"Hand looked {self.hand_state}. Extended fingers: {finger_text}. {self.label_reason}"
        )

    def cue_text(self) -> str:
        if not self.geometric_cues:
            return "No geometric cues available."
        order = ["thumb_index", "index_middle", "middle_ring", "ring_pinky", "thumb_pinky", "thumb_wrist", "palm_spread"]
        parts: list[str] = []
        for key in order:
            if key in self.geometric_cues:
                parts.append(f"{key}={self.geometric_cues[key]:.3f}")
        for key, value in self.geometric_cues.items():
            if key not in order:
                parts.append(f"{key}={value:.3f}")
        return "Cues: " + ", ".join(parts) + "."

    def detailed_text(self) -> str:
        return f"{self.short_text()} {self.cue_text()}"


@dataclass
class CorrectionDecision:
    """
    Tracks how a fingerspelled candidate was corrected.
    """
    raw_candidate: str = ""
    local_suggestion: str = ""
    final_suggestion: str = ""
    source: str = "Local"
    changed: bool = False
    nearby_matches: list[str] = field(default_factory=list)
    reason: str = "No correction decision yet."

    def short_text(self) -> str:
        if not self.raw_candidate:
            return "No fingerspelled candidate yet."
        if not self.changed:
            return f"Kept '{self.final_suggestion or self.raw_candidate}' as-is. {self.reason}"
        return (
            f"Changed '{self.raw_candidate}' to '{self.final_suggestion}'. "
            f"{self.reason}"
        )


@dataclass
class EvaluationMetrics:
    """Tracks one correction run for the performance panel and CSV log."""
    scenario_type: str = ""
    raw_input: str = ""
    fuzzy_output: str = ""
    gemma_output: str = ""
    openai_output: str = ""
    final_output: str = ""
    correction_source: str = "Local"
    mode: str = "letter"
    transcript_snapshot: str = ""
    fuzzy_time_ms: float = 0.0
    gemma_time_ms: float = 0.0
    openai_time_ms: float = 0.0
    total_time_ms: float = 0.0
    changed: bool = False
    created_at: float = 0.0
    openai_available: bool = False
    openai_attempted: bool = False
    openai_status: str = ""

    def signature(self) -> str:
        return "|".join([
            self.scenario_type,
            self.raw_input,
            self.final_output,
            self.correction_source,
            self.mode,
        ])


def infer_scenario_type(raw_input: str, final_output: str) -> str:
    raw = normalize_token_for_match(raw_input)
    final = normalize_token_for_match(final_output)
    if not raw:
        return "None"
    edits = levenshtein_distance(raw, final) if final else 0
    if len(raw) <= 6 and edits <= 2 and " " not in (final_output or ""):
        return "Simple"
    return "Complex"


class PerformanceTracker:
    """Collects finalized correction runs and appends them to a CSV log."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.records: list[EvaluationMetrics] = []
        self.latest: Optional[EvaluationMetrics] = None
        self._active_metrics: Optional[EvaluationMetrics] = None
        self._last_logged_signature = ""
        self._last_log_message = ""
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "scenario_type",
                "mode",
                "raw_input",
                "fuzzy_output",
                "gemma_output",
                "openai_output",
                "final_output",
                "correction_source",
                "changed",
                "fuzzy_time_ms",
                "gemma_time_ms",
                "openai_time_ms",
                "total_time_ms",
                "openai_available",
                "openai_attempted",
                "openai_status",
                "transcript_snapshot",
            ])

    def _append_csv(self, metrics: EvaluationMetrics) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                format_csv_timestamp(metrics.created_at),
                metrics.scenario_type,
                metrics.mode,
                metrics.raw_input,
                metrics.fuzzy_output,
                metrics.gemma_output,
                metrics.openai_output,
                metrics.final_output,
                metrics.correction_source,
                int(metrics.changed),
                f"{metrics.fuzzy_time_ms:.3f}",
                f"{metrics.gemma_time_ms:.3f}",
                f"{metrics.openai_time_ms:.3f}",
                f"{metrics.total_time_ms:.3f}",
                int(metrics.openai_available),
                int(metrics.openai_attempted),
                sanitize_text_for_display(metrics.openai_status, max_len=180),
                metrics.transcript_snapshot,
            ])

    def finalize_candidate_if_ready(self, candidate: str, metrics: EvaluationMetrics, transcript_snapshot: str) -> None:
        candidate = sanitize_token_text(candidate)
        metrics.transcript_snapshot = sanitize_text_for_display(transcript_snapshot, max_len=180)
        self.latest = metrics

        if candidate:
            self._active_metrics = metrics
            return

        if self._active_metrics is not None:
            self._active_metrics.transcript_snapshot = sanitize_text_for_display(transcript_snapshot, max_len=180)
            self.latest = self._active_metrics

    def force_finalize(self) -> None:
        if self._active_metrics is None:
            return
        sig = self._active_metrics.signature()
        if sig and sig != self._last_logged_signature and self._active_metrics.raw_input:
            self.records.append(self._active_metrics)
            self._append_csv(self._active_metrics)
            self._last_logged_signature = sig
            self._last_log_message = f"Logged: {self._active_metrics.raw_input} -> {self._active_metrics.final_output}"
        self.latest = self._active_metrics
        self._active_metrics = None

    def count(self) -> int:
        return len(self.records)

    def avg_total_ms(self) -> float:
        if not self.records:
            return 0.0
        return float(sum(r.total_time_ms for r in self.records) / len(self.records))

    def avg_fuzzy_ms(self) -> float:
        if not self.records:
            return 0.0
        return float(sum(r.fuzzy_time_ms for r in self.records) / len(self.records))

    def avg_gemma_ms(self) -> float:
        if not self.records:
            return 0.0
        return float(sum(r.gemma_time_ms for r in self.records) / len(self.records))

    def avg_openai_ms(self) -> float:
        if not self.records:
            return 0.0
        return float(sum(r.openai_time_ms for r in self.records) / len(self.records))

    def source_counts(self) -> dict[str, int]:
        counts = {"Fuzzy": 0, "Gemma": 0, "OpenAI": 0, "Local": 0}
        for r in self.records:
            key = r.correction_source if r.correction_source in counts else "Local"
            counts[key] += 1
        return counts


@dataclass
class ContextExplanation:
    """
    Tracks sentence-level explanation text.
    """
    raw_text: str = ""
    corrected_text: str = ""
    reason: str = "No sentence context yet."
    source: str = "Local"

    def short_text(self) -> str:
        return clean_overlay_reason(self.reason)


def describe_confidence(conf: float, threshold: float) -> str:
    if conf < threshold:
        return "below threshold"
    margin = conf - threshold
    if margin >= 0.25:
        return "well above threshold"
    if margin >= 0.10:
        return "comfortably above threshold"
    return "just above threshold"


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute edit distance without external libraries.

    @param a: First string.
    @param b: Second string.
    @return: Levenshtein edit distance.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def summarize_hand_state(fingers_extended: list[str], palm_spread: float, thumb_wrist: float) -> str:
    """
    Convert geometric cues into a readable hand-state summary.
    """
    ext_count = len(fingers_extended)

    if ext_count == 0:
        if thumb_wrist < 0.55:
            return "compact closed fist"
        return "closed hand"
    if ext_count == 1:
        return f"mostly closed hand with {fingers_extended[0]} extended"
    if ext_count == 2:
        return f"partially open hand with {fingers_extended[0]} and {fingers_extended[1]} extended"
    if ext_count >= 3:
        if palm_spread < 0.18:
            return "open hand with fingers held close together"
        return "open hand with visible finger spread"
    return "unclear hand posture"


def label_template_reason(label: str, trace: LetterExplanationTrace) -> str:
    """
    Produce a label-specific explanation from local geometric cues.
    """
    fingers = trace.fingers_extended
    cue = trace.geometric_cues
    confidence_state = describe_confidence(trace.confidence, trace.threshold)

    if label == "A":
        return (
            f"I favored A because the hand looked like a compact fist, the thumb stayed outside the hand, "
            f"and only {len(fingers)} non-thumb fingers appeared extended; confidence was {confidence_state}."
        )
    if label == "B":
        return (
            "I favored B because several fingers appeared extended together and the finger spacing stayed relatively tight, "
            "which matches a flat open-palm shape."
        )
    if label == "=":
        return (
            "I recognized '=' as the sentence-space sign, so I inserted a word break instead of showing the symbol literally."
        )
    if label == " ":
        return (
            "I recognized the trained space sign and treated it as a real separator, so it inserts a word break instead of a visible letter."
        )

    ext_text = ", ".join(fingers) if fingers else "no clearly extended fingers"
    thumb_index = cue.get("thumb_index", 0.0)
    return (
        f"I kept {repr(label)} because the observed finger pattern ({ext_text}) and thumb-index spacing "
        f"({thumb_index:.3f}) matched that class better than the alternatives."
    )


def build_letter_explanation_trace(
    raw_label: str,
    final_label: str,
    conf: float,
    threshold: float,
    lm_norm: np.ndarray
) -> LetterExplanationTrace:
    """
    Build a structured letter explanation trace from normalized landmarks.
    """
    def _is_extended(tip_idx: int, pip_idx: int) -> bool:
        return bool(lm_norm[tip_idx, 1] < lm_norm[pip_idx, 1])

    thumb_tip = lm_norm[4, :2]
    index_tip = lm_norm[8, :2]
    middle_tip = lm_norm[12, :2]
    ring_tip = lm_norm[16, :2]
    pinky_tip = lm_norm[20, :2]
    wrist = lm_norm[0, :2]

    fingers_extended: list[str] = []
    if _is_extended(8, 6):
        fingers_extended.append("index")
    if _is_extended(12, 10):
        fingers_extended.append("middle")
    if _is_extended(16, 14):
        fingers_extended.append("ring")
    if _is_extended(20, 18):
        fingers_extended.append("pinky")

    cue = {
        "thumb_index": float(np.linalg.norm(thumb_tip - index_tip)),
        "index_middle": float(np.linalg.norm(index_tip - middle_tip)),
        "middle_ring": float(np.linalg.norm(middle_tip - ring_tip)),
        "ring_pinky": float(np.linalg.norm(ring_tip - pinky_tip)),
        "thumb_pinky": float(np.linalg.norm(thumb_tip - pinky_tip)),
        "thumb_wrist": float(np.linalg.norm(thumb_tip - wrist)),
        "palm_spread": float(np.linalg.norm(index_tip - pinky_tip)),
    }
    hand_state = summarize_hand_state(fingers_extended, cue["palm_spread"], cue["thumb_wrist"])

    trace = LetterExplanationTrace(
        raw_label=raw_label,
        final_label=final_label,
        confidence=conf,
        threshold=threshold,
        fingers_extended=fingers_extended,
        geometric_cues=cue,
        hand_state=hand_state,
    )
    trace.label_reason = label_template_reason(raw_label, trace)
    return trace


def explain_local_correction(candidate: str, corrected: str, source: str, vocabulary: list[str]) -> CorrectionDecision:
    """
    Explain why a candidate word was kept or corrected.

    @param candidate: Raw fingerspelled candidate.
    @param corrected: Final chosen correction.
    @param source: Local or OpenAI.
    @param vocabulary: Demo vocabulary for nearby-match hints.
    @return: Structured correction decision.
    """
    norm_candidate = normalize_token_for_match(candidate)
    norm_corrected = normalize_token_for_match(corrected) or norm_candidate
    nearby = get_close_matches(norm_candidate, vocabulary, n=3, cutoff=0.0) if norm_candidate else []

    if not norm_candidate:
        return CorrectionDecision(reason="No letters have been committed into a candidate word yet.")

    if norm_corrected == norm_candidate:
        if len(norm_candidate) < 3:
            reason = "The candidate is still too short to justify a stronger correction."
        elif nearby and nearby[0] == norm_candidate:
            reason = "The observed letters already match a known vocabulary word closely enough."
        else:
            reason = "No clearly better alternative scored above the correction threshold."
        return CorrectionDecision(
            raw_candidate=norm_candidate,
            local_suggestion=norm_candidate,
            final_suggestion=norm_candidate,
            source="Local",
            changed=False,
            nearby_matches=nearby,
            reason=reason,
        )

    edit_distance = levenshtein_distance(norm_candidate, norm_corrected)
    matches_text = ""
    if nearby:
        matches_text = " Nearby options: " + ", ".join(nearby[:3]) + "."
    reason = (
        f"The observed letters were closest to '{norm_corrected}' with edit distance {edit_distance}."
        f"{matches_text}"
    )
    return CorrectionDecision(
        raw_candidate=norm_candidate,
        local_suggestion=normalize_token_for_match(nearby[0]) if nearby else norm_corrected,
        final_suggestion=norm_corrected,
        source=source,
        changed=True,
        nearby_matches=nearby,
        reason=reason,
    )


def build_context_fallback(
    transcript_tokens: list[str],
    corrected_text: str,
    correction_decision: CorrectionDecision,
    mode: str
) -> ContextExplanation:
    """
    Create a sentence-level explanation without using the API.
    """
    raw_text = transcript_to_string(transcript_tokens)
    corrected_clean = corrected_text.strip()

    if not raw_text and not corrected_clean:
        return ContextExplanation(
            raw_text=raw_text,
            corrected_text=corrected_clean,
            reason="No sentence context is available yet.",
            source="Local",
        )

    if correction_decision.changed and correction_decision.final_suggestion:
        if raw_text:
            reason = (
                f"I replaced '{correction_decision.raw_candidate}' with '{correction_decision.final_suggestion}' "
                f"because the letter pattern supports that correction better than the original '{raw_text}'."
            )
        else:
            reason = (
                f"I replaced '{correction_decision.raw_candidate}' with '{correction_decision.final_suggestion}' "
                f"because it is the closest plausible word from the observed letter sequence."
            )
        return ContextExplanation(
            raw_text=raw_text,
            corrected_text=corrected_clean or correction_decision.final_suggestion,
            reason=reason,
            source="Local",
        )

    if mode == "word" and raw_text:
        reason = (
            f"I kept the current sentence as '{raw_text}' because the committed word sequence already fits the available context."
        )
    elif raw_text:
        reason = (
            f"I kept '{raw_text}' because no stronger word-level correction was justified yet."
        )
    else:
        reason = "I am still collecting enough letters to form a more reliable sentence-level explanation."

    return ContextExplanation(
        raw_text=raw_text,
        corrected_text=corrected_clean or raw_text,
        reason=reason,
        source="Local",
    )


# ============================================================
# Fuzzy correction helpers
# ============================================================

def normalize_token_for_match(token: str) -> str:
    token = sanitize_text_for_display(token)
    return "".join(ch.lower() for ch in token if ch.isalnum())


def build_demo_vocabulary() -> list[str]:
    return sorted(set([
        "hello", "help", "thanks", "thankyou", "thank", "yes", "no", "please",
        "sorry", "love", "friend", "family", "school", "computer", "water",
        "food", "good", "bad", "name", "where", "what", "who", "why", "how",
        "i", "you", "me", "want", "need", "drink", "eat", "bathroom", "book",
        "work", "home", "house", "office", "teacher", "student", "class",
        "learn", "study", "read", "write", "paper", "phone", "tablet", "laptop",
        "keyboard", "mouse", "screen", "monitor", "internet", "website", "email",
        "message", "text", "chat", "call", "video", "camera", "picture", "photo",
        "music", "movie", "game", "play", "stop", "start", "go", "come", "leave",
        "stay", "wait", "later", "today", "tomorrow", "yesterday", "morning",
        "afternoon", "evening", "night", "now", "soon", "again", "more", "less",
        "big", "small", "fast", "slow", "happy", "sad", "angry", "tired", "sick",
        "fine", "okay", "ok", "great", "awesome", "cool", "hot", "cold", "warm",
        "all", "call", "fall", "hall", "small", "tall", "talk", "tell", "wall",
        "hungry", "thirsty", "sleep", "bed", "chair", "table", "door", "window",
        "car", "bus", "train", "plane", "road", "street", "city", "state", "country",
        "left", "right", "up", "down", "open", "close", "inside", "outside",
        "before", "after", "first", "last", "next", "question", "answer",
        "problem", "solution", "project", "assignment", "meeting", "team",
        "group", "doctor", "nurse", "hospital", "medicine", "pain", "head",
        "hand", "face", "eye", "ear", "mouth", "banana", "apple", "orange",
        "grape", "rice", "bread", "milk", "coffee", "tea", "soda", "juice",
        "brother", "sister", "mother", "father", "baby", "child", "children",
        "man", "woman", "person", "people", "everyone", "someone", "nobody",
        "because", "understand", "understood", "understanding", "autocorrect",
        "correct", "correction", "recognize", "recognition", "sign", "signing",
        "letter", "letters", "word", "words", "sentence", "sentences"
    ]))


def repeated_letter_variants(token: str) -> list[str]:
    norm = normalize_token_for_match(token)
    if not norm:
        return []

    groups: list[tuple[str, int]] = []
    cur = norm[0]
    count = 1
    for ch in norm[1:]:
        if ch == cur:
            count += 1
        else:
            groups.append((cur, count))
            cur = ch
            count = 1
    groups.append((cur, count))

    variants = {norm}

    def _build(i: int, prefix: str) -> None:
        if i >= len(groups):
            variants.add(prefix)
            return
        ch, cnt = groups[i]
        max_keep = min(cnt, 2)
        for keep in range(1, max_keep + 1):
            _build(i + 1, prefix + (ch * keep))

    _build(0, "")
    return sorted(variants, key=lambda s: (abs(len(s) - len(norm)), s != norm, s))


def best_supported_vocabulary_match(
    token: str,
    vocabulary: list[str],
    cutoff: float,
    accept_close_cutoff: float,
    accept_max_len_delta: int,
    accept_max_edit_ratio: float,
    accept_min_shared_chars: int,
) -> str:
    norm = normalize_token_for_match(token)
    if not norm:
        return token

    variants = repeated_letter_variants(norm)
    vocab_set = set(vocabulary)

    for variant in variants:
        if variant in vocab_set:
            return variant

    candidate_pool: list[str] = []
    for variant in variants:
        for match in get_close_matches(variant, vocabulary, n=5, cutoff=cutoff):
            if match not in candidate_pool:
                candidate_pool.append(match)

    if not candidate_pool:
        return token

    best = token
    best_score: tuple[float, float, float] | None = None
    for match in candidate_pool:
        for variant in variants:
            if plausible_word_correction(
                variant,
                match,
                vocabulary,
                accept_close_cutoff=accept_close_cutoff,
                accept_max_len_delta=accept_max_len_delta,
                accept_max_edit_ratio=accept_max_edit_ratio,
                accept_min_shared_chars=accept_min_shared_chars,
            ):
                shared = shared_character_count(variant, match)
                edit = levenshtein_distance(variant, match)
                score = (
                    shared / max(1, len(variant)),
                    -edit / max(1, len(variant)),
                    -abs(len(variant) - len(match)),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = match
    return best


def fuzzy_correct_token(token: str, vocabulary: list[str], cutoff: float = 0.72, cfg: Optional[DemoConfig] = None) -> str:
    norm = normalize_token_for_match(token)
    if not norm:
        return token

    if cfg is None:
        return best_supported_vocabulary_match(
            norm,
            vocabulary,
            cutoff=cutoff,
            accept_close_cutoff=0.84,
            accept_max_len_delta=1,
            accept_max_edit_ratio=0.34,
            accept_min_shared_chars=2,
        )

    return best_supported_vocabulary_match(
        norm,
        vocabulary,
        cutoff=cutoff,
        accept_close_cutoff=float(cfg.fuzzy_accept_close_cutoff),
        accept_max_len_delta=int(cfg.fuzzy_accept_max_len_delta),
        accept_max_edit_ratio=float(cfg.fuzzy_accept_max_edit_ratio),
        accept_min_shared_chars=int(cfg.fuzzy_accept_min_shared_chars),
    )


def shared_character_count(a: str, b: str) -> int:
    return len(set(a) & set(b))


def plausible_word_correction(
    candidate: str,
    suggestion: str,
    vocabulary: list[str],
    accept_close_cutoff: float,
    accept_max_len_delta: int,
    accept_max_edit_ratio: float,
    accept_min_shared_chars: int,
) -> bool:
    cand = normalize_token_for_match(candidate)
    sugg = normalize_token_for_match(suggestion)

    if not cand or not sugg or sugg == cand:
        return False
    if not sugg.isalpha():
        return False
    if abs(len(cand) - len(sugg)) > int(accept_max_len_delta):
        return False

    shared = shared_character_count(cand, sugg)
    if shared < min(len(cand), int(accept_min_shared_chars)):
        return False

    max_edit = max(1, int(round(len(cand) * float(accept_max_edit_ratio))))
    edit = levenshtein_distance(cand, sugg)
    if edit > max_edit:
        return False

    top_close = get_close_matches(
        cand,
        vocabulary,
        n=8,
        cutoff=float(accept_close_cutoff),
    )
    if sugg in top_close:
        return True

    return edit <= max_edit and shared >= min(len(cand), int(accept_min_shared_chars))


def plausible_ai_correction(
    candidate: str,
    suggestion: str,
    vocabulary: list[str],
    cfg: DemoConfig,
) -> bool:
    variants = repeated_letter_variants(candidate)
    for variant in variants:
        if plausible_word_correction(
            variant,
            suggestion,
            vocabulary,
            accept_close_cutoff=float(cfg.ai_accept_close_cutoff),
            accept_max_len_delta=int(cfg.ai_accept_max_len_delta),
            accept_max_edit_ratio=float(cfg.ai_accept_max_edit_ratio),
            accept_min_shared_chars=int(cfg.ai_accept_min_shared_chars),
        ):
            return True
    return False


def tokens_to_fingerspelled_candidate(tokens: list[str], max_len: int) -> str:
    trailing_letters: list[str] = []
    for tok in reversed(tokens):
        if tok == " ":
            break
        if len(tok) == 1 and tok.isalpha():
            trailing_letters.append(tok.lower())
            continue
        break
    if not trailing_letters:
        return ""
    trailing_letters.reverse()
    return "".join(trailing_letters[-max_len:])


def corrected_transcript_view(
    tokens: list[str],
    vocabulary: list[str],
    cfg: DemoConfig,
    mode: str
) -> list[str]:
    if not cfg.fuzzy_correction_enabled:
        return tokens

    if mode == "letter":
        corrected_tokens: list[str] = []
        word_buf: list[str] = []

        def flush_word() -> None:
            nonlocal word_buf, corrected_tokens
            if not word_buf:
                return
            candidate = "".join(word_buf[-cfg.fuzzy_letter_commit_len:]).lower()
            if len(candidate) >= cfg.fuzzy_min_word_len:
                corrected = fuzzy_correct_token(candidate, vocabulary, cutoff=cfg.fuzzy_match_cutoff, cfg=cfg)
                corrected_tokens.append(corrected)
            else:
                corrected_tokens.extend(word_buf)
            word_buf = []

        for tok in tokens:
            if tok == " ":
                flush_word()
                if corrected_tokens and corrected_tokens[-1] != " ":
                    corrected_tokens.append(" ")
                continue
            if len(tok) == 1 and tok.isalpha():
                word_buf.append(tok)
                continue
            flush_word()
            corrected_tokens.append(tok)

        flush_word()
        return corrected_tokens

    corrected_tokens: list[str] = []
    for tok in tokens:
        if tok == " ":
            corrected_tokens.append(tok)
            continue

        norm = normalize_token_for_match(tok)
        if len(norm) >= cfg.fuzzy_min_word_len:
            corrected_tokens.append(
                fuzzy_correct_token(tok, vocabulary, cutoff=cfg.fuzzy_match_cutoff, cfg=cfg)
            )
        else:
            corrected_tokens.append(tok)

    return corrected_tokens


# ============================================================
# Text backends
# ============================================================

class LocalGemmaClient:
    """
    Local Gemma text backend using Transformers.
    """

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.label = "Local Gemma"
        self.available = False
        self.status = "Local Gemma disabled"
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_path = ""

        if not cfg.ai_context_enabled or not cfg.local_llm_enabled:
            self.status = "Local Gemma disabled"
            return

        try:
            model_path = self._resolve_model_path()
            self._load_model(model_path)
            self.model_path = model_path
            self.available = True
            self.status = f"Local Gemma ready ({model_path})"
            print(f"[Local Gemma Debug] Loaded model from: {model_path}")
        except Exception as e:
            self.available = False
            self.status = f"Local Gemma unavailable: {e}"
            print(f"[Local Gemma Debug] Failed to initialize local backend: {e}")

    def _resolve_model_path(self) -> str:
        if self.cfg.local_llm_model_path:
            candidate = self.cfg.local_llm_model_path.strip()
            if os.path.exists(candidate):
                return candidate

        ref = self.cfg.local_llm_model_ref.strip()
        if not ref:
            raise RuntimeError("No local_llm_model_ref configured.")

        if os.path.exists(ref):
            return ref

        if self.cfg.local_llm_download_via.lower() == "kagglehub":
            try:
                import kagglehub
            except Exception as e:
                raise RuntimeError(f"kagglehub import failed: {e}")
            try:
                return kagglehub.model_download(ref)
            except Exception as e:
                raise RuntimeError(f"kagglehub download failed: {e}")

        return ref

    def _choose_torch_dtype(self, torch_module: Any):
        dtype_name = (self.cfg.local_llm_dtype or "auto").lower()
        if dtype_name == "float16":
            return torch_module.float16
        if dtype_name == "bfloat16":
            return getattr(torch_module, "bfloat16", torch_module.float32)
        if dtype_name == "float32":
            return torch_module.float32
        if torch_module.cuda.is_available():
            return torch_module.float16
        return torch_module.float32

    def _load_model(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(f"transformers/torch import failed: {e}")

        torch_dtype = self._choose_torch_dtype(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.cfg.local_llm_trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.cfg.local_llm_trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=self.cfg.local_llm_device_map,
        )

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self.tokenizer.pad_token_id = eos_id

        self.device = str(getattr(self.model, "device", "cpu"))

    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if not self.available or self.model is None or self.tokenizer is None:
            return ""

        import torch

        max_tokens = int(max_new_tokens or self.cfg.local_llm_max_new_tokens)
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                model_inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
                prompt_len = int(model_inputs["input_ids"].shape[-1])
            else:
                model_inputs = self.tokenizer(prompt, return_tensors="pt")
                model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
                prompt_len = int(model_inputs["input_ids"].shape[-1])

            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=bool(self.cfg.local_llm_do_sample),
                    temperature=float(self.cfg.local_llm_temperature),
                    top_p=float(self.cfg.local_llm_top_p),
                    repetition_penalty=float(self.cfg.local_llm_repetition_penalty),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )

            generated = outputs[0][prompt_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return str(text).strip()
        except Exception as e:
            self.status = f"Local Gemma request failed: {e}"
            print(f"[Local Gemma Debug] Request failed: {e}")
            return ""


class OpenAITextClient:
    """
    OpenAI text backend wrapper used as optional fallback.
    """

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.label = "OpenAI"
        self.available = False
        self.status = "OpenAI fallback disabled"
        self.client = None

        if not cfg.ai_context_enabled or not cfg.openai_fallback_enabled:
            self.status = "OpenAI fallback disabled"
            return

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            self.status = "OPENAI_API_KEY not found"
            return

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
            self.status = "OpenAI fallback ready"
            print("[OpenAI Debug] OpenAI client initialized for fallback.")
        except Exception as e:
            self.available = False
            self.status = f"OpenAI init failed: {e}"
            print(f"[OpenAI Debug] Failed to initialize OpenAI client: {e}")

    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if not self.available or self.client is None:
            return ""
        try:
            response = self.client.responses.create(
                model=self.cfg.ai_context_model,
                input=prompt,
            )
            return str((response.output_text or "").strip())
        except Exception as e:
            self.status = f"OpenAI request failed: {e}"
            print(f"[OpenAI Debug] Request failed: {e}")
            return ""


# ============================================================
# Hybrid autocorrect
# ============================================================

class HybridAutoCorrector:
    """
    Local fuzzy autocorrect first.
    If local correction does not improve the candidate, try Local Gemma first
    and then optionally use OpenAI as a fallback backend.
    """

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.vocabulary = build_demo_vocabulary()

        self.local_client = LocalGemmaClient(cfg)
        self.openai_client = OpenAITextClient(cfg)
        self.client = self._pick_explainer_client()

        self.debug_source = "Local"
        self.debug_status = "Local autocorrect only"
        self.last_display_text = ""
        self.last_candidate = ""
        self.last_local_suggestion = ""
        self.last_ai_suggestion = ""
        self.last_decision = CorrectionDecision()
        self.last_metrics = EvaluationMetrics()
        self._last_ai_trace = {"gemma_output": "", "gemma_time_ms": 0.0, "openai_output": "", "openai_time_ms": 0.0, "openai_attempted": False, "openai_status": ""}

        self._printed_ready_once = False
        self._printed_error_once = False

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._pending_candidate: str = ""
        self._pending_backend_label: str = ""

        self._disable_local_backend = not self.local_client.available
        self._disable_openai_backend = not self.openai_client.available

        if not cfg.ai_context_enabled:
            self.debug_status = "AI fallback disabled"
        elif self.local_client.available:
            self.debug_status = "Local Gemma fallback ready"
        elif self.openai_client.available:
            self.debug_status = "OpenAI fallback ready"
        else:
            self.debug_status = "No AI backend available; local fuzzy only"

    def _pick_explainer_client(self) -> Optional[Any]:
        backend = (self.cfg.ai_backend or "local_gemma").lower()
        if backend == "openai" and self.openai_client.available:
            return self.openai_client
        if backend in {"local_gemma", "auto"} and self.local_client.available:
            return self.local_client
        if self.local_client.available:
            return self.local_client
        if self.openai_client.available:
            return self.openai_client
        return None

    def local_correct(self, candidate: str) -> str:
        if len(candidate) < self.cfg.fuzzy_min_word_len:
            return candidate
        return fuzzy_correct_token(candidate, self.vocabulary, cutoff=self.cfg.fuzzy_match_cutoff, cfg=self.cfg)

    def should_try_ai(self, candidate: str, local_suggestion: str) -> bool:
        if not self.cfg.ai_context_enabled:
            return False
        if len(candidate) < self.cfg.ai_context_min_tokens:
            return False
        if local_suggestion != candidate:
            return False
        if self.local_client.available and not self._disable_local_backend:
            return True
        if self.openai_client.available and not self._disable_openai_backend:
            return True
        return False

    def _candidate_shortlist(self, candidate: str) -> list[str]:
        cand = normalize_token_for_match(candidate)
        if not cand:
            return []
        shortlist: list[str] = []
        for variant in repeated_letter_variants(cand):
            if variant in self.vocabulary and variant not in shortlist:
                shortlist.append(variant)
            for match in get_close_matches(
                variant,
                self.vocabulary,
                n=max(1, int(self.cfg.ai_candidate_shortlist_size)),
                cutoff=float(self.cfg.ai_candidate_shortlist_cutoff),
            ):
                if match not in shortlist:
                    shortlist.append(match)
                if len(shortlist) >= int(self.cfg.ai_candidate_shortlist_size):
                    return shortlist
        return shortlist

    def _build_autocorrect_prompt(self, candidate: str) -> str:
        shortlist = self._candidate_shortlist(candidate)
        shortlist_text = ", ".join(shortlist) if shortlist else "(none)"
        return (
            "You are validating a noisy ASL fingerspelled word. "
            "Be conservative and avoid false positives. "
            "Output exactly one lowercase ASCII word only. "
            "Only change the input when the correction is very close to the letters. "
            "Do not guess a short unrelated common word. "
            "If uncertain, return the original letters unchanged. "
            "Prefer one of the shortlist words only when it is clearly supported by the letters. "
            "Examples: hllo -> hello, frend -> friend, watr -> water, xqpt -> xqpt, cfwh -> cfwh. "
            f"Input letters: {candidate}. "
            f"Closest shortlist words: {shortlist_text}."
        )

    def _is_supported_correction(self, candidate: str, suggestion: str) -> bool:
        return plausible_ai_correction(candidate, suggestion, self.vocabulary, self.cfg)

    def _query_backend(self, backend: Any, candidate: str) -> str:
        if backend is None:
            return candidate
        prompt = self._build_autocorrect_prompt(candidate)
        raw_text = backend.generate_text(prompt, max_new_tokens=self.cfg.local_llm_max_new_tokens)
        best = extract_best_candidate_word(str(raw_text or ""), candidate, self.vocabulary)
        cleaned = sanitize_token_text(best, fallback=candidate)
        return cleaned or candidate

    def _ai_correct_sync(self, candidate: str) -> Tuple[str, str, dict[str, float | str]]:
        backend_order: list[tuple[str, Any]] = []
        preferred = (self.cfg.ai_backend or "local_gemma").lower()
        trace: dict[str, float | str | bool] = {
            "gemma_output": "",
            "gemma_time_ms": 0.0,
            "openai_output": "",
            "openai_time_ms": 0.0,
            "openai_attempted": False,
            "openai_status": sanitize_text_for_display(self.openai_client.status, max_len=120),
        }

        if preferred == "openai":
            backend_order = [
                ("OpenAI", None if self._disable_openai_backend else self.openai_client),
                ("Local Gemma", None if self._disable_local_backend else self.local_client),
            ]
        else:
            backend_order = [
                ("Local Gemma", None if self._disable_local_backend else self.local_client),
                ("OpenAI", None if self._disable_openai_backend else self.openai_client),
            ]

        for backend_label, backend in backend_order:
            if backend is None or not getattr(backend, "available", False):
                continue
            try:
                print(f"[{backend_label} Debug] Sending hybrid autocorrect request in background...")
                req_start = time.time()
                text = self._query_backend(backend, candidate)
                elapsed_ms = (time.time() - req_start) * 1000.0
                text = sanitize_token_text(text, fallback=candidate)
                if backend_label == "Local Gemma":
                    trace["gemma_output"] = text
                    trace["gemma_time_ms"] = elapsed_ms
                elif backend_label == "OpenAI":
                    trace["openai_attempted"] = True
                    trace["openai_output"] = text
                    trace["openai_time_ms"] = elapsed_ms
                    trace["openai_status"] = sanitize_text_for_display(self.openai_client.status, max_len=120) or "OpenAI response received"
                if not text:
                    continue

                if text != candidate and not self._is_supported_correction(candidate, text):
                    print(f"[{backend_label} Debug] Rejected weak correction: {candidate} -> {text}")
                    if backend_label == "Local Gemma" and self.openai_client.available and not self._disable_openai_backend:
                        self.debug_status = "Local Gemma suggestion rejected; trying OpenAI"
                        continue
                    self.debug_status = f"{backend_label} suggestion rejected; keeping original"
                    continue

                if text == candidate:
                    if backend_label == "Local Gemma" and self.openai_client.available and not self._disable_openai_backend:
                        self.debug_status = "Local Gemma kept original; trying OpenAI"
                        print("[Local Gemma Debug] No better correction found. Trying OpenAI fallback...")
                        continue
                    if backend_label == "OpenAI":
                        self.debug_status = "OpenAI kept original; using original"
                        print("[OpenAI Debug] No better correction found. Keeping original candidate.")

                self.debug_source = backend_label if text != candidate else "Local"
                self.debug_status = f"{backend_label} correction accepted" if text != candidate else self.debug_status
                return text, backend_label if text != candidate else "Local", trace
            except Exception as e:
                if backend_label == "Local Gemma":
                    self._disable_local_backend = True
                    self.debug_status = f"{backend_label} failed; trying fallback"
                else:
                    trace["openai_attempted"] = True
                    trace["openai_status"] = sanitize_text_for_display(f"OpenAI request failed: {e}", max_len=120)
                    self._disable_openai_backend = True
                    self.debug_status = f"{backend_label} failed; local only"
                print(f"[{backend_label} Debug] Hybrid autocorrect request failed: {e}")

        self.debug_source = "Local"
        self.debug_status = "No AI backend improved the candidate"
        return candidate, "Local", trace

    def request_ai_if_needed(self, candidate: str, local_suggestion: str) -> None:
        if not self.should_try_ai(candidate, local_suggestion):
            return
        if candidate == self._pending_candidate:
            return
        if self._future is not None and not self._future.done():
            return

        self._pending_candidate = candidate
        self._pending_backend_label = ""
        self.debug_source = "Local"
        self.debug_status = "Local suggestion kept; backend fallback running"
        self._future = self._executor.submit(self._ai_correct_sync, candidate)

    def poll_ai_result(self) -> None:
        if self._future is None or not self._future.done():
            return

        try:
            result, backend_label, trace = self._future.result()
            result = sanitize_token_text(result, fallback=self._pending_candidate)
            self.last_ai_suggestion = result if result else self._pending_candidate
            self._pending_backend_label = backend_label
            self._last_ai_trace = {
                "gemma_output": str(trace.get("gemma_output", "") or ""),
                "gemma_time_ms": float(trace.get("gemma_time_ms", 0.0) or 0.0),
                "openai_output": str(trace.get("openai_output", "") or ""),
                "openai_time_ms": float(trace.get("openai_time_ms", 0.0) or 0.0),
                "openai_attempted": bool(trace.get("openai_attempted", False)),
                "openai_status": str(trace.get("openai_status", "") or ""),
            }
        except Exception:
            self.last_ai_suggestion = self._pending_candidate
            self._pending_backend_label = "Local"
            self._last_ai_trace = {"gemma_output": "", "gemma_time_ms": 0.0, "openai_output": "", "openai_time_ms": 0.0, "openai_attempted": False, "openai_status": ""}
        finally:
            self._future = None

    def update(self, candidate: str, mode: str = "letter", transcript_snapshot: str = "") -> str:
        candidate = sanitize_token_text(candidate)
        self.last_candidate = candidate
        eval_start = time.time()

        if not candidate:
            self.last_local_suggestion = ""
            self.last_ai_suggestion = ""
            self.last_display_text = ""
            self.debug_source = "Local"
            self.debug_status = "No letters yet"
            self.last_decision = explain_local_correction("", "", "Local", self.vocabulary)
            self.last_metrics = EvaluationMetrics(mode=mode, transcript_snapshot=sanitize_text_for_display(transcript_snapshot, max_len=180), created_at=time.time(), openai_available=bool(self.openai_client.available), openai_status=sanitize_text_for_display(self.openai_client.status, max_len=120))
            return ""

        fuzzy_start = time.time()
        local_suggestion = normalize_token_for_match(self.local_correct(candidate))
        fuzzy_time_ms = (time.time() - fuzzy_start) * 1000.0
        self.last_local_suggestion = local_suggestion

        base_metrics = EvaluationMetrics(
            raw_input=candidate,
            fuzzy_output=local_suggestion or candidate,
            mode=mode,
            transcript_snapshot=sanitize_text_for_display(transcript_snapshot, max_len=180),
            fuzzy_time_ms=fuzzy_time_ms,
            created_at=time.time(),
            openai_available=bool(self.openai_client.available),
            openai_status=sanitize_text_for_display(self.openai_client.status, max_len=120),
        )

        if local_suggestion != candidate:
            self.last_display_text = local_suggestion
            self.debug_source = "Local"
            self.debug_status = "Local fuzzy correction used"
            self.last_ai_suggestion = ""
            self._last_ai_trace = {"gemma_output": "", "gemma_time_ms": 0.0, "openai_output": "", "openai_time_ms": 0.0, "openai_attempted": False, "openai_status": ""}
            self.last_decision = explain_local_correction(
                candidate,
                local_suggestion,
                "Local",
                self.vocabulary,
            )
            base_metrics.final_output = local_suggestion
            base_metrics.correction_source = "Fuzzy"
            base_metrics.changed = True
            base_metrics.scenario_type = infer_scenario_type(candidate, local_suggestion)
            base_metrics.openai_attempted = False
            base_metrics.openai_status = sanitize_text_for_display(self.openai_client.status, max_len=120)
            base_metrics.total_time_ms = compute_total_time_ms(base_metrics.fuzzy_time_ms, base_metrics.gemma_time_ms, base_metrics.openai_time_ms)
            self.last_metrics = base_metrics
            return self.last_display_text

        self.request_ai_if_needed(candidate, local_suggestion)
        self.poll_ai_result()

        if self.last_ai_suggestion and self.last_ai_suggestion != candidate:
            backend_label = self._pending_backend_label or self.debug_source or "AI"
            self.last_display_text = self.last_ai_suggestion
            self.debug_source = backend_label
            self.debug_status = f"{backend_label} correction used"
            self.last_decision = explain_local_correction(
                candidate,
                self.last_ai_suggestion,
                backend_label,
                self.vocabulary,
            )
            base_metrics.gemma_output = sanitize_token_text(str(self._last_ai_trace.get("gemma_output", "") or ""))
            base_metrics.gemma_time_ms = float(self._last_ai_trace.get("gemma_time_ms", 0.0) or 0.0)
            base_metrics.openai_output = sanitize_token_text(str(self._last_ai_trace.get("openai_output", "") or ""))
            base_metrics.openai_time_ms = float(self._last_ai_trace.get("openai_time_ms", 0.0) or 0.0)
            base_metrics.openai_attempted = bool(self._last_ai_trace.get("openai_attempted", False))
            base_metrics.openai_status = sanitize_text_for_display(str(self._last_ai_trace.get("openai_status", self.openai_client.status) or self.openai_client.status), max_len=120)
            base_metrics.final_output = self.last_ai_suggestion
            base_metrics.correction_source = "Gemma" if backend_label == "Local Gemma" else ("OpenAI" if backend_label == "OpenAI" else "Local")
            base_metrics.changed = True
            base_metrics.scenario_type = infer_scenario_type(candidate, self.last_ai_suggestion)
            base_metrics.total_time_ms = compute_total_time_ms(base_metrics.fuzzy_time_ms, base_metrics.gemma_time_ms, base_metrics.openai_time_ms)
            self.last_metrics = base_metrics
            return self.last_display_text

        self.last_display_text = candidate
        if self.should_try_ai(candidate, local_suggestion):
            self.debug_source = "Local"
            self.debug_status = "No supported correction yet"
        else:
            self.debug_source = "Local"
            self.debug_status = "Local suggestion kept"

        self.last_decision = explain_local_correction(
            candidate,
            candidate,
            self.debug_source,
            self.vocabulary,
        )
        base_metrics.gemma_output = sanitize_token_text(str(self._last_ai_trace.get("gemma_output", "") or ""))
        base_metrics.gemma_time_ms = float(self._last_ai_trace.get("gemma_time_ms", 0.0) or 0.0)
        base_metrics.openai_output = sanitize_token_text(str(self._last_ai_trace.get("openai_output", "") or ""))
        base_metrics.openai_time_ms = float(self._last_ai_trace.get("openai_time_ms", 0.0) or 0.0)
        base_metrics.openai_attempted = bool(self._last_ai_trace.get("openai_attempted", False))
        base_metrics.openai_status = sanitize_text_for_display(str(self._last_ai_trace.get("openai_status", self.openai_client.status) or self.openai_client.status), max_len=120)
        base_metrics.final_output = candidate
        base_metrics.correction_source = "Local"
        base_metrics.changed = False
        base_metrics.scenario_type = infer_scenario_type(candidate, candidate)
        base_metrics.total_time_ms = compute_total_time_ms(base_metrics.fuzzy_time_ms, base_metrics.gemma_time_ms, base_metrics.openai_time_ms)
        self.last_metrics = base_metrics
        return self.last_display_text

    def reset(self) -> None:
        self.debug_source = "Local"
        self.debug_status = "No letters yet"
        self.last_display_text = ""
        self.last_candidate = ""
        self.last_local_suggestion = ""
        self.last_ai_suggestion = ""
        self.last_decision = CorrectionDecision()
        self.last_metrics = EvaluationMetrics()
        self._last_ai_trace = {"gemma_output": "", "gemma_time_ms": 0.0, "openai_output": "", "openai_time_ms": 0.0, "openai_attempted": False, "openai_status": ""}

        self._pending_candidate = ""
        self._pending_backend_label = ""
        if self._future is not None and not self._future.done():
            try:
                self._future.cancel()
            except Exception:
                pass
        self._future = None

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


class RecognitionExplainer:
    """
    Produces sentence-level explanations for why the system kept or corrected text.
    Uses local reasoning first and optionally asks the selected backend in the
    background for a short natural-language explanation.
    """

    def __init__(self, cfg: DemoConfig, client: object = None):
        self.cfg = cfg
        self.client = client
        self.last_context = ContextExplanation()
        self.debug_status = "Local sentence explanation only"
        self._disable_remote_after_error = False
        self._future: Optional[Future] = None
        self._pending_key = ""
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_completed_key = ""

    def _make_key(
        self,
        transcript_tokens: list[str],
        corrected_text: str,
        correction_decision: CorrectionDecision,
        mode: str
    ) -> str:
        return "|".join([
            mode,
            transcript_to_string(transcript_tokens),
            corrected_text.strip(),
            correction_decision.raw_candidate,
            correction_decision.final_suggestion,
            correction_decision.source,
        ])

    def _ai_explain_sync(
        self,
        transcript_tokens: list[str],
        corrected_text: str,
        correction_decision: CorrectionDecision,
        mode: str
    ) -> str:
        if self.client is None or self._disable_remote_after_error:
            return ""

        raw_text = transcript_to_string(transcript_tokens)
        prompt = (
            "Explain this ASL demo result in one very short sentence. "
            "Be concise. Mention the observed text and, if changed, the final corrected word or phrase. "
            "Do not use bullets, JSON, or extra filler. "
            f"Mode: {mode}. "
            f"Raw transcript: {raw_text or '(none)'}. "
            f"Candidate: {correction_decision.raw_candidate or '(none)'}. "
            f"Final: {correction_decision.final_suggestion or correction_decision.raw_candidate or '(none)'}. "
            f"Source: {correction_decision.source}. "
            f"Corrected transcript: {corrected_text or raw_text or '(none)'}."
        )

        try:
            text = self.client.generate_text(prompt, max_new_tokens=self.cfg.local_llm_explain_max_new_tokens)
            return sanitize_text_for_display(" ".join(str(text or "").strip().split()), max_len=260)
        except Exception:
            self._disable_remote_after_error = True
            return ""

    def update(
        self,
        transcript_tokens: list[str],
        corrected_text: str,
        correction_decision: CorrectionDecision,
        mode: str
    ) -> ContextExplanation:
        local_context = build_context_fallback(
            transcript_tokens,
            corrected_text,
            correction_decision,
            mode,
        )

        key = self._make_key(transcript_tokens, corrected_text, correction_decision, mode)
        remote_allowed = (
            self.client is not None
            and not self._disable_remote_after_error
            and correction_decision.source in {"OpenAI"}
            and correction_decision.changed
            and bool(corrected_text or transcript_to_string(transcript_tokens))
        )

        if not (self.last_context.source in {"OpenAI"} and self._last_completed_key == key):
            self.last_context = local_context

        if remote_allowed and key != self._pending_key and key != self._last_completed_key:
            if self._future is None or self._future.done():
                self._pending_key = key
                self.debug_status = "Sentence explanation running"
                self._future = self._executor.submit(
                    self._ai_explain_sync,
                    list(transcript_tokens),
                    corrected_text,
                    correction_decision,
                    mode,
                )

        if self._future is not None and self._future.done():
            try:
                ai_reason = " ".join(str(self._future.result() or "").split())
                if ai_reason:
                    backend_label = getattr(self.client, "label", "AI")
                    self.last_context = ContextExplanation(
                        raw_text=local_context.raw_text,
                        corrected_text=local_context.corrected_text,
                        reason=ai_reason,
                        source=backend_label,
                    )
                    self._last_completed_key = self._pending_key
                    self.debug_status = f"{backend_label} sentence explanation used"
                else:
                    if not (self.last_context.source in {"OpenAI"} and self._last_completed_key == key):
                        self.last_context = local_context
                    self.debug_status = "Local sentence explanation used"
            except Exception:
                if not (self.last_context.source in {"OpenAI"} and self._last_completed_key == key):
                    self.last_context = local_context
                self.debug_status = "Local sentence explanation used"
            finally:
                self._future = None

        if self.last_context.source in {"OpenAI"} and self._last_completed_key == key:
            self.debug_status = f"{self.last_context.source} sentence explanation used"
        elif remote_allowed:
            self.debug_status = "Sentence explanation running" if self._future is not None else "Local sentence explanation used"
        else:
            self.debug_status = "Local sentence explanation used"

        return self.last_context

    def reset(self) -> None:
        self.last_context = ContextExplanation()
        self.debug_status = "Local sentence explanation only"
        self._pending_key = ""
        self._last_completed_key = ""
        if self._future is not None and not self._future.done():
            try:
                self._future.cancel()
            except Exception:
                pass
        self._future = None

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ============================================================
# Fast letter stabilizer
# ============================================================

class LetterStabilizer:
    """
    Fast temporal stabilizer for letter mode.

    Behavior:
    - A label must appear for stable_frames_required consecutive frames
      to become stable.
    - Once committed, the same label will not be committed again immediately.
    - The same label can be committed again only after:
      1) it disappears for release_frames_required frames, or
      2) the repeat cooldown expires.
    """

    def __init__(
        self,
        stable_frames_required: int = 2,
        release_frames_required: int = 3,
    ) -> None:
        self.stable_frames_required = max(1, int(stable_frames_required))
        self.release_frames_required = max(1, int(release_frames_required))

        self.candidate = "Unknown"
        self.candidate_count = 0

        self.stable_label = "Unknown"
        self.last_committed_label = ""
        self.release_count = 0

    def update(self, label: str) -> str:
        """
        Update the stabilizer with the latest raw label.

        @param label: Current frame label.
        @return: Stable label or "Unknown".
        """
        if label == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = label
            self.candidate_count = 1

        if label == "Unknown":
            self.release_count += 1
            if self.release_count >= self.release_frames_required:
                self.stable_label = "Unknown"
            return self.stable_label

        self.release_count = 0

        if self.candidate_count >= self.stable_frames_required:
            self.stable_label = self.candidate

        return self.stable_label

    def mark_committed(self, label: str) -> None:
        """
        Record the most recently committed label.
        """
        self.last_committed_label = label

    def released_since_commit(self) -> bool:
        """
        Whether the sign has been released long enough since the last commit.
        """
        return self.release_count >= self.release_frames_required


# ============================================================
# Sklearn Model Loader
# ============================================================

class SklearnClassifier:
    """
    Loads a scikit-learn model and optional scaler/label encoder.
    """

    def __init__(self, model_path: str, scaler_path: str = "", label_encoder_path: str = ""):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._label_map = None

    def load(self) -> bool:
        try:
            import joblib
        except Exception:
            return False

        if not os.path.exists(self.model_path):
            return False

        try:
            self.model = joblib.load(self.model_path)
        except Exception:
            self.model = None
            return False

        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
            except Exception:
                self.scaler = None

        if (
            self.label_encoder_path
            and not self.label_encoder_path.lower().endswith(".json")
            and os.path.exists(self.label_encoder_path)
        ):
            try:
                self.label_encoder = joblib.load(self.label_encoder_path)
            except Exception:
                self.label_encoder = None

        if (
            self.label_encoder_path
            and self.label_encoder_path.lower().endswith(".json")
            and os.path.exists(self.label_encoder_path)
        ):
            try:
                import json as _json
                with open(self.label_encoder_path, "r", encoding="utf-8") as f:
                    self._label_map = _json.load(f)
            except Exception:
                self._label_map = None

        return True

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        if self.model is None:
            return "Unknown", 0.0

        x = features.astype(np.float32).reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            pred = self.model.classes_[idx]
        else:
            pred = self.model.predict(x)[0]
            conf = 1.0

        if self.label_encoder is not None:
            try:
                pred = self.label_encoder.inverse_transform([int(pred)])[0]
            except Exception:
                pass

        if self._label_map is not None:
            try:
                if isinstance(self._label_map, dict) and "labels" in self._label_map:
                    labs = self._label_map["labels"]
                    if isinstance(pred, (int, np.integer)) and 0 <= int(pred) < len(labs):
                        pred = labs[int(pred)]
                    elif isinstance(pred, str) and pred.isdigit() and 0 <= int(pred) < len(labs):
                        pred = labs[int(pred)]
                elif isinstance(self._label_map, list):
                    labs = self._label_map
                    if isinstance(pred, (int, np.integer)) and 0 <= int(pred) < len(labs):
                        pred = labs[int(pred)]
            except Exception:
                pass

        return str(pred), conf


# ============================================================
# Stub Classifiers
# ============================================================

def classify_letter_stub(features: np.ndarray) -> Tuple[str, float]:
    v = float(np.clip(np.std(features) * 2.5, 0.0, 1.0))
    labels = ["A", "T", "M", "N", "S", "=", " "]
    idx = int((np.mean(features) * 1000) % len(labels))
    return labels[idx], v


def classify_word_stub(word_features: np.ndarray) -> Tuple[str, float]:
    conf = float(np.clip(np.std(word_features) * 1.25, 0.0, 1.0))
    words = ["HELLO", "THANK-YOU", "YES", "NO"]
    idx = int((np.mean(word_features) * 1000) % len(words))
    return words[idx], conf


# ============================================================
# Explanation
# ============================================================

def explain_prediction(label: str, final_label: str, conf: float, threshold: float, lm_norm: np.ndarray) -> str:
    """
    Backward-compatible wrapper that returns a readable explanation string.
    """
    trace = build_letter_explanation_trace(label, final_label, conf, threshold, lm_norm)
    return trace.detailed_text()


# ============================================================
# Model selector
# ============================================================

def build_letter_predictor(cfg: DemoConfig) -> Callable[[np.ndarray], Tuple[str, float]]:
    clf = SklearnClassifier(
        model_path=cfg.letter_model_path,
        scaler_path=cfg.letter_scaler_path,
        label_encoder_path=cfg.letter_label_encoder_path,
    )
    loaded = clf.load()
    print("Letter model loaded:", loaded, "| path:", cfg.letter_model_path)

    def predict(features: np.ndarray) -> Tuple[str, float]:
        if loaded:
            return clf.predict(features)
        return classify_letter_stub(features)

    return predict


def build_handseq_word_predictor(cfg: DemoConfig) -> Callable[[np.ndarray], Tuple[str, float]]:
    clf = SklearnClassifier(
        model_path=cfg.word_handseq_model_path,
        scaler_path="",
        label_encoder_path=cfg.word_handseq_labels_path,
    )
    loaded = clf.load()
    print("HandSeq word model loaded (sklearn):", loaded, "| path:", cfg.word_handseq_model_path)

    expected_n_features = None
    if loaded and hasattr(clf.model, "n_features_in_"):
        try:
            expected_n_features = int(clf.model.n_features_in_)
        except Exception:
            expected_n_features = None

    def predict(hand_seq_summary: np.ndarray) -> Tuple[str, float]:
        if loaded:
            return clf.predict(hand_seq_summary)
        return ("Unknown", 0.0)

    setattr(predict, "expected_n_features", expected_n_features)
    return predict


def motion_energy(hand_hist: Deque[np.ndarray]) -> float:
    if len(hand_hist) < 2:
        return 0.0
    vals = list(hand_hist)
    deltas = []
    for i in range(1, len(vals)):
        a = vals[i - 1]
        b = vals[i]
        deltas.append(float(np.linalg.norm(b - a)))
    return float(np.mean(deltas)) if deltas else 0.0


def build_word_predictor(cfg: DemoConfig) -> Callable[[np.ndarray], Tuple[str, float]]:
    clf = SklearnClassifier(
        model_path=cfg.word_model_path,
        scaler_path="",
        label_encoder_path=cfg.word_label_encoder_path,
    )
    loaded = clf.load()
    print("Word model loaded:", loaded, "| path:", cfg.word_model_path)

    expected_n_features = None
    if loaded and hasattr(clf.model, "n_features_in_"):
        try:
            expected_n_features = int(clf.model.n_features_in_)
        except Exception:
            expected_n_features = None

    def predict(word_features: np.ndarray) -> Tuple[str, float]:
        if loaded:
            return clf.predict(word_features)
        return classify_word_stub(word_features)

    setattr(predict, "expected_n_features", expected_n_features)
    return predict


# ============================================================
# Scenarios
# ============================================================

def scenario_still_image(image_path: str, cfg: DemoConfig) -> dict:
    hand_extractor = HandLandmarkExtractor(model_path=cfg.hand_model_path, max_hands=cfg.max_hands)
    predict_letter = build_letter_predictor(cfg)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    hand_lm, _hand_result = hand_extractor.extract(image)
    if hand_lm is None:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "explanation": "No hand detected.",
            "finger_reason": "No hand landmarks were available.",
        }

    hand_norm = normalize_landmarks(hand_lm)
    feats = flatten_features(hand_norm)

    label, conf = predict_letter(feats)
    out_label = label if conf >= cfg.letter_conf_threshold else "Unknown"
    trace = build_letter_explanation_trace(
        label,
        out_label,
        float(conf),
        cfg.letter_conf_threshold,
        hand_norm,
    )

    return {
        "label": out_label,
        "confidence": float(conf),
        "explanation": trace.detailed_text(),
        "finger_reason": trace.short_text(),
    }


def scenario_webcam(cfg: DemoConfig) -> None:
    hand_extractor = HandLandmarkExtractor(model_path=cfg.hand_model_path, max_hands=cfg.max_hands)
    face_extractor = FaceLandmarkExtractor(model_path=cfg.face_model_path)

    predict_letter = build_letter_predictor(cfg)
    predict_word = build_word_predictor(cfg)
    predict_handseq_word = build_handseq_word_predictor(cfg)
    hybrid_corrector = HybridAutoCorrector(cfg)
    recognition_explainer = RecognitionExplainer(cfg, client=hybrid_corrector.client)
    performance_tracker = PerformanceTracker(cfg.performance_log_path)
    performance_mode = False
    demo_vocabulary = build_demo_vocabulary()

    hand_dim = 63

    expected_n = getattr(predict_word, "expected_n_features", None)
    face_dim_expected: Optional[int] = None
    if isinstance(expected_n, int):
        half = expected_n // 2
        face_dim_expected = max(0, int(half - hand_dim - 63))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    label_hist: Deque[str] = deque(maxlen=cfg.smoothing_window)
    letter_stabilizer = LetterStabilizer(
        stable_frames_required=cfg.stable_letter_frames,
        release_frames_required=cfg.letter_release_frames,
    )

    hand_seq: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)
    face_seq: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)
    rel_seq: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)

    handword_seq: Deque[np.ndarray] = deque(maxlen=cfg.word_handseq_seq_len)
    hand_motion_hist: Deque[np.ndarray] = deque(maxlen=max(2, int(cfg.word_motion_window)))

    transcript_tokens: Deque[str] = deque(maxlen=256)
    last_autocorrect_text = ""
    last_committed_token = ""
    last_seen_token = ""
    token_commit_armed = True
    last_commit_time = 0.0
    last_word_commit_time = 0.0

    word_mode = False
    last_word = "Unknown"
    last_word_conf = 0.0

    max_face_misses = 8
    reuse_last_face_on_miss = True
    face_miss_streak = 0
    last_face_vec: Optional[np.ndarray] = None
    last_face_center: Optional[np.ndarray] = None
    last_face_scale: Optional[float] = None

    toggle_streak = 0
    last_sign_toggle_time = 0.0

    gesture_template = load_toggle_template(cfg.toggle_gesture_template_path)
    gesture_streak = 0
    last_gesture_toggle_time = 0.0

    recording = False
    record_buf: list[np.ndarray] = []

    last_letter_trace: Optional[LetterExplanationTrace] = None
    last_context = ContextExplanation()
    last_t = time.time()
    word_toggle_ord = ord(cfg.word_toggle_key) if cfg.word_toggle_key else ord("=")
    record_key_ord = ord(cfg.toggle_gesture_record_key) if cfg.toggle_gesture_record_key else ord("r")
    clear_key_ord = ord(cfg.clear_transcript_key) if cfg.clear_transcript_key else ord("c")

    def shorten_reason(text: str, max_len: int = 115) -> str:
        clean = sanitize_text_for_display(text)
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 3].rstrip() + "..."

    def do_toggle(now_ts: float, reason: str) -> None:
        nonlocal word_mode, toggle_streak, gesture_streak, last_word, last_word_conf
        nonlocal last_sign_toggle_time, last_gesture_toggle_time
        nonlocal face_miss_streak, last_face_vec, last_face_center, last_face_scale
        nonlocal last_letter_trace

        word_mode = not word_mode
        toggle_streak = 0
        gesture_streak = 0
        label_hist.clear()
        letter_stabilizer.update("Unknown")
        hand_seq.clear()
        face_seq.clear()
        rel_seq.clear()
        handword_seq.clear()
        hand_motion_hist.clear()
        face_miss_streak = 0
        last_face_vec = None
        last_face_center = None
        last_face_scale = None
        last_word = "Unknown"
        last_word_conf = 0.0
        last_letter_trace = None

        if reason == "sign":
            last_sign_toggle_time = now_ts
        elif reason == "gesture":
            last_gesture_toggle_time = now_ts
        elif reason == "key":
            last_sign_toggle_time = now_ts
            last_gesture_toggle_time = now_ts

    def clear_transcript_state() -> None:
        nonlocal last_autocorrect_text, last_committed_token, last_seen_token
        nonlocal token_commit_armed, last_commit_time, last_word_commit_time, last_context
        nonlocal last_letter_trace

        performance_tracker.force_finalize()
        transcript_tokens.clear()
        last_autocorrect_text = ""
        last_committed_token = ""
        last_seen_token = ""
        token_commit_armed = True
        last_commit_time = 0.0
        last_word_commit_time = 0.0
        last_context = ContextExplanation()
        last_letter_trace = None

        hybrid_corrector.reset()
        recognition_explainer.reset()

    def maybe_commit_token(token: str, mode_name: str) -> None:
        nonlocal last_committed_token, last_seen_token, token_commit_armed
        nonlocal last_commit_time, last_word_commit_time

        now_ts = time.time()

        if token == "Unknown" or token == cfg.word_toggle_label:
            last_seen_token = token
            token_commit_armed = True
            return

        if token in {"=", "[space]", "space", "SPACE"}:
            token = " "

        if token == " ":
            if transcript_tokens and transcript_tokens[-1] != " ":
                transcript_tokens.append(" ")
                last_committed_token = " "
                last_commit_time = now_ts
            last_seen_token = token
            token_commit_armed = True
            return

        if mode_name == "letter":
            if len(token) != 1:
                last_seen_token = token
                token_commit_armed = True
                return
        else:
            if len(token) < 1:
                last_seen_token = token
                token_commit_armed = True
                return

        if token != last_seen_token:
            token_commit_armed = True
            last_seen_token = token

        if (now_ts - last_commit_time) < cfg.min_letter_commit_gap_sec:
            return

        if mode_name == "letter":
            same_as_last = token == last_committed_token
            repeat_cooldown_done = (now_ts - last_commit_time) >= cfg.repeat_letter_cooldown_sec
            released = letter_stabilizer.released_since_commit()

            should_commit = False

            if not same_as_last:
                should_commit = True
            elif same_as_last and (released or repeat_cooldown_done):
                should_commit = True

            if should_commit and token_commit_armed:
                transcript_tokens.append(token)
                last_committed_token = token
                last_commit_time = now_ts
                token_commit_armed = False
                letter_stabilizer.mark_committed(token)

            return

        # Word mode commits complete words, so add a larger word-only delay.
        # This prevents rapid duplicate/near-duplicate word inserts while keeping
        # letter mode responsive.
        if (now_ts - last_word_commit_time) < cfg.word_commit_cooldown_sec:
            return

        if token_commit_armed:
            transcript_tokens.append(token)
            last_committed_token = token
            last_commit_time = now_ts
            last_word_commit_time = now_ts
            token_commit_armed = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()

        hand_lm, hand_result = hand_extractor.extract(frame)
        face_lm, face_result = face_extractor.extract(frame)

        mode = "word" if word_mode else "letter"
        display_label = "Unknown"
        conf = 0.0
        explanation = ""
        gesture_sim = 0.0
        motion_value = motion_energy(hand_motion_hist)

        if hand_lm is None:
            display_label = "Unknown"
            conf = 0.0
            explanation = "No hand detected; output suppressed."
            last_letter_trace = None
            label_hist.append("Unknown")
            letter_stabilizer.update("Unknown")
            hand_seq.clear()
            face_seq.clear()
            rel_seq.clear()
            handword_seq.clear()
            hand_motion_hist.clear()
            toggle_streak = 0
            gesture_streak = 0

            if recording:
                recording = False
                record_buf = []

        else:
            hand_extractor.draw(frame, hand_result)
            if face_result is not None:
                face_extractor.draw(frame, face_result, step=6)

            hand_norm = normalize_landmarks(hand_lm)
            hand_feats = fit_dim(flatten_features(hand_norm), hand_dim)
            hand_motion_hist.append(hand_feats.copy())
            handword_seq.append(hand_feats.copy())
            motion_value = motion_energy(hand_motion_hist)

            if recording:
                record_buf.append(hand_feats.copy())
                need = int(max(1, cfg.toggle_gesture_record_frames))
                explanation = f"Recording toggle gesture: {len(record_buf)}/{need} frames..."
                if len(record_buf) >= need:
                    template = np.mean(np.stack(record_buf, axis=0), axis=0).astype(np.float32)
                    ok_save = save_toggle_template(cfg.toggle_gesture_template_path, template)
                    if ok_save:
                        gesture_template = template
                        explanation = f"Saved toggle gesture to '{cfg.toggle_gesture_template_path}'."
                    else:
                        explanation = "Failed to save toggle gesture template."
                    recording = False
                    record_buf = []

            if cfg.toggle_gesture_enabled and (gesture_template is not None) and (not recording):
                tpl = fit_dim(gesture_template, hand_dim)
                gesture_sim = cosine_similarity(hand_feats, tpl)

                if (now - last_gesture_toggle_time) >= cfg.toggle_gesture_cooldown_sec:
                    if gesture_sim >= cfg.toggle_gesture_threshold:
                        gesture_streak += 1
                    else:
                        gesture_streak = 0
                else:
                    gesture_streak = 0

                if gesture_streak >= cfg.toggle_gesture_streak_required:
                    do_toggle(now, reason="gesture")
                    explanation = "Toggled mode via recorded gesture."
                    mode = "word" if word_mode else "letter"

            if word_mode:
                last_letter_trace = None
                goto_dynamic = motion_value >= cfg.word_motion_threshold

                if not goto_dynamic:
                    if len(handword_seq) == int(cfg.word_handseq_seq_len):
                        hwf = summarize_hand_sequence(np.stack(handword_seq))
                        expected_handseq_n = getattr(predict_handseq_word, "expected_n_features", None)

                        if isinstance(expected_handseq_n, int) and hwf.size != expected_handseq_n:
                            word, wconf = "Unknown", 0.0
                            if not explanation:
                                explanation = (
                                    f"HandSeq feature size mismatch: got {hwf.size}, expected {expected_handseq_n}."
                                )
                        else:
                            word, wconf = predict_handseq_word(hwf)

                        last_word = word if wconf >= cfg.word_handseq_conf_threshold else "Unknown"
                        last_word_conf = float(wconf)

                        display_label = last_word
                        conf = last_word_conf

                        if not explanation:
                            explanation = (
                                f"Word mode used the lower-motion branch because motion {motion_value:.4f} stayed "
                                f"below {cfg.word_motion_threshold:.4f}. After summarizing "
                                f"{cfg.word_handseq_seq_len} hand frames, it predicted {repr(word)} at {wconf:.2f}."
                            )

                        handword_seq.clear()
                    else:
                        display_label = last_word
                        conf = last_word_conf
                        if not explanation:
                            explanation = (
                                f"Word mode static branch is buffering {len(handword_seq)}/{cfg.word_handseq_seq_len} "
                                f"hand frames. Last stable word is {repr(last_word)} at {last_word_conf:.2f}."
                            )

                    hand_seq.clear()
                    face_seq.clear()
                    rel_seq.clear()

                else:
                    face_feats_raw: Optional[np.ndarray] = None
                    face_center: Optional[np.ndarray] = None
                    face_scale: Optional[float] = None

                    if face_lm is not None:
                        face_feats_raw = flatten_features(normalize_landmarks(face_lm))
                        face_center, face_scale = face_center_and_scale(face_lm)

                        face_miss_streak = 0
                        last_face_center = face_center.copy()
                        last_face_scale = float(face_scale)
                        if face_dim_expected is not None:
                            last_face_vec = fit_dim(face_feats_raw, face_dim_expected).copy()
                        else:
                            last_face_vec = face_feats_raw.copy()

                    else:
                        face_miss_streak += 1
                        if reuse_last_face_on_miss and face_miss_streak <= max_face_misses:
                            if last_face_vec is not None:
                                face_feats_raw = last_face_vec.copy()
                            if last_face_center is not None and last_face_scale is not None:
                                face_center = last_face_center.copy()
                                face_scale = float(last_face_scale)

                    if face_dim_expected is None:
                        if face_feats_raw is None:
                            if not explanation:
                                explanation = (
                                    "Word mode dynamic branch is waiting for face landmarks so the face feature size can lock."
                                )
                            display_label = last_word
                            conf = last_word_conf
                        else:
                            face_dim_expected = int(face_feats_raw.size)

                    if (
                        face_dim_expected is not None
                        and face_feats_raw is not None
                        and face_center is not None
                        and face_scale is not None
                    ):
                        face_feats = fit_dim(face_feats_raw, face_dim_expected)
                        rel_feats = fit_dim(hand_relative_to_face(hand_lm, face_center, face_scale), 63)

                        hand_seq.append(hand_feats)
                        face_seq.append(face_feats)
                        rel_seq.append(rel_feats)

                        if (
                            len(hand_seq) == cfg.seq_len
                            and len(face_seq) == cfg.seq_len
                            and len(rel_seq) == cfg.seq_len
                        ):
                            wf = make_word_features(
                                np.stack(hand_seq),
                                np.stack(face_seq),
                                np.stack(rel_seq),
                            )

                            expected_n2 = getattr(predict_word, "expected_n_features", None)
                            if isinstance(expected_n2, int) and wf.size != expected_n2:
                                last_word = "Unknown"
                                last_word_conf = 0.0
                                display_label = last_word
                                conf = last_word_conf
                                explanation = f"Word feature size mismatch: got {wf.size}, expected {expected_n2}."
                            else:
                                word, wconf = predict_word(wf)
                                last_word = word if wconf >= cfg.word_conf_threshold else "Unknown"
                                last_word_conf = float(wconf)

                                display_label = last_word
                                conf = last_word_conf

                                if not explanation:
                                    explanation = (
                                        f"Word mode used the dynamic branch because motion {motion_value:.4f} exceeded "
                                        f"{cfg.word_motion_threshold:.4f}. After summarizing {cfg.seq_len} frames of "
                                        f"hand, face, and relative features, it predicted {repr(word)} at {wconf:.2f}."
                                    )

                            hand_seq.clear()
                            face_seq.clear()
                            rel_seq.clear()
                        else:
                            display_label = last_word
                            conf = last_word_conf
                            if not explanation:
                                explanation = (
                                    f"Word mode dynamic branch is buffering {len(hand_seq)}/{cfg.seq_len} frames. "
                                    f"Last stable word is {repr(last_word)} at {last_word_conf:.2f}."
                                )
                    else:
                        display_label = last_word
                        conf = last_word_conf
                        if not explanation and face_miss_streak > 0:
                            explanation = (
                                f"Word mode dynamic branch is holding the last prediction because face landmarks are missing "
                                f"({face_miss_streak}/{max_face_misses})."
                            )

                handseq_label = "Unknown"
                handseq_conf = 0.0
                if len(handword_seq) == int(cfg.word_handseq_seq_len):
                    hwf = summarize_hand_sequence(np.stack(handword_seq))
                    expected_handseq_n = getattr(predict_handseq_word, "expected_n_features", None)

                    if isinstance(expected_handseq_n, int) and hwf.size != expected_handseq_n:
                        handseq_label, handseq_conf = "Unknown", 0.0
                    else:
                        handseq_label, handseq_conf = predict_handseq_word(hwf)

                    if handseq_conf < cfg.word_handseq_conf_threshold:
                        handseq_label = "Unknown"

                if handseq_label != "Unknown" and (
                    last_word == "Unknown" or handseq_conf >= (last_word_conf + cfg.word_handseq_margin)
                ):
                    display_label = handseq_label
                    conf = float(handseq_conf)
                    if not explanation:
                        explanation = (
                            f"Hand-sequence override won because its confidence {handseq_conf:.2f} beat the current "
                            f"word estimate by at least {cfg.word_handseq_margin:.2f}."
                        )
                    handword_seq.clear()

            else:
                mode = "letter"

                raw_label, conf = predict_letter(hand_feats)
                raw_display_label = raw_label if conf >= cfg.letter_conf_threshold else "Unknown"
                last_letter_trace = build_letter_explanation_trace(
                    raw_label,
                    raw_display_label,
                    float(conf),
                    cfg.letter_conf_threshold,
                    hand_norm,
                )
                explanation = last_letter_trace.detailed_text()

                label_hist.append(raw_display_label)
                voted_label = raw_display_label
                if label_hist:
                    counts: dict[str, int] = {}
                    for lab in label_hist:
                        counts[lab] = counts.get(lab, 0) + 1
                    voted_label = max(counts, key=counts.get)

                display_label = letter_stabilizer.update(voted_label)

                if raw_label == cfg.word_toggle_label and conf >= cfg.letter_conf_threshold:
                    if (now - last_sign_toggle_time) >= cfg.toggle_cooldown_sec:
                        toggle_streak += 1
                    else:
                        toggle_streak = 0

                    if toggle_streak >= cfg.toggle_streak_required:
                        do_toggle(now, reason="sign")
                        explanation = "Toggled mode via sign label."
                        mode = "word" if word_mode else "letter"
                        toggle_streak = 0
                else:
                    toggle_streak = 0

                hand_seq.clear()
                face_seq.clear()
                rel_seq.clear()
                handword_seq.clear()

        maybe_commit_token(display_label, mode)

        candidate_letters = tokens_to_fingerspelled_candidate(
            list(transcript_tokens),
            cfg.fuzzy_letter_commit_len
        )
        last_autocorrect_text = hybrid_corrector.update(candidate_letters, mode=mode, transcript_snapshot=transcript_to_string(list(transcript_tokens)))

        corrected_tokens = corrected_transcript_view(
            list(transcript_tokens),
            demo_vocabulary,
            cfg,
            mode
        )

        fps = 1.0 / max(now - last_t, 1e-6)
        last_t = now

        transcript_text = sanitize_text_for_display(
            transcript_to_string(list(transcript_tokens)),
            max_len=180,
        )
        local_corrected_transcript_text = sanitize_text_for_display(
            transcript_to_string(corrected_tokens),
            max_len=180,
        )

        correction_decision = hybrid_corrector.last_decision

        corrected_transcript_text = sanitize_text_for_display(
            build_corrected_transcript_from_decision(
                list(transcript_tokens),
                correction_decision,
                fallback_text=local_corrected_transcript_text,
            ),
            max_len=180,
        )

        display_context_text = corrected_transcript_text

        performance_tracker.finalize_candidate_if_ready(
            candidate_letters,
            hybrid_corrector.last_metrics,
            display_context_text or transcript_text,
        )

        last_context = recognition_explainer.update(
            list(transcript_tokens),
            display_context_text,
            correction_decision,
            mode,
        )

        if last_letter_trace is not None:
            finger_reason = last_letter_trace.short_text()
        elif hand_lm is None:
            finger_reason = "No hand evidence is available because no hand was detected."
        else:
            finger_reason = (
                f"Word mode relied on temporal evidence. Motion was {motion_value:.4f}, and the system summarized "
                "multiple hand and optional face frames instead of a single letter pose."
            )

        word_reason = clean_overlay_reason(correction_decision.short_text(), max_len=180)
        sentence_reason = clean_overlay_reason(last_context.short_text(), max_len=180)

        short_why = word_reason or sentence_reason or "No explanation yet."
        short_summary = sentence_reason or word_reason or "No explanation yet."

        h, w = frame.shape[:2]
        panel_x = 12
        panel_y = 12
        panel_w = int(min(max(560, w * 0.88), max(560, w - 20)))
        panel_h = int(min(max(420, h * 0.95), max(420, h - 20)))
        draw_overlay_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.38)

        x0 = panel_x + 12
        y = panel_y + 22
        max_w = int(max(60, panel_w - 24))
        bottom_margin = max(16, h - (panel_y + panel_h))
        available_overlay_height = max(60, panel_h - 28)

        font = cv2.FONT_HERSHEY_SIMPLEX
        th_fg = 1
        th_bg = 3

        safe_prediction = sanitize_text_for_display(display_label_text(display_label), max_len=40)
        safe_transcript = sanitize_text_for_display(transcript_text)
        safe_corrected = sanitize_text_for_display(display_context_text)
        safe_word_reason = clean_overlay_reason(word_reason)
        safe_summary = sanitize_text_for_display(short_summary)

        if hybrid_corrector.debug_source == "OpenAI":
            correction_label = "OpenAI"
        elif hybrid_corrector.debug_source == "Local Gemma":
            correction_label = "Gemma"
        elif hybrid_corrector.debug_status == "Local fuzzy correction used":
            correction_label = "Fuzzy"
        else:
            correction_label = "Local"

        if last_context.source == "OpenAI":
            explanation_label = "OpenAI"
        elif last_context.source == "Local Gemma":
            explanation_label = "Gemma"
        else:
            explanation_label = "Local"

        if performance_mode:
            latest_metrics = performance_tracker.latest or hybrid_corrector.last_metrics
            counts = performance_tracker.source_counts()
            perf_title = "Sign-Bridge Performance"
            perf_line1 = f"Mode: {mode.upper()} | Pred: {safe_prediction} | FPS: {fps:.1f}"
            perf_line2 = f"Latest: {latest_metrics.raw_input or '(none)'} -> {latest_metrics.final_output or '(none)'}"
            perf_line3 = f"Scenario/Src: {latest_metrics.scenario_type or 'None'} / {latest_metrics.correction_source}"
            perf_line4 = f"Times ms | Fuzzy {latest_metrics.fuzzy_time_ms:.1f} | Gemma {latest_metrics.gemma_time_ms:.1f} | OpenAI {latest_metrics.openai_time_ms:.1f} | Total {latest_metrics.total_time_ms:.1f}"
            perf_line5 = f"Runs: {performance_tracker.count()} | Avg Total: {performance_tracker.avg_total_ms():.1f} ms"
            openai_label = "used" if latest_metrics.openai_attempted else ("ready" if latest_metrics.openai_available else "off")
            perf_line6 = f"Wins F/G/O: {counts['Fuzzy']}/{counts['Gemma']}/{counts['OpenAI']} | OpenAI: {openai_label}"
            perf_line7 = sanitize_text_for_display(latest_metrics.openai_status or performance_tracker._last_log_message or 'Waiting for completed run to log...', max_len=120)
            perf_line8 = f"CSV: {sanitize_text_for_display(cfg.performance_log_path, max_len=60)}"
            line5 = f"Keys: ESC quit {cfg.performance_toggle_key} perf {cfg.word_toggle_key} toggle {cfg.toggle_gesture_record_key} rec {cfg.clear_transcript_key} clear"
            overlay_blocks = [
                {"text": perf_title, "scale": 0.74, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": perf_line1, "scale": 0.55, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": perf_line2, "scale": 0.52, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": perf_line3, "scale": 0.50, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": perf_line4, "scale": 0.43, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": perf_line5, "scale": 0.46, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": perf_line6, "scale": 0.40, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": perf_line7, "scale": 0.40, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": perf_line8, "scale": 0.38, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": sanitize_text_for_display(line5), "scale": 0.40, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
            ]
        else:
            title = "Sign-Bridge Demo"
            line1 = f"Mode: {mode.upper()} | Pred: {safe_prediction} | Conf: {conf:.2f} | FPS: {fps:.1f}"
            line2 = f"Transcript: {safe_transcript if safe_transcript else '(none)'}"
            line3 = f"Corrected: {safe_corrected if safe_corrected else '(none)'}"
            line4 = f"Why: {safe_word_reason if safe_word_reason else '(none)'}"
            line4b = f"Summary: {safe_summary if safe_summary else '(none)'}"
            line4c = f"Src: {correction_label}/{explanation_label}"

            if recording:
                line5 = f"Recording: {len(record_buf)}/{max(1, cfg.toggle_gesture_record_frames)}"
            else:
                line5 = f"Keys: ESC quit {cfg.performance_toggle_key} perf {cfg.word_toggle_key} toggle {cfg.toggle_gesture_record_key} rec {cfg.clear_transcript_key} clear"
            line5 = sanitize_text_for_display(line5)

            overlay_blocks = [
                {"text": title, "scale": 0.74, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": line1, "scale": 0.55, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": line2, "scale": 0.50, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": line3, "scale": 0.50, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 4},
                {"text": line4, "scale": 0.44, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": line4b, "scale": 0.44, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
                {"text": line4c, "scale": 0.43, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3},
            ]
            if line5:
                overlay_blocks.append({"text": line5, "scale": 0.40, "thickness_fg": th_fg, "thickness_bg": th_bg, "line_gap": 3})

        draw_fitted_overlay_blocks(
            frame,
            x=x0,
            y=y,
            max_width_px=max_w,
            available_height_px=available_overlay_height,
            blocks=overlay_blocks,
            font_face=font,
            bottom_margin=bottom_margin,
        )

        cv2.imshow("Sign-Bridge", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key in (ord(cfg.performance_toggle_key.lower()), ord(cfg.performance_toggle_key.upper())):
            performance_mode = not performance_mode

        if key == word_toggle_ord:
            do_toggle(time.time(), reason="key")

        if key == record_key_ord and not recording:
            recording = True
            record_buf = []

        if key == clear_key_ord:
            clear_transcript_state()

    performance_tracker.force_finalize()
    cap.release()
    hybrid_corrector.shutdown()
    recognition_explainer.shutdown()
    cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    cfg = DemoConfig()
    scenario_webcam(cfg)
