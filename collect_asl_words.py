from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Face subset (reduce dimensionality + noise for words)
# ============================================================

# A compact set of FaceMesh landmark indices focused on mouth + eyes + brows.
# This keeps word features small and stable while still capturing useful facial cues.
FACE_SUBSET_IDX: List[int] = [
    # Lips (outer + inner-ish)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185,
    40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 80, 191, 78,
    # Left eye
    33, 133, 160, 159, 158, 144, 145, 153,
    # Right eye
    362, 263, 387, 386, 385, 373, 374, 380,
    # Brows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
]


# ============================================================
# Overlay helpers (wrap text to avoid going off-screen)
# ============================================================

def wrap_text_to_width(
    text: str,
    max_width_px: int,
    font_face: int,
    font_scale: float,
    thickness: int
) -> List[str]:
    """
    Wrap text into multiple lines so each line fits within max_width_px.
    Also hard-wraps long tokens so they can't overflow.

    @param text: Input string to wrap.
    @param max_width_px: Maximum pixel width allowed for each line.
    @param font_face: cv2 font face.
    @param font_scale: Font scale used for cv2.putText.
    @param thickness: Text thickness used for cv2.putText.
    @return: List of wrapped lines.
    """
    if not text:
        return [""]

    def text_width(s: str) -> int:
        (tw, _), _ = cv2.getTextSize(s, font_face, font_scale, thickness)
        return int(tw)

    def split_long_token(token: str) -> List[str]:
        if text_width(token) <= max_width_px:
            return [token]
        out: List[str] = []
        cur = ""
        for ch in token:
            test = cur + ch
            if text_width(test) <= max_width_px:
                cur = test
            else:
                if cur:
                    out.append(cur)
                    cur = ch
                else:
                    out.append(ch)
                    cur = ""
        if cur:
            out.append(cur)
        return out

    words = text.replace("\n", " ").split()
    lines: List[str] = []
    cur = ""

    for w in words:
        pieces = split_long_token(w) if text_width(w) > max_width_px else [w]
        for piece in pieces:
            test = piece if not cur else (cur + " " + piece).strip()
            if text_width(test) <= max_width_px:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = piece

    if cur:
        lines.append(cur)

    return lines


def draw_wrapped_block(
    frame_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font: int,
    font_scale: float,
    fg_thickness: int,
    bg_thickness: int,
    max_lines: int = 6,
    line_gap: int = 6
) -> int:
    """
    Draw a multi-line wrapped text block starting at (x,y).

    @param frame_bgr: Frame to draw on.
    @param text: Text to wrap/draw.
    @param x: X origin.
    @param y: Y baseline for first line.
    @param max_width: Max pixel width.
    @param font: cv2 font.
    @param font_scale: Font scale.
    @param fg_thickness: Foreground thickness.
    @param bg_thickness: Background thickness.
    @param max_lines: Maximum lines to draw.
    @param line_gap: Gap between lines in pixels.
    @return: Updated y position after drawing.
    """
    lines = wrap_text_to_width(text, max_width, font, font_scale, fg_thickness)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1] + " ..."

    (tw, th), _ = cv2.getTextSize("Ag", font, font_scale, fg_thickness)
    line_h = int(th) + line_gap

    for line in lines:
        cv2.putText(frame_bgr, line, (x, y), font, font_scale, (0, 0, 0), bg_thickness, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (x, y), font, font_scale, (255, 255, 255), fg_thickness, cv2.LINE_AA)
        y += line_h

    return y


# ============================================================
# MediaPipe Tasks Hand / Face Landmarkers
# ============================================================

@dataclass
class TaskExtractorConfig:
    """
    Configuration for MediaPipe Tasks landmarkers.

    @param hand_model_path: Path to the hand_landmarker.task model.
    @param face_model_path: Path to the face_landmarker.task model.
    @param max_hands: Maximum number of hands to detect.
    """
    hand_model_path: str = os.path.join("models", "hand_landmarker.task")
    face_model_path: str = os.path.join("models", "face_landmarker.task")
    max_hands: int = 1


class HandLandmarkExtractor:
    """
    Extract hand landmarks using MediaPipe Tasks Hand Landmarker.

    @param cfg: TaskExtractorConfig settings.
    """

    def __init__(self, cfg: TaskExtractorConfig) -> None:
        """
        Initialize the Tasks hand landmarker.

        @param cfg: TaskExtractorConfig settings.
        @raises FileNotFoundError: If cfg.hand_model_path does not exist.
        """
        if not os.path.exists(cfg.hand_model_path):
            raise FileNotFoundError(
                f"Missing model file: {cfg.hand_model_path}. "
                "Download a real hand_landmarker.task into the models/ folder."
            )

        base_options = python.BaseOptions(model_asset_path=cfg.hand_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[object]]:
        """
        Extract 21 hand landmarks from a BGR frame.

        @param frame_bgr: OpenCV BGR frame.
        @return: (landmarks ndarray shape (21,3) or None, raw result or None).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None, None

        lms = result.hand_landmarks[0]
        lm_array = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
        return lm_array, result

    def draw(self, frame_bgr: np.ndarray, result: object) -> None:
        """
        Draw detected hand landmarks.

        @param frame_bgr: Frame to draw on.
        @param result: MediaPipe Tasks result.
        """
        if result is None or not getattr(result, "hand_landmarks", None):
            return
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * frame_bgr.shape[1])
                y = int(lm.y * frame_bgr.shape[0])
                cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)


class FaceLandmarkExtractor:
    """
    Extract face landmarks using MediaPipe Tasks Face Landmarker.

    @param cfg: TaskExtractorConfig settings.
    """

    def __init__(self, cfg: TaskExtractorConfig) -> None:
        """
        Initialize the Tasks face landmarker.

        @param cfg: TaskExtractorConfig settings.
        @raises FileNotFoundError: If cfg.face_model_path does not exist.
        """
        if not os.path.exists(cfg.face_model_path):
            raise FileNotFoundError(
                f"Missing model file: {cfg.face_model_path}. "
                "Download a real face_landmarker.task into the models/ folder."
            )

        base_options = python.BaseOptions(model_asset_path=cfg.face_model_path)
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
        """
        Extract face landmarks from a BGR frame.

        @param frame_bgr: OpenCV BGR frame.
        @return: (landmarks ndarray shape (N,3) or None, raw result or None).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None, None

        lms = result.face_landmarks[0]
        lm_array = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
        return lm_array, result

    def draw(self, frame_bgr: np.ndarray, result: object, step: int = 6) -> None:
        """
        Draw a sparse subset of face landmarks.

        @param frame_bgr: Frame to draw on.
        @param result: MediaPipe Tasks result.
        @param step: Draw every Nth landmark to reduce clutter.
        """
        if result is None or not getattr(result, "face_landmarks", None):
            return
        lms = result.face_landmarks[0]
        for i, lm in enumerate(lms):
            if i % step != 0:
                continue
            x = int(lm.x * frame_bgr.shape[1])
            y = int(lm.y * frame_bgr.shape[0])
            cv2.circle(frame_bgr, (x, y), 1, (255, 0, 0), -1)


# ============================================================
# Feature helpers
# ============================================================

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks by subtracting mean and dividing by scale.

    @param landmarks: Array shape (N,3).
    @return: Normalized landmarks shape (N,3).
    """
    pts = landmarks.astype(np.float32)
    center = pts.mean(axis=0)
    pts = pts - center
    scale = float(np.linalg.norm(pts, axis=1).mean() + 1e-6)
    pts = pts / scale
    return pts


def flatten_xyz(landmarks: np.ndarray) -> np.ndarray:
    """
    Flatten (N,3) to (N*3,).

    @param landmarks: Array shape (N,3).
    @return: Flattened vector.
    """
    return landmarks.reshape(-1).astype(np.float32)


def face_subset_features(face_lm: np.ndarray, subset_idx: List[int]) -> np.ndarray:
    """
    Normalize full face landmarks, take subset indices, flatten to 1D.

    @param face_lm: Full face landmarks (N,3), N usually 468.
    @param subset_idx: Indices to keep.
    @return: Flattened face subset vector shape (len(subset_idx)*3,).
    """
    face_norm = normalize_landmarks(face_lm)
    idx = [i for i in subset_idx if 0 <= i < face_norm.shape[0]]
    sub = face_norm[idx, :] if idx else np.zeros((0, 3), dtype=np.float32)
    return flatten_xyz(sub)


def face_center_and_scale(face_lm: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute a face reference frame (center + scale) for relative features.

    @param face_lm: Face landmarks array (N,3).
    @return: (center shape (3,), scale > 0).
    """
    pts = face_lm.astype(np.float32)
    center = pts.mean(axis=0).astype(np.float32)
    scale = float(np.linalg.norm(pts - center[None, :], axis=1).mean() + 1e-6)
    return center, scale


def hand_relative_to_face(hand_lm: np.ndarray, face_center: np.ndarray, face_scale: float) -> np.ndarray:
    """
    Express hand landmarks in the face coordinate system.

    @param hand_lm: Hand landmarks (21,3).
    @param face_center: Face center (3,).
    @param face_scale: Face scale.
    @return: Flattened relative vector shape (63,).
    """
    rel = (hand_lm.astype(np.float32) - face_center[None, :]) / float(face_scale)
    return rel.reshape(-1).astype(np.float32)


def hand_features(hand_lm: np.ndarray) -> np.ndarray:
    """
    Normalize and flatten hand landmarks.

    @param hand_lm: Hand landmarks (21,3).
    @return: Flattened hand vector shape (63,).
    """
    return flatten_xyz(normalize_landmarks(hand_lm))


def frame_word_features(hand_lm: np.ndarray, face_lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-frame word features (hand + face subset + rel).

    @param hand_lm: Hand landmarks (21,3).
    @param face_lm: Face landmarks (N,3).
    @return: (hand_vec(63), face_vec(face_dim), rel_vec(63)).
    """
    hv = hand_features(hand_lm)
    fv = face_subset_features(face_lm, FACE_SUBSET_IDX)
    fc, fs = face_center_and_scale(face_lm)
    rv = hand_relative_to_face(hand_lm, fc, fs)
    return hv, fv, rv


# ============================================================
# Collection config and state machine
# ============================================================

@dataclass
class CollectConfig:
    """
    Collector configuration.

    @param labels: Word labels to record.
    @param seq_len: Frames per sample.
    @param sample_hz: Sampling rate (frames/sec) for buffering.
    @param countdown_sec: Countdown before recording begins.
    @param output_dir: Root output directory.
    @param cooldown_sec: Pause after saving a sample.
    @param motion_start_thresh: Start capturing when motion energy exceeds this threshold.
    @param motion_stop_thresh: Optional: if set, stop early when motion falls below threshold (not used for fixed seq_len).
    @param motion_window: Window length for smoothing motion energy.
    @param max_hand_misses: Max consecutive hand misses allowed while capturing.
    @param max_face_misses: Max consecutive face misses allowed while capturing.
    @param reuse_last_face_on_miss: If face flickers, reuse last face landmarks (short streak).
    """
    labels: List[str]
    seq_len: int = 25
    sample_hz: float = 10.0
    countdown_sec: float = 2.0
    output_dir: str = os.path.join("data", "words_seq")
    cooldown_sec: float = 0.5

    motion_start_thresh: float = 0.020
    motion_stop_thresh: Optional[float] = None
    motion_window: int = 5

    max_hand_misses: int = 6
    max_face_misses: int = 8
    reuse_last_face_on_miss: bool = True


class CaptureState:
    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    COOLDOWN = "COOLDOWN"


def ensure_dirs(cfg: CollectConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)


def save_sample_npz(
    cfg: CollectConfig,
    label: str,
    hand_seq: np.ndarray,
    face_seq: np.ndarray,
    rel_seq: np.ndarray,
    meta: dict
) -> str:
    """
    Save a single sample as .npz.

    @param cfg: Collector config.
    @param label: Word label.
    @param hand_seq: (T,63)
    @param face_seq: (T,face_dim)
    @param rel_seq: (T,63)
    @param meta: Metadata dict.
    @return: Saved file path.
    """
    label_dir = os.path.join(cfg.output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{label}_{ts}_{int(time.time()*1000)%100000}.npz"
    out_path = os.path.join(label_dir, fname)

    np.savez_compressed(
        out_path,
        hand_seq=hand_seq.astype(np.float32),
        face_seq=face_seq.astype(np.float32),
        rel_seq=rel_seq.astype(np.float32),
        label=np.array([label]),
        meta=json.dumps(meta),
        face_subset_idx=np.array(FACE_SUBSET_IDX, dtype=np.int32),
    )
    return out_path


def compute_motion_energy(prev_rel: Optional[np.ndarray], cur_rel: np.ndarray) -> float:
    """
    Simple motion proxy: mean absolute delta of rel vector.

    @param prev_rel: Previous rel vector or None.
    @param cur_rel: Current rel vector.
    @return: Motion energy scalar.
    """
    if prev_rel is None:
        return 0.0
    return float(np.mean(np.abs(cur_rel - prev_rel)))


def main() -> None:
    """
    Word collector for a long-run sequence model.

    What it saves:
    - data/words_seq/<LABEL>/*.npz
      Each .npz contains hand_seq (T,63), face_seq (T,face_dim_reduced), rel_seq (T,63),
      plus label + metadata. This format is ideal for a CNN/GRU word model.

    Controls:
    - N / P : next / previous label
    - R     : start/stop recording (continuous)
    - C     : clear current buffers
    - ESC   : quit
    """
    cfg = CollectConfig(labels=["HELLO", "THANK-YOU", "YES", "NO"])
    ensure_dirs(cfg)

    task_cfg = TaskExtractorConfig(max_hands=1)
    hand_extractor = HandLandmarkExtractor(task_cfg)
    face_extractor = FaceLandmarkExtractor(task_cfg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    label_idx = 0
    current_label = cfg.labels[label_idx]

    state = CaptureState.IDLE
    recording_enabled = False
    countdown_until: Optional[float] = None
    cooldown_until: float = 0.0

    # Buffers (store per-frame features)
    hand_buf: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)
    face_buf: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)
    rel_buf: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)

    # Face dropout tolerance (reuse last)
    face_miss_streak = 0
    last_face_lm: Optional[np.ndarray] = None

    # Hand/face miss while capturing
    hand_miss_streak = 0

    # Motion smoothing
    prev_rel: Optional[np.ndarray] = None
    motion_hist: Deque[float] = deque(maxlen=max(1, int(cfg.motion_window)))

    saved = 0
    last_sample_t = 0.0
    sample_interval = 1.0 / max(1e-6, float(cfg.sample_hz))

    font = cv2.FONT_HERSHEY_SIMPLEX

    def clear_buffers() -> None:
        nonlocal prev_rel
        hand_buf.clear()
        face_buf.clear()
        rel_buf.clear()
        motion_hist.clear()
        prev_rel = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()

        # Extract landmarks
        hand_lm, hand_result = hand_extractor.extract(frame)
        face_lm, face_result = face_extractor.extract(frame)

        # Face fallback (reuse last face landmarks briefly)
        if face_lm is not None:
            last_face_lm = face_lm.copy()
            face_miss_streak = 0
        else:
            face_miss_streak += 1
            if cfg.reuse_last_face_on_miss and last_face_lm is not None and face_miss_streak <= cfg.max_face_misses:
                face_lm = last_face_lm.copy()

        # Compute motion energy if possible
        cur_motion = 0.0
        if hand_lm is not None and face_lm is not None:
            _, _, rel_vec = frame_word_features(hand_lm, face_lm)
            cur_motion = compute_motion_energy(prev_rel, rel_vec)
            prev_rel = rel_vec
        motion_hist.append(cur_motion)
        motion_smooth = float(np.mean(motion_hist)) if motion_hist else 0.0

        # State transitions
        if not recording_enabled:
            state = CaptureState.IDLE
            countdown_until = None
            clear_buffers()

        if state == CaptureState.COOLDOWN:
            if now >= cooldown_until:
                state = CaptureState.IDLE

        if recording_enabled and state == CaptureState.IDLE:
            # Optional countdown gate
            if countdown_until is None:
                countdown_until = now + float(cfg.countdown_sec)

            if now >= countdown_until:
                # Start only when we see enough motion AND have both hand+face
                if hand_lm is not None and face_lm is not None and motion_smooth >= cfg.motion_start_thresh:
                    state = CaptureState.CAPTURING
                    clear_buffers()
                    hand_miss_streak = 0
                    face_miss_streak = 0
                    last_sample_t = 0.0

        if recording_enabled and state == CaptureState.CAPTURING:
            # Throttle sampling to sample_hz
            if last_sample_t == 0.0 or (now - last_sample_t) >= sample_interval:
                last_sample_t = now

                if hand_lm is None:
                    hand_miss_streak += 1
                else:
                    hand_miss_streak = 0

                if face_lm is None:
                    face_miss_streak += 1
                else:
                    # face_miss_streak already managed above, but keep safe
                    face_miss_streak = 0

                if hand_miss_streak > cfg.max_hand_misses or face_miss_streak > cfg.max_face_misses:
                    # Abort sample if tracking is too unstable
                    state = CaptureState.IDLE
                    countdown_until = now + float(cfg.countdown_sec)
                    clear_buffers()

                elif hand_lm is not None and face_lm is not None:
                    hv, fv, rv = frame_word_features(hand_lm, face_lm)
                    hand_buf.append(hv)
                    face_buf.append(fv)
                    rel_buf.append(rv)

                    if len(hand_buf) == cfg.seq_len:
                        hand_seq = np.stack(list(hand_buf), axis=0)
                        face_seq = np.stack(list(face_buf), axis=0)
                        rel_seq = np.stack(list(rel_buf), axis=0)

                        meta = {
                            "label": current_label,
                            "seq_len": int(cfg.seq_len),
                            "sample_hz": float(cfg.sample_hz),
                            "motion_start_thresh": float(cfg.motion_start_thresh),
                            "face_subset_len": int(len(FACE_SUBSET_IDX)),
                            "timestamp": float(now),
                        }

                        out_path = save_sample_npz(cfg, current_label, hand_seq, face_seq, rel_seq, meta)
                        saved += 1

                        state = CaptureState.COOLDOWN
                        cooldown_until = now + float(cfg.cooldown_sec)
                        countdown_until = now + float(cfg.countdown_sec)
                        clear_buffers()

        # Draw landmarks
        if hand_result is not None:
            hand_extractor.draw(frame, hand_result)
        if face_result is not None:
            face_extractor.draw(frame, face_result, step=6)

        # UI
        h, w = frame.shape[:2]
        x0 = 20
        max_w = int(max(50, w - 40))

        title = "ASL Word Collector (Sequence .npz)"
        line1 = f"Label: {current_label} ({label_idx+1}/{len(cfg.labels)}) | Saved: {saved}"
        rec_line = f"Recording: {'ON' if recording_enabled else 'OFF'} | State: {state} | Motion: {motion_smooth:.3f}"

        help_text = "Keys: R toggle record | N/P label | C clear | ESC quit"

        y = 30
        y = draw_wrapped_block(frame, title, x0, y, max_w, font, 0.75, 1, 3, max_lines=2)
        y += 4
        y = draw_wrapped_block(frame, line1, x0, y, max_w, font, 0.65, 1, 3, max_lines=2)
        y += 2
        y = draw_wrapped_block(frame, rec_line, x0, y, max_w, font, 0.55, 1, 3, max_lines=2)
        y += 2

        if recording_enabled and state == CaptureState.IDLE and countdown_until is not None:
            remain = max(0.0, countdown_until - now)
            y = draw_wrapped_block(frame, f"Countdown: {remain:.1f}s (move to start capture)", x0, y, max_w, font, 0.55, 1, 3, max_lines=2)
            y += 2
        if state == CaptureState.CAPTURING:
            y = draw_wrapped_block(frame, f"Capturing: {len(hand_buf)}/{cfg.seq_len} frames", x0, y, max_w, font, 0.55, 1, 3, max_lines=2)
            y += 2

        y = draw_wrapped_block(frame, help_text, x0, h - 20, max_w, font, 0.55, 1, 3, max_lines=2)

        cv2.imshow("Word Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            recording_enabled = not recording_enabled
        if key in (ord("c"), ord("C")):
            clear_buffers()
        if key in (ord("n"), ord("N")):
            label_idx = (label_idx + 1) % len(cfg.labels)
            current_label = cfg.labels[label_idx]
            clear_buffers()
        if key in (ord("p"), ord("P")):
            label_idx = (label_idx - 1) % len(cfg.labels)
            current_label = cfg.labels[label_idx]
            clear_buffers()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
