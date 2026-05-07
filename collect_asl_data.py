from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Feature helpers
# ============================================================

def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks for scale/translation invariance.

    @param lm: Raw landmarks array of shape (21, 3).
    @return: Normalized landmarks array of shape (21, 3).
    """
    wrist = lm[0, :2].copy()
    xy = lm[:, :2] - wrist
    scale = np.max(np.linalg.norm(xy, axis=1)) + 1e-6
    xy = xy / scale
    z = lm[:, 2:3]
    return np.concatenate([xy, z], axis=1)


def flatten_features(lm_norm: np.ndarray) -> np.ndarray:
    """
    Flatten normalized landmarks into a single feature vector.

    @param lm_norm: Normalized landmarks array of shape (21, 3).
    @return: Feature vector of shape (63,).
    """
    return lm_norm.reshape(-1)


def format_label_for_display(label: str) -> str:
    """
    Convert an internal label into a readable overlay label.

    @param label: Internal dataset label.
    @return: Display string.
    """
    if label == " ":
        return "[space]"
    return label


def draw_outlined_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    fg: Tuple[int, int, int] = (255, 255, 255),
    outline: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    outline_thickness: int = 3,
) -> None:
    """
    Draw readable outlined text on a frame.

    @param frame: OpenCV BGR frame.
    @param text: Text to render.
    @param org: Bottom-left corner of the text.
    @param font_scale: Font scale multiplier.
    @param fg: Foreground text color.
    @param outline: Outline color.
    @param thickness: Foreground thickness.
    @param outline_thickness: Outline thickness.
    @return: None.
    """
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
# MediaPipe Tasks Hand Landmarker
# ============================================================

@dataclass
class TaskHandExtractorConfig:
    """
    Configuration for the MediaPipe Tasks Hand Landmarker.

    @param model_path: Path to the .task model file.
    @param max_hands: Maximum number of hands to detect.
    @param min_hand_detection_confidence: Minimum detection confidence.
    @param min_hand_presence_confidence: Minimum hand presence confidence.
    @param min_tracking_confidence: Minimum tracking confidence.
    """
    model_path: str = os.path.join("models", "hand_landmarker.task")
    max_hands: int = 1
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class HandLandmarkExtractor:
    """
    Extract hand landmarks using MediaPipe Tasks Hand Landmarker.

    @param cfg: TaskHandExtractorConfig settings.
    """

    def __init__(self, cfg: TaskHandExtractorConfig) -> None:
        """
        Initialize the Tasks hand landmarker.

        @param cfg: TaskHandExtractorConfig settings.
        @raises FileNotFoundError: If cfg.model_path does not exist.
        """
        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(
                f"Missing model file: {cfg.model_path}. "
                "Download a real hand_landmarker.task into the models/ folder."
            )

        base_options = python.BaseOptions(model_asset_path=cfg.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
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
        Draw landmarks on the frame.

        @param frame_bgr: OpenCV BGR frame.
        @param result: HandLandmarkerResult from Tasks.
        """
        if not getattr(result, "hand_landmarks", None):
            return

        h, w = frame_bgr.shape[:2]
        for hand in result.hand_landmarks:
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Collect labeled landmark features for a demo dataset.

    Controls:
    - Press A-Z to set the current label.
    - Press '=' to set the special label '='.
    - Press apostrophe (') to set the current label to an actual space character ' '.
    - Press SPACE to save one sample for the current label.
    - Press ESC to quit.

    Output:
    - Writes data/asl_landmarks.csv with columns: label, f0..f62
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    out_path = os.path.join("data", "asl_landmarks.csv")

    extractor = HandLandmarkExtractor(TaskHandExtractorConfig(max_hands=1))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    current_label = "A"
    saved = 0

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            header = ["label"] + [f"f{i}" for i in range(63)]
            w.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            lm, result = extractor.extract(frame)
            if result is not None:
                extractor.draw(frame, result)

            shown_label = format_label_for_display(current_label)

            draw_outlined_text(frame, f"Label: {shown_label}", (20, 35), font_scale=0.9)
            draw_outlined_text(
                frame,
                "A-Z: letters | =: equals sign | ': space",
                (20, 70),
                font_scale=0.55,
            )
            draw_outlined_text(
                frame,
                f"Samples saved: {saved} | SPACE: save | ESC: quit",
                (20, 100),
                font_scale=0.6,
            )

            cv2.imshow("Collect ASL Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if 65 <= key <= 90:
                current_label = chr(key)
            elif 97 <= key <= 122:
                current_label = chr(key).upper()
            elif key == ord("="):
                current_label = "="
            elif key == ord("'"):
                current_label = " "
            elif key == 32:
                if lm is None:
                    print("No hand detected; sample not saved.")
                    continue

                feats = flatten_features(normalize_landmarks(lm))
                w.writerow([current_label] + [float(x) for x in feats])
                saved += 1
                print(f"Saved sample #{saved} for label {repr(current_label)}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Dataset saved to: {out_path}")


if __name__ == "__main__":
    main()