"""
member2_video_fusion/inference.py
==================================
Singleton inference wrapper for the trained VideoEmotionModel (v3.1).

Loads fold_2_best.pt once on first call and caches the model in memory.
Provides predict_video() which accepts a raw video file path and returns the
M2 → M3 prediction_output dict (video-only mode, physiological modality missing).

Preprocessing matches TRAINING exactly:
    1. Sample T=16 frames uniformly across the clip (sequential decode).
    2. Resize each frame directly to 224×224 (INTER_AREA) — NO face crop.
    3. ImageNet mean/std normalisation → (T, 3, 224, 224) tensor.

IMPORTANT — why there is no face crop here (empirically verified, not a guess):
    The CREMA-D portion of the v3.1 training .npy frames (the majority of the
    training set) was extracted as a plain full-frame resize. Only the later
    RAVDESS extraction script (training/kaggle_ravdess_extract.py) added a
    YOLO face-crop step, for RAVDESS only. Applying a YOLO crop here creates a
    train/inference distribution mismatch: on the exact held-out test clips
    (fold_2_best.pt, actors reported in test_summary_v2.json), full-frame
    resize scored macro-F1 ≈ 0.70 (matches the reported 0.6615 test F1),
    while YOLO-cropping the same clips dropped it to ≈ 0.56. So this module
    deliberately feeds the model uncropped, resized frames.

    YOLO is still used — but ONLY to compute the video signal-quality grade
    (good / degraded / poor) from face detection rate, Laplacian sharpness, and
    face bounding-box area, using the thresholds in preprocessing/face_detector.py
    (CLAUDE.md, Pech-Pacheco et al. 2000). It never touches the model input.

Usage:
    from member2_video_fusion.inference import predict_video
    output = predict_video("/path/to/clip.mp4")

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from member2_video_fusion.models.video_model import VideoEmotionModel
from member2_video_fusion.preprocessing.multi_corpus_dataset import (
    normalise_and_to_tensor,
    N_FRAMES,
    FRAME_SIZE,
)
from member2_video_fusion.preprocessing.face_detector import (
    _laplacian_variance,
    _grade,
    _worst,
    _DETECT_RATE_GOOD,
    _DETECT_RATE_DEGRADED,
    _LAP_VAR_GOOD,
    _LAP_VAR_DEGRADED,
    _FACE_AREA_GOOD,
    _FACE_AREA_DEGRADED,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_PATH  = Path(__file__).parent / "models"  / "fold_2_best.pt"   # v3.1 best (val 0.6834)
_YOLO_PATH   = Path(__file__).parent / "weights" / "yolov8n-face.pt"
_YOLO_CONF   = 0.30    # used only for signal-quality scoring, not cropping — see module docstring
_K           = 5       # number of emotion classes

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_model:  Optional[VideoEmotionModel] = None
_device: Optional[torch.device]      = None
_yolo                                = None
_yolo_tried                          = False


def _load_model() -> Tuple[VideoEmotionModel, torch.device]:
    global _model, _device
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Video model checkpoint not found at {_MODEL_PATH}. "
                "Download fold_2_best.pt from the Kaggle checkpoint dataset "
                "(vanaiyan/medoracle-v2-checkpoints) into member2_video_fusion/models/."
            )
        device = VideoEmotionModel.get_device()
        model  = VideoEmotionModel(pretrained=False)
        ckpt   = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(device)
        _model  = model
        _device = device
        logger.info(
            "Loaded video model %s (fold %s, epoch %s, val_macro_f1=%.4f) on %s",
            _MODEL_PATH.name, ckpt.get("fold"), ckpt.get("epoch"),
            ckpt.get("val_macro_f1", float("nan")), device,
        )
    return _model, _device


def _load_yolo():
    """Load the YOLO face detector once. Returns None if unavailable (→ full-frame)."""
    global _yolo, _yolo_tried
    if not _yolo_tried:
        _yolo_tried = True
        try:
            from ultralytics import YOLO
            path = str(_YOLO_PATH) if _YOLO_PATH.exists() else "yolov8n-face.pt"
            _yolo = YOLO(path)
            logger.info("Loaded YOLO face detector: %s", path)
        except Exception as exc:  # ultralytics missing or weights unavailable
            logger.warning(
                "YOLO face detector unavailable (%s); falling back to full-frame "
                "resize (predictions will be flagged as poor video quality)", exc
            )
            _yolo = None
    return _yolo


# ---------------------------------------------------------------------------
# Preprocessing — mirrors training/kaggle_ravdess_extract.py
# ---------------------------------------------------------------------------

def _sample_frames_uniform(video_path: Path, n_frames: int = N_FRAMES) -> Optional[np.ndarray]:
    """Sample n_frames uniformly across a video (sequential decode, BGR→RGB).

    Sequential reading (not frame seeking) is used deliberately: H.264 videos use
    infrequent keyframes, so cap.set(POS_FRAMES) silently fails on non-keyframes.
    Returns (n_frames, H, W, 3) uint8 RGB, or None if the clip is unreadable or
    has fewer than n_frames frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < n_frames:
        return None
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    return np.stack([frames[i] for i in indices])


def _face_metrics_only(frame: np.ndarray, yolo) -> Tuple[bool, float]:
    """Run YOLO for QUALITY SCORING ONLY — never used to crop the model input.

    Empirically verified (see git history / dev notes): the CREMA-D portion of
    the v3.1 training .npy frames was extracted as a plain full-frame resize,
    NOT a YOLO face-crop (only the later RAVDESS extraction script used
    YOLO-cropping). Cropping at inference time creates a severe train/inference
    distribution mismatch — on the held-out test set it dropped macro-F1 from
    0.70 (full-frame, matches training) to 0.56 (YOLO-cropped). So the face
    detector here is used purely to report signal_quality; the pixels handed to
    the model are always the uncropped, resized frame.

    Returns (has_face, face_area_ratio).
    """
    if yolo is None:
        return False, 0.0
    results   = yolo.predict(frame, verbose=False, conf=_YOLO_CONF)
    best_area = 0
    H, W = frame.shape[:2]
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
    if best_area == 0:
        return False, 0.0
    return True, best_area / float(H * W) if H * W else 0.0


def _process_video(video_path: Path) -> Tuple[Optional[np.ndarray], str]:
    """Decode → full-frame resize (matches training) → return (frames, quality).

    YOLO face detection runs alongside purely to compute signal_quality
    (detection rate, face-area ratio) — it does NOT crop the frames fed to
    the model. See _face_metrics_only for why.
    """
    raw = _sample_frames_uniform(video_path)
    if raw is None:
        return None, "poor"

    yolo     = _load_yolo()
    resized  = []
    detected = []
    areas    = []
    lap_vars = []
    for frame in raw:
        lap_vars.append(_laplacian_variance(frame))
        has_face, area = _face_metrics_only(frame, yolo)
        detected.append(has_face)
        if has_face:
            areas.append(area)
        resized.append(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA))

    frames        = np.stack(resized)                       # (T, 224, 224, 3)
    detect_rate   = sum(detected) / len(detected)
    mean_lap_var  = float(np.mean(lap_vars))
    mean_area     = float(np.mean(areas)) if areas else 0.0

    grade_detect  = _grade(detect_rate,  _DETECT_RATE_GOOD, _DETECT_RATE_DEGRADED)
    grade_lap     = _grade(mean_lap_var, _LAP_VAR_GOOD,     _LAP_VAR_DEGRADED)
    grade_area    = _grade(mean_area,    _FACE_AREA_GOOD,   _FACE_AREA_DEGRADED)
    quality       = _worst([grade_detect, grade_lap, grade_area])
    return frames, quality


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_video(video_path: str | Path) -> dict:
    """
    Run full video inference on a raw video file.

    Operates in video-only mode (physio missing → graceful degradation):
      - modality_weights = {"physio": 0.0, "video": 1.0}
      - signal_quality.eeg / .gsr = "poor"

    Parameters
    ----------
    video_path : path to any video file readable by OpenCV

    Returns
    -------
    dict : prediction_output matching the M2 → M3 interface contract
    """
    video_path = Path(video_path)

    # 1. Decode + uniform sampling + YOLO face-crop (matches training)
    frames, video_quality = _process_video(video_path)
    if frames is None:
        raise ValueError(
            f"Could not extract {N_FRAMES} frames from {video_path.name}. "
            "The file may be too short, corrupt, or an unsupported format."
        )

    # 2. ImageNet normalise → (T, 3, 224, 224) tensor
    tensor = normalise_and_to_tensor(frames)

    # 3. Model inference
    model, device = _load_model()
    video_pred = model.predict_clip(tensor, device=device)
    # video_pred = {"predicted_emotion": str, "confidence": float, "class_probabilities": dict}

    # 4. Graceful degradation: physio missing → video weight = 1.0
    uniform_probs = {e: 1.0 / _K for e in video_pred["class_probabilities"]}

    return {
        "predicted_emotion":   video_pred["predicted_emotion"],
        "confidence":          video_pred["confidence"],
        "class_probabilities": video_pred["class_probabilities"],
        "modality_weights":    {"physio": 0.0, "video": 1.0},
        "signal_quality": {
            "eeg":   "poor",       # not provided
            "gsr":   "poor",       # not provided
            "video": video_quality,
        },
        "per_modality_predictions": {
            "physio": {
                "predicted_emotion":   "calm",   # placeholder — physio absent
                "confidence":          0.0,
                "class_probabilities": uniform_probs,
            },
            "video": video_pred,
        },
    }


# ---------------------------------------------------------------------------
# Smoke test (requires a real video file path as argv[1])
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m member2_video_fusion.inference <video_file>")
        sys.exit(0)

    out = predict_video(sys.argv[1])
    print(json.dumps(out, indent=2))
