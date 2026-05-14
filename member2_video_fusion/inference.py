"""
member2_video_fusion/inference.py
==================================
Singleton inference wrapper for the trained VideoEmotionModel.

Loads fold_3_best.pt once on first call, caches the model in memory.
Provides predict_video() which accepts a raw video file path and returns
the M2 → M3 prediction_output dict (video-only mode, physio missing).

Usage:
    from member2_video_fusion.inference import predict_video
    output = predict_video("/path/to/clip.mp4")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from member2_video_fusion.models.video_model import VideoEmotionModel
from member2_video_fusion.preprocessing.dataset import (
    sample_frames_uniform,
    resize_frames,
    normalise_frames,
    frames_to_tensor,
    assess_video_quality_basic,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(__file__).parent / "models" / "fold_3_best.pt"
_N_FRAMES   = 16
_K          = 5   # number of emotion classes

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_model:  Optional[VideoEmotionModel] = None
_device: Optional[torch.device]      = None


def _load_model() -> tuple[VideoEmotionModel, torch.device]:
    global _model, _device
    if _model is None:
        device = VideoEmotionModel.get_device()
        model  = VideoEmotionModel(pretrained=False)
        ckpt   = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(device)
        _model  = model
        _device = device
    return _model, _device


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

    # 1. Sample T=16 frames uniformly across the clip
    frames = sample_frames_uniform(str(video_path), n_frames=_N_FRAMES)
    if frames is None:
        raise ValueError(
            f"Could not extract {_N_FRAMES} frames from {video_path.name}. "
            "The file may be too short, corrupt, or an unsupported format."
        )

    # 2. Assess quality before resizing (matches training: Laplacian variance only)
    video_quality = assess_video_quality_basic(frames)

    # 3. Resize to 224×224, normalise (ImageNet mean/std), convert to tensor
    #    — matches CREMADDataset.__getitem__ exactly (no face detection was used in training)
    frames = resize_frames(frames)
    normed = normalise_frames(frames)
    tensor = frames_to_tensor(normed)

    # 4. Model inference
    model, device = _load_model()
    video_pred = model.predict_clip(tensor, device=device)
    # video_pred = {"predicted_emotion": str, "confidence": float, "class_probabilities": dict}

    # 5. Graceful degradation: physio missing → video weight = 1.0
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
                "predicted_emotion":   "calm",
                "confidence":          0.0,
                "class_probabilities": uniform_probs,
            },
            "video": video_pred,
        },
    }
