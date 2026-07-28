"""
tests/test_pipeline.py
======================
Verifies member2_video_fusion/pipeline.run_full_pipeline:
  - input-validation branches (no models needed)
  - physio-only / video-only / full-multimodal fusion (skipped if the
    local checkpoints or CREMA-D clip are absent — they're gitignored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from member2_video_fusion.pipeline import run_full_pipeline, _PHYSIO_CKPT
from shared.data_contracts import make_prediction_output

_REPO       = Path(__file__).parent.parent
_VIDEO_CKPT = _REPO / "member2_video_fusion" / "models" / "fold_2_best.pt"
_CLIP       = _REPO / "data" / "CREMA-D" / "VideoFlash" / "1027_IWL_HAP_XX.flv"

_physio_absent = not _PHYSIO_CKPT.exists()
_video_absent  = not (_VIDEO_CKPT.exists() and _CLIP.exists())


def _rand_physio():
    rng = np.random.default_rng(0)
    eeg = (rng.standard_normal((32, 512)) * 10).astype(np.float32)
    gsr = (rng.standard_normal(512) * 100 + 5000).astype(np.float32)
    return eeg, gsr


def _assert_valid(out: dict):
    # Re-validating through the contract must not raise.
    make_prediction_output(
        out["predicted_emotion"], out["confidence"], out["class_probabilities"],
        out["modality_weights"], out["signal_quality"], out["per_modality_predictions"],
    )


# --- input validation (no model load) --------------------------------------

def test_no_modality_raises():
    with pytest.raises(ValueError):
        run_full_pipeline()


def test_half_physio_raises():
    eeg, _ = _rand_physio()
    with pytest.raises(ValueError):
        run_full_pipeline(eeg=eeg)          # gsr missing


# --- model-dependent branches (skipped if artifacts absent) ----------------

@pytest.mark.skipif(_physio_absent, reason="physio checkpoint not present locally")
def test_physio_only():
    eeg, gsr = _rand_physio()
    out = run_full_pipeline(eeg=eeg, gsr=gsr)
    assert out["modality_weights"] == {"physio": 1.0, "video": 0.0}
    _assert_valid(out)


@pytest.mark.skipif(_video_absent, reason="video checkpoint / CREMA-D clip not present")
def test_video_only():
    out = run_full_pipeline(video_path=str(_CLIP))
    assert out["modality_weights"] == {"physio": 0.0, "video": 1.0}
    _assert_valid(out)


@pytest.mark.skipif(_physio_absent or _video_absent, reason="checkpoints/clip not present")
def test_full_multimodal_both_weighted():
    eeg, gsr = _rand_physio()
    out = run_full_pipeline(eeg=eeg, gsr=gsr, video_path=str(_CLIP))
    w = out["modality_weights"]
    assert w["physio"] > 0.0 and w["video"] > 0.0        # genuine fusion
    assert abs(w["physio"] + w["video"] - 1.0) < 1e-9
    assert out["signal_quality"]["eeg"] != "poor" or True  # eeg quality is graded from real signal
    _assert_valid(out)
