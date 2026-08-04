from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from member2_video_fusion.fusion import gated_fusion
from member2_video_fusion.inference import predict_video_modality, predict_video_frames
from shared.data_contracts import SynchronizedInput

logger = logging.getLogger(__name__)

# Physiological checkpoint (random-split demo model for now — see module docstring)
_PHYSIO_CKPT = (
    Path(__file__).parent.parent
    / "member1_physiological" / "checkpoints" / "random_split_best.pt"
)

_physio_predictor = None   # lazy singleton


def _get_physio_predictor():
    """Load Member 1's PhysiologicalPredictor once, cache it in memory."""
    global _physio_predictor
    if _physio_predictor is None:
        from member1_physiological.predict import PhysiologicalPredictor
        if not _PHYSIO_CKPT.exists():
            raise FileNotFoundError(
                f"Physio checkpoint not found at {_PHYSIO_CKPT}. "
                "Member 1 must provide a trained checkpoint (random_split_best.pt "
                "or a LOSO fold checkpoint)."
            )
        _physio_predictor = PhysiologicalPredictor(str(_PHYSIO_CKPT), device="auto")
        logger.info("Loaded physio predictor: %s", _PHYSIO_CKPT.name)
    return _physio_predictor


def run_full_pipeline(
    eeg:           Optional[np.ndarray] = None,
    gsr:           Optional[np.ndarray] = None,
    video_path:    Optional[Union[str, Path]] = None,
    video_frames:  Optional[np.ndarray] = None,
    video_quality: str = "good",
) -> dict:
    """
    End-to-end multimodal inference → fused `prediction_output`.

    Provide physio (both `eeg` and `gsr`) and/or video (`video_path` OR
    `video_frames`). At least one modality is required; missing modalities
    degrade gracefully via `fusion.gated_fusion`.

    Parameters
    ----------
    eeg : (32, 512) np.ndarray or None      — raw EEG window
    gsr : (512,)   np.ndarray or None       — raw GSR window
    video_path   : path to a raw video file (decoded + quality-graded), or None
    video_frames : pre-extracted (16, 224, 224, 3) uint8 frames, or None
                   (paired with `video_quality`; e.g. from a SynchronizedInput)
    video_quality : quality grade to attach when `video_frames` is supplied
                    (ignored when `video_path` is used — quality is measured there)

    Returns
    -------
    dict — prediction_output validated by shared.data_contracts.

    Raises
    ------
    ValueError if no modality is supplied, or if exactly one of eeg/gsr is given.
    """
    # --- physiological modality (M1) ---
    physio_pred = None
    if eeg is not None and gsr is not None:
        physio_pred = _get_physio_predictor().predict(eeg, gsr)
    elif (eeg is None) != (gsr is None):
        raise ValueError("Provide BOTH eeg and gsr, or neither.")

    # --- video modality (M2) ---
    video_pred, vq = None, "poor"
    if video_path is not None:
        video_pred, vq = predict_video_modality(video_path)
    elif video_frames is not None:
        video_pred, vq = predict_video_frames(video_frames), video_quality

    if physio_pred is None and video_pred is None:
        raise ValueError(
            "run_full_pipeline needs at least one modality "
            "(eeg+gsr, video_path, or video_frames)."
        )

    # --- gated fusion (canonical) ---
    return gated_fusion(physio_pred=physio_pred, video_pred=video_pred, video_quality=vq)


def run_from_synchronized(si: SynchronizedInput) -> dict:
    """Run the full pipeline from a validated `SynchronizedInput` (all 3 modalities)."""
    si.validate()
    return run_full_pipeline(
        eeg=si.eeg, gsr=si.gsr, video_frames=si.video, video_quality="good"
    )


# ---------------------------------------------------------------------------
# Smoke test:  python -m member2_video_fusion.pipeline [video_file]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)
    video = sys.argv[1] if len(sys.argv) > 1 else None

    # Synthetic physio window (real DEAP data would replace this)
    rng = np.random.default_rng(0)
    eeg = (rng.standard_normal((32, 512)) * 10).astype(np.float32)
    gsr = (rng.standard_normal(512) * 100 + 5000).astype(np.float32)

    print("=== physio only ===")
    out = run_full_pipeline(eeg=eeg, gsr=gsr)
    print(f"fused: {out['predicted_emotion']}  weights={out['modality_weights']}")

    if video:
        print("\n=== FULL multimodal (physio + video) ===")
        out = run_full_pipeline(eeg=eeg, gsr=gsr, video_path=video)
        print(f"fused: {out['predicted_emotion']}  weights={out['modality_weights']}")
        print(f"signal_quality: {out['signal_quality']}")
        print(json.dumps(out["class_probabilities"], indent=2))
    else:
        print("\n(pass a video file path to test full multimodal fusion)")

    print("\n✓ pipeline smoke test complete")
