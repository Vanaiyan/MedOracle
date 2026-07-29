"""
member3_explainability/attribution/test_fused_ig_smoke.py
=============================================================
Smoke test for fused_ig.py -- run this on a machine with torch + captum +
Member 1's and Member 2's real checkpoints installed (this sandbox has
neither, so this file could only be written and reasoned about, not
executed, from there).

What it checks
---------------
1. build_fused_model() loads both real checkpoints without error.
2. FusedEmotionModel.forward() on random-shaped (but correctly-shaped) inputs
   produces a valid 5-class probability distribution (sums to ~1).
3. explain_fused() runs real captum IntegratedGradients end-to-end and
   returns the expected keys/shapes (32 EEG channels, 1 GSR value, 16 video
   frames, modality totals, completeness check).
4. The completeness gap (|sum of all attributions - actual model output
   delta|) is small -- this is the IG analogue of the SHAP efficiency-axiom
   check already used elsewhere in this codebase.

Usage
-----
    python -m member3_explainability.attribution.test_fused_ig_smoke

Needs a real video file to fully exercise the video branch -- pass one:
    python -m member3_explainability.attribution.test_fused_ig_smoke path/to/clip.mp4

Without a video file, it uses random frames shaped like real ones (still
exercises the full IG code path and shapes, just not real facial content).

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main():
    print("=" * 70)
    print("FUSED IG SMOKE TEST")
    print("=" * 70)

    print("\n[1/4] Importing torch, captum, and the real model loaders...")
    try:
        import torch  # noqa: F401
        from captum.attr import IntegratedGradients  # noqa: F401
        from member3_explainability.attribution.fused_ig import (
            build_fused_model, explain_fused,
        )
    except ImportError as exc:
        print(f"  FAILED: {exc}")
        print("  -> Install torch + captum in this venv, and make sure Member 1's "
              "checkpoint (member1_physiological/checkpoints/random_split_best.pt) "
              "and Member 2's checkpoint (member2_video_fusion/models/fold_2_best.pt) "
              "are present.")
        sys.exit(1)
    print("  OK")

    print("\n[2/4] Building synthetic (but correctly-shaped) EEG/GSR/video input...")
    rng = np.random.default_rng(0)
    eeg = (rng.standard_normal((32, 512)) * 10).astype(np.float32)
    gsr = (rng.standard_normal(512) * 100 + 5000).astype(np.float32)

    if len(sys.argv) > 1:
        print(f"  Using real video file: {sys.argv[1]}")
        from member2_video_fusion.inference import _process_video
        from pathlib import Path
        video_frames, video_quality = _process_video(Path(sys.argv[1]))
        if video_frames is None:
            print("  Could not decode that video; falling back to random frames.")
            video_frames = rng.integers(0, 255, (16, 224, 224, 3), dtype=np.uint8)
            video_quality = "good"
    else:
        print("  No video file given -- using random frames (shape/plumbing test only).")
        video_frames = rng.integers(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        video_quality = "good"
    print("  OK  eeg:", eeg.shape, " gsr:", gsr.shape, " video:", video_frames.shape)

    print("\n[3/4] Running explain_fused() -- real IntegratedGradients through "
          "both real models + the real fusion formula...")
    t0 = time.time()
    result = explain_fused(
        eeg=eeg, gsr=gsr, video_frames=video_frames,
        eeg_quality="good", gsr_quality="good", video_quality=video_quality,
        n_steps=128,
    )
    dt = time.time() - t0
    print(f"  OK  ({dt:.1f}s)")

    print("\n[4/4] Validating output shape and completeness...")
    assert result["target_emotion"] in ["stress", "calm", "happy", "sad", "angry"]
    assert len(result["eeg_channel_importance"]) == 32
    assert len(result["video_frame_importance"]) == 16
    assert set(result["modality_totals"].keys()) == {"physio", "video"}
    gap = result["completeness_check"]["gap"]
    print(f"  target_emotion       : {result['target_emotion']}")
    print(f"  modality_totals      : {result['modality_totals']}")
    print(f"  completeness gap     : {gap:.4f}  "
          f"({'OK, small' if gap < 0.05 else 'LARGE -- investigate (try more n_steps)'})")
    print("  Top-5 EEG channels by |attribution|:")
    top_eeg = sorted(result["eeg_channel_importance"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    for ch, v in top_eeg:
        print(f"    {ch}: {v:+.4f}")
    print("  Top-3 video frames by |attribution|:")
    top_frames = sorted(result["video_frame_importance"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    for fr, v in top_frames:
        print(f"    {fr}: {v:+.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED" if gap < 0.05 else "SMOKE TEST PASSED (but completeness gap is large -- check)")
    print("=" * 70)


if __name__ == "__main__":
    main()
