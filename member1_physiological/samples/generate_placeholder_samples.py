"""
member1_physiological/samples/generate_placeholder_samples.py
=============================================================
Generate PLACEHOLDER synthetic physiological sample windows (one per emotion),
so the multimodal web demo works before real DEAP data is available.

Each file is a 4-second window: EEG (32, 512) + GSR (512,), the exact shapes
`PhysiologicalPredictor.predict` expects. Values are synthetic (Gaussian with a
per-emotion offset) — the physio model's prediction on them is NOT meaningful.

Replace these with real DEAP windows later (see samples/__init__.py docstring).

Run:
    python -m member1_physiological.samples.generate_placeholder_samples
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]
_OUT = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(42)
    for i, emotion in enumerate(EMOTIONS):
        # Per-emotion offsets so the samples differ; still synthetic placeholders.
        eeg = (rng.standard_normal((32, 512)) * (8.0 + i * 1.5)).astype(np.float32)
        gsr = (rng.standard_normal(512) * 80.0 + 4000.0 + i * 250.0).astype(np.float32)
        out_path = _OUT / f"{emotion}.npz"
        np.savez(out_path, eeg=eeg, gsr=gsr)
        print(f"  wrote {out_path.name}  eeg{eeg.shape} gsr{gsr.shape}")
    print(f"\n✓ {len(EMOTIONS)} placeholder physio samples written to {_OUT}")
    print("  (synthetic — swap in real DEAP windows when available)")


if __name__ == "__main__":
    main()
