"""
member1_physiological/samples/
==============================
Bundled physiological sample windows for the multimodal web demo.

Each sample is a single 4-second window: EEG (32, 512) + GSR (512,), matching the
input shapes expected by `PhysiologicalPredictor.predict`. There is one sample
per emotion so the web app can pair an uploaded video with a physiological
"reading" and demonstrate real gated fusion.

Data source
-----------
The bundled `.npz` files are **real DEAP windows** — one genuine EEG+GSR window
per emotion, chosen as the clearest ground-truth example (largest V/A/D margin),
extracted by `extract_deap_samples.py`. Each file also stores provenance
(subject, trial, valence/arousal/dominance) — see `sample_info()`.

To (re)generate from DEAP:  `python -m member1_physiological.samples.extract_deap_samples`
(needs data/DEAP/s01.dat … s32.dat). `generate_placeholder_samples.py` remains as
a synthetic fallback for machines without the DEAP data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLES_DIR = Path(__file__).parent
AVAILABLE_EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]


def sample_path(emotion: str) -> Path:
    return SAMPLES_DIR / f"{emotion.lower()}.npz"


def load_sample(emotion: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a bundled physio window.

    Returns
    -------
    (eeg, gsr) : eeg (32, 512) float32, gsr (512,) float32
    """
    path = sample_path(emotion)
    if not path.exists():
        raise FileNotFoundError(
            f"No physio sample for '{emotion}'. Available: {list_available()}"
        )
    data = np.load(path)
    return data["eeg"].astype(np.float32), data["gsr"].astype(np.float32)


def list_available() -> list[str]:
    """Return the emotions that currently have a bundled sample file."""
    return [e for e in AVAILABLE_EMOTIONS if sample_path(e).exists()]


def sample_info(emotion: str) -> dict:
    """Return provenance metadata for a bundled sample (empty for placeholders).

    Keys (when present): subject, trial, valence, arousal, dominance.
    """
    path = sample_path(emotion)
    if not path.exists():
        raise FileNotFoundError(f"No physio sample for '{emotion}'.")
    data = np.load(path, allow_pickle=True)
    keys = ("subject", "trial", "valence", "arousal", "dominance")
    return {k: data[k].item() for k in keys if k in data.files}
