"""
member1_physiological/preprocessing/de_features.py
==================================================
Differential Entropy (DE) frequency-band features for EEG + GSR.

DE is the standard, well-generalising feature for DEAP emotion recognition
(Duan et al., 2013; Zheng & Lu, 2015). For a band-limited Gaussian signal,

    DE = 0.5 * log(2 * pi * e * variance)   ≈  0.5 * log(band power) + const

i.e. DE is essentially *log* band power. The log makes the heavy-tailed band
power roughly Gaussian, which is far easier for a classifier to separate and
generalises across subjects much better than raw band power (what the SVM/RF
baselines currently use) or the raw time-domain signal (what PhysiologicalNet
currently uses — and which collapses to a single class under LOSO).

Feature vector (per 4-second window)
------------------------------------
    EEG : 32 channels × 5 bands (delta/theta/alpha/beta/gamma) = 160 DE features
    GSR : mean, std, min, max, slope, mean|Δ|, std(Δ)          =   7 features
    Total                                                       = 167 features

Author: MedOracle (Member 1 module — feature improvement)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

FS = 128
BANDS = [
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 45.0),
]
N_BANDS   = len(BANDS)
N_EEG_DE  = 32 * N_BANDS      # 160
N_GSR     = 7
N_FEATURES = N_EEG_DE + N_GSR  # 167
_EPS = 1e-8


def eeg_de(eeg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Differential entropy per channel per band.

    Parameters
    ----------
    eeg : (C, T) EEG window (C channels, T samples)

    Returns
    -------
    (C, 5) DE values.
    """
    nperseg = min(256, eeg.shape[-1])
    freqs, psd = welch(eeg, fs=fs, nperseg=nperseg, axis=-1)   # psd: (C, F)
    out = np.empty((eeg.shape[0], N_BANDS), dtype=np.float32)
    for i, (_, lo, hi) in enumerate(BANDS):
        idx = (freqs >= lo) & (freqs < hi)
        band_power = psd[:, idx].mean(axis=1) if idx.any() else np.zeros(eeg.shape[0])
        out[:, i] = 0.5 * np.log(2.0 * np.pi * np.e * (band_power + _EPS))
    return out


def gsr_features(gsr: np.ndarray) -> np.ndarray:
    """7 robust GSR features from one window.

    DEAP's channel-36 GSR is corrupted (non-physical negatives, huge artifact
    spikes up to ±200k, and inconsistent scaling across subjects). Raw mean/
    min/max/std are dominated by artifacts, so we (1) winsorize the window to its
    own 5th–95th percentile to remove spikes, (2) z-score it to make subjects
    comparable, then (3) take robust statistics. NOTE: measured impact on LOSO
    macro-F1 is ~nil (GSR carries little 5-class signal here) — this is data
    hygiene, not a performance lever.
    """
    lo, hi = np.percentile(gsr, 5), np.percentile(gsr, 95)
    g = np.clip(gsr, lo, hi)
    g = (g - g.mean()) / (g.std() + _EPS)          # per-window z-score
    t = np.arange(len(g))
    slope = float(np.polyfit(t, g, 1)[0]) if len(g) > 1 else 0.0
    diff = np.diff(g)
    return np.array([
        float(np.median(g)), float(g.std()),
        float(np.percentile(g, 25)), float(np.percentile(g, 75)),
        slope, float(np.abs(diff).mean()), float(diff.std()),
    ], dtype=np.float32)


def extract_de_features(eeg: np.ndarray, gsr: np.ndarray) -> np.ndarray:
    """
    Extract the 167-dim DE feature vector from one window.

    Parameters
    ----------
    eeg : (32, 512)   GSR : (512,)

    Returns
    -------
    (167,) float32 feature vector.
    """
    de = eeg_de(eeg).reshape(-1)          # 160
    g  = gsr_features(gsr)                 # 7
    return np.concatenate([de, g]).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    f = extract_de_features(rng.standard_normal((32, 512)).astype(np.float32),
                            rng.standard_normal(512).astype(np.float32))
    print(f"DE feature vector shape: {f.shape}  (expected ({N_FEATURES},))")
    assert f.shape == (N_FEATURES,)
    print("✓ de_features OK")
