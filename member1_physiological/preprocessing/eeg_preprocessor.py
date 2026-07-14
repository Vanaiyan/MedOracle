from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------

EEG_CHANNELS   = 32
WINDOW_SAMPLES = 512   # 4 s × 128 Hz
EPSILON        = 1e-8  # avoid division by zero in z-score


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EEGPreprocessor:

    def __init__(self):
        self._mean: Optional[np.ndarray] = None   # shape (32,)
        self._std:  Optional[np.ndarray] = None   # shape (32,)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit (training only)
    # ------------------------------------------------------------------

    def fit(self, eeg_windows: np.ndarray) -> "EEGPreprocessor":

        if eeg_windows.ndim == 2:
            # Single window (32, 512) → add batch dimension
            eeg_windows = eeg_windows[np.newaxis, ...]

        assert eeg_windows.ndim == 3 and eeg_windows.shape[1] == EEG_CHANNELS, \
            f"Expected shape (N, 32, 512), got {eeg_windows.shape}"

        # Step 1: apply within-window baseline subtraction (same as in transform)
        # We MUST compute statistics on baseline-subtracted data to match transform()
        window_mean = eeg_windows.mean(axis=2, keepdims=True)  # (N, 32, 1)
        eeg_centered = eeg_windows - window_mean               # (N, 32, 512) — baseline removed

        # Flatten across windows and time: shape (32, N*512)
        flat = eeg_centered.transpose(1, 0, 2).reshape(EEG_CHANNELS, -1)

        self._mean = flat.mean(axis=1)   # (32,)
        self._std  = flat.std(axis=1)    # (32,)
        self._std  = np.where(self._std < EPSILON, EPSILON, self._std)  # avoid /0
        self._fitted = True
        return self

    def load_stats(self, mean: np.ndarray, std: np.ndarray) -> "EEGPreprocessor":

        assert mean.shape == (EEG_CHANNELS,), f"mean must be shape (32,), got {mean.shape}"
        assert std.shape  == (EEG_CHANNELS,), f"std must be shape (32,), got {std.shape}"
        self._mean   = mean.copy()
        self._std    = np.where(std < EPSILON, EPSILON, std.copy())
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, eeg: np.ndarray) -> np.ndarray:

        if not self._fitted:
            raise RuntimeError(
                "EEGPreprocessor has not been fitted. "
                "Call fit() or load_stats() first."
            )

        single = (eeg.ndim == 2)
        if single:
            eeg = eeg[np.newaxis, ...]   # (1, 32, 512)

        eeg = eeg.astype(np.float32)

        # Step 1: subtract within-window per-channel mean
        # This removes slow DC drifts specific to this 4-second segment
        # Shape operations: mean over axis=2 (time) → (N, 32) → unsqueeze → (N, 32, 1)
        window_mean = eeg.mean(axis=2, keepdims=True)   # (N, 32, 1)
        eeg = eeg - window_mean

        # Step 2 & 3: z-score using subject-level statistics
        # _mean shape (32,) → reshape to (1, 32, 1) for broadcasting
        mean = self._mean.astype(np.float32)[np.newaxis, :, np.newaxis]  # (1, 32, 1)
        std  = self._std.astype(np.float32)[np.newaxis, :, np.newaxis]   # (1, 32, 1)
        eeg  = (eeg - mean) / std

        if single:
            eeg = eeg[0]  # back to (32, 512)

        return eeg

    def fit_transform(self, eeg_windows: np.ndarray) -> np.ndarray:
        """Convenience: fit on eeg_windows then return transformed version."""
        return self.fit(eeg_windows).transform(eeg_windows)

    # ------------------------------------------------------------------
    # Save / load stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:

        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return self._mean.copy(), self._std.copy()

    def save_stats(self, filepath: str) -> None:
        """Save statistics to a .npz file for loading at inference time."""
        mean, std = self.get_stats()
        np.savez(filepath, eeg_mean=mean, eeg_std=std)
        print(f"EEG stats saved to {filepath}")

    @classmethod
    def from_saved_stats(cls, filepath: str) -> "EEGPreprocessor":
        """Load a previously saved EEGPreprocessor from a .npz file."""
        data = np.load(filepath)
        instance = cls()
        instance.load_stats(data["eeg_mean"], data["eeg_std"])
        return instance


# ---------------------------------------------------------------------------
# Convenience function (stateless — normalises a batch using given stats)
# ---------------------------------------------------------------------------

def normalize_eeg_batch(
    eeg: np.ndarray,
    mean: np.ndarray,
    std:  np.ndarray,
) -> np.ndarray:

    prep = EEGPreprocessor()
    prep.load_stats(mean, std)
    return prep.transform(eeg)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate one subject's worth of EEG windows (N, 32, 512)
    N = 300
    fake_eeg = rng.normal(loc=50, scale=30, size=(N, 32, 512)).astype(np.float32)

    prep = EEGPreprocessor()
    normed = prep.fit_transform(fake_eeg)

    print("EEGPreprocessor self-test:")
    print(f"  Input  — shape: {fake_eeg.shape}, mean: {fake_eeg.mean():.2f}, std: {fake_eeg.std():.2f}")
    print(f"  Output — shape: {normed.shape}, mean: {normed.mean():.4f} (≈0), std: {normed.std():.4f} (≈1)")
    assert normed.shape == fake_eeg.shape, "Shape mismatch!"
    assert abs(normed.mean()) < 0.1,       "Mean not close to 0"
    assert 0.8 < normed.std() < 1.2,       "Std not close to 1"

    # Test single-window inference
    single = rng.normal(size=(32, 512)).astype(np.float32)
    out    = prep.transform(single)
    assert out.shape == (32, 512), f"Single window output shape wrong: {out.shape}"

    print("All checks passed")
