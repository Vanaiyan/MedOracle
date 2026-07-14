from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

WINDOW_SAMPLES = 512   # 4 s × 128 Hz
EPSILON        = 1e-8


class GSRPreprocessor:

    def __init__(self):
        self._mean: Optional[float] = None
        self._std:  Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, gsr_windows: np.ndarray) -> "GSRPreprocessor":

        if gsr_windows.ndim == 1:
            gsr_windows = gsr_windows[np.newaxis, :]   # (1, 512)

        assert gsr_windows.ndim == 2 and gsr_windows.shape[1] == WINDOW_SAMPLES, \
            f"Expected shape (N, 512), got {gsr_windows.shape}"

        # Apply within-window baseline subtraction first (mirrors transform step 1)
        window_mean  = gsr_windows.mean(axis=1, keepdims=True)   # (N, 1)
        gsr_centered = gsr_windows - window_mean                  # (N, 512)

        flat       = gsr_centered.ravel()   # all baseline-corrected samples
        self._mean = float(flat.mean())
        self._std  = float(flat.std())
        if self._std < EPSILON:
            self._std = EPSILON
        self._fitted = True
        return self

    def load_stats(self, mean: float, std: float) -> "GSRPreprocessor":
        self._mean   = float(mean)
        self._std    = float(std) if float(std) >= EPSILON else EPSILON
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, gsr: np.ndarray) -> np.ndarray:

        if not self._fitted:
            raise RuntimeError("GSRPreprocessor not fitted. Call fit() or load_stats() first.")

        single = (gsr.ndim == 1)
        if single:
            gsr = gsr[np.newaxis, :]   # (1, 512)

        gsr = gsr.astype(np.float32)

        # Step 1: remove within-window mean (tonic offset)
        window_mean = gsr.mean(axis=1, keepdims=True)   # (N, 1)
        gsr = gsr - window_mean

        # Step 2: z-score with subject statistics
        gsr = (gsr - self._mean) / self._std

        if single:
            gsr = gsr[0]   # back to (512,)
        return gsr

    def fit_transform(self, gsr_windows: np.ndarray) -> np.ndarray:
        return self.fit(gsr_windows).transform(gsr_windows)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def get_stats(self) -> Tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        return self._mean, self._std

    def save_stats(self, filepath: str) -> None:
        np.savez(filepath, gsr_mean=np.array(self._mean), gsr_std=np.array(self._std))
        print(f"GSR stats saved to {filepath}")

    @classmethod
    def from_saved_stats(cls, filepath: str) -> "GSRPreprocessor":
        data = np.load(filepath)
        instance = cls()
        instance.load_stats(float(data["gsr_mean"]), float(data["gsr_std"]))
        return instance


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def normalize_gsr_batch(
    gsr:  np.ndarray,
    mean: float,
    std:  float,
) -> np.ndarray:
    """Fast stateless normalisation."""
    prep = GSRPreprocessor()
    prep.load_stats(mean, std)
    return prep.transform(gsr)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate GSR: tonic level around 5 µS with phasic fluctuations
    N        = 300
    fake_gsr = rng.normal(loc=5.0, scale=1.5, size=(N, 512)).astype(np.float32)

    prep   = GSRPreprocessor()
    normed = prep.fit_transform(fake_gsr)

    print("GSRPreprocessor self-test:")
    print(f"  Input  — shape: {fake_gsr.shape}, mean: {fake_gsr.mean():.3f}, std: {fake_gsr.std():.3f}")
    print(f"  Output — shape: {normed.shape}, mean: {normed.mean():.4f} (≈0), std: {normed.std():.4f} (≈1)")
    assert normed.shape == fake_gsr.shape
    assert abs(normed.mean()) < 0.1

    # Single-window test
    single = rng.normal(size=(512,)).astype(np.float32)
    out    = prep.transform(single)
    assert out.shape == (512,), f"Single window output shape wrong: {out.shape}"

    print("All checks passed")
