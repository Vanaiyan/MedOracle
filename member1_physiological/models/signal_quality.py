from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np
from scipy.signal import find_peaks

QualityLevel = Literal["good", "degraded", "poor"]

SAMPLE_RATE    = 128   # Hz
WINDOW_SAMPLES = 512   # 4 s × 128 Hz

# EEG thresholds (µV — applies to raw or scaled signal)
EEG_AMP_GOOD      = 100.0   # µV — max absolute amplitude for "good"
EEG_AMP_DEGRADED  = 150.0   # µV — threshold for "poor"
EEG_FLAT_GOOD     = 0       # flat-line channels for "good"
EEG_FLAT_DEGRADED = 2       # max flat-line channels for "degraded"
EEG_NAN_GOOD      = 0.01    # fraction — < 1% NaN for "good"
EEG_NAN_DEGRADED  = 0.05    # fraction — < 5% NaN for "degraded"

# GSR thresholds (µS)
GSR_RANGE_GOOD_LO   = 0.5    # µS  lower bound for "good" range
GSR_RANGE_GOOD_HI   = 30.0   # µS  upper bound for "good" range
GSR_RANGE_DEG_LO    = 0.1    # µS  lower bound for "degraded" range
GSR_RANGE_DEG_HI    = 50.0   # µS  upper bound for "degraded" range
GSR_PEAKS_GOOD      = 2      # minimum SCR peaks for "good"
GSR_PEAKS_DEGRADED  = 1      # minimum SCR peaks for "degraded"
GSR_DRIFT_GOOD      = 0.1    # µS/s — slope magnitude for "good"
GSR_DRIFT_DEGRADED  = 0.5    # µS/s — slope magnitude for "degraded"

# Flat-line detection: std < this threshold → flat channel
EEG_FLAT_STD_THRESHOLD = 1e-4


# ---------------------------------------------------------------------------
# EEG quality
# ---------------------------------------------------------------------------

def assess_eeg_quality(eeg: np.ndarray, raw: bool = True) -> QualityLevel:
    assert eeg.shape == (32, WINDOW_SAMPLES), \
        f"Expected EEG shape (32, 512), got {eeg.shape}"

    grades = []

    # ---- Metric 1: NaN ratio ----
    nan_ratio = float(np.isnan(eeg).mean())
    if nan_ratio > EEG_NAN_DEGRADED:
        grades.append("poor")
    elif nan_ratio > EEG_NAN_GOOD:
        grades.append("degraded")
    else:
        grades.append("good")

    # ---- Metric 2: Amplitude range (only meaningful on raw µV data) ----
    if raw:
        max_abs = float(np.nanmax(np.abs(eeg)))
        if max_abs > EEG_AMP_DEGRADED:
            grades.append("poor")
        elif max_abs > EEG_AMP_GOOD:
            grades.append("degraded")
        else:
            grades.append("good")

    # ---- Metric 3: Flat-line channels ----
    # A channel is "flat" if its std is near zero (constant signal → broken electrode)
    channel_stds = np.nanstd(eeg, axis=1)   # (32,)
    n_flat = int(np.sum(channel_stds < EEG_FLAT_STD_THRESHOLD))
    if n_flat > EEG_FLAT_DEGRADED:
        grades.append("poor")
    elif n_flat > EEG_FLAT_GOOD:
        grades.append("degraded")
    else:
        grades.append("good")

    return _worst_grade(grades)


def get_eeg_quality_details(eeg: np.ndarray, raw: bool = True) -> Dict:
    """Return detailed quality metrics dictionary for debugging."""
    nan_ratio  = float(np.isnan(eeg).mean())
    max_abs    = float(np.nanmax(np.abs(eeg))) if raw else None
    stds       = np.nanstd(eeg, axis=1)
    n_flat     = int(np.sum(stds < EEG_FLAT_STD_THRESHOLD))
    flat_chs   = list(np.where(stds < EEG_FLAT_STD_THRESHOLD)[0].tolist())
    overall    = assess_eeg_quality(eeg, raw=raw)
    return {
        "overall"        : overall,
        "nan_ratio"      : nan_ratio,
        "max_abs_amp_uV" : max_abs,
        "n_flat_channels": n_flat,
        "flat_channel_ids": flat_chs,
    }


# ---------------------------------------------------------------------------
# GSR quality
# ---------------------------------------------------------------------------

def assess_gsr_quality(gsr: np.ndarray, use_raw_units: bool = True) -> QualityLevel:
    """
    Assess quality of a single GSR window.

    Parameters
    ----------
    gsr : np.ndarray, shape (512,)
        GSR signal. In the DEAP preprocessed dataset this is in raw
        amplifier units (NOT µS). Use use_raw_units=True for this case.
        The checks are adjusted accordingly.
    use_raw_units : bool
        True  → use statistical/relative checks (DEAP preprocessed data)
        False → use absolute µS range checks (only for truly converted data)
    """
    assert gsr.shape == (WINDOW_SAMPLES,), \
        f"Expected GSR shape (512,), got {gsr.shape}"

    grades = []

    # ---- Metric 1: Signal validity ----
    # Instead of µS range, check for flat-line or all-NaN
    if np.isnan(gsr).all():
        return "poor"

    nan_ratio = float(np.isnan(gsr).mean())
    if nan_ratio > 0.05:
        grades.append("poor")
    elif nan_ratio > 0.01:
        grades.append("degraded")
    else:
        grades.append("good")

    if use_raw_units:
        # Statistical check: signal should have meaningful variance
        # A flat-line or saturated GSR has near-zero std
        gsr_std = float(np.nanstd(gsr))
        gsr_range = float(np.nanmax(gsr) - np.nanmin(gsr))
        if gsr_std < 1e-6 or gsr_range < 1e-6:
            grades.append("poor")   # completely flat — sensor failure
        else:
            grades.append("good")
    else:
        # Original µS-based range check (for truly converted data)
        gsr_min = float(np.nanmin(gsr))
        gsr_max = float(np.nanmax(gsr))
        if gsr_min < GSR_RANGE_DEG_LO or gsr_max > GSR_RANGE_DEG_HI:
            grades.append("poor")
        elif gsr_min < GSR_RANGE_GOOD_LO or gsr_max > GSR_RANGE_GOOD_HI:
            grades.append("degraded")
        else:
            grades.append("good")

    # ---- Metric 2: SCR peaks (relative detection) ----
    # Use deviation from local median instead of absolute height threshold
    median_val = float(np.nanmedian(gsr))
    gsr_std    = float(np.nanstd(gsr))
    # Peak must be at least 0.5 std above median to count
    peak_height_threshold = median_val + 0.5 * gsr_std
    peaks, _ = find_peaks(
        gsr,
        height   = peak_height_threshold,
        distance = SAMPLE_RATE,   # at least 1s between peaks
    )
    n_peaks = len(peaks)
    if n_peaks < GSR_PEAKS_DEGRADED:
        grades.append("poor")
    elif n_peaks < GSR_PEAKS_GOOD:
        grades.append("degraded")
    else:
        grades.append("good")

    # ---- Metric 3: Drift (relative) ----
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    slope     = _linear_slope(t, gsr)
    gsr_std   = float(np.nanstd(gsr))
    # Normalise slope by signal std so it's unit-agnostic
    rel_slope = abs(slope) / (gsr_std + 1e-8)
    if rel_slope > 0.5:
        grades.append("poor")
    elif rel_slope > 0.1:
        grades.append("degraded")
    else:
        grades.append("good")

    return _worst_grade(grades)

def get_gsr_quality_details(gsr: np.ndarray) -> Dict:
    t       = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    slope   = _linear_slope(t, gsr)
    gsr_std = float(np.nanstd(gsr))
    median  = float(np.nanmedian(gsr))
    peaks, _ = find_peaks(gsr, height=median + 0.5*gsr_std, distance=SAMPLE_RATE)
    overall = assess_gsr_quality(gsr, use_raw_units=True)
    return {
        "overall"        : overall,
        "gsr_min"        : float(np.nanmin(gsr)),   # raw units, not µS
        "gsr_max"        : float(np.nanmax(gsr)),
        "gsr_std"        : gsr_std,
        "n_scr_peaks"    : len(peaks),
        "drift_rel_slope": float(abs(slope) / (gsr_std + 1e-8)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GRADE_ORDER = {"poor": 2, "degraded": 1, "good": 0}
_GRADE_NAMES = ["good", "degraded", "poor"]

def _worst_grade(grades: list) -> QualityLevel:
    """Return the worst grade from a list of grade strings."""
    worst = max(grades, key=lambda g: _GRADE_ORDER[g])
    return worst

def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Compute slope of OLS linear fit y = slope * t + intercept."""
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0
    t_v, y_v = t[valid], y[valid]
    # slope = cov(t, y) / var(t)
    t_mean = t_v.mean()
    y_mean = y_v.mean()
    slope  = float(((t_v - t_mean) * (y_v - y_mean)).sum() /
                   ((t_v - t_mean) ** 2).sum())
    return slope


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Good EEG: amplitudes within ±100 µV, no NaN, no flat channels
    good_eeg = rng.uniform(-80, 80, size=(32, 512)).astype(np.float32)
    assert assess_eeg_quality(good_eeg, raw=True) == "good", "Should be good"

    # Poor EEG: flat channels
    bad_eeg = good_eeg.copy()
    bad_eeg[0, :] = 0.0
    bad_eeg[1, :] = 0.0
    bad_eeg[2, :] = 0.0
    assert assess_eeg_quality(bad_eeg, raw=True) == "poor", "3 flat channels → poor"

    # Good GSR: 5 µS, multiple peaks, minimal drift
    t         = np.linspace(0, 4, 512)
    good_gsr  = (5.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t) +
                 0.3 * np.sin(2 * np.pi * 1.0 * t)).astype(np.float32)
    q = assess_gsr_quality(good_gsr)
    print(f"Good GSR quality: {q}")

    # Poor GSR: signal out of range
    bad_gsr = np.full(512, 0.05, dtype=np.float32)   # < 0.1 µS
    assert assess_gsr_quality(bad_gsr) == "poor", "Below range → poor"

    print("✓ Signal quality module self-test passed")
    print(get_eeg_quality_details(good_eeg, raw=True))
    print(get_gsr_quality_details(good_gsr))
