"""
viz_signal_quality.py
======================
Visualise signal quality assessment for EEG and GSR.
Shows the exact checks that assess_eeg_quality() and assess_gsr_quality() run.

Run from project root:
    python viz_signal_quality.py

Requirements: matplotlib, numpy, scipy
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\Desktop\FYP\V2 MedOracle\MedOracle\data\DEAP"
SUBJECT_FILE = os.path.join(DATA_DIR, "s01.dat")
TRIAL_IDX    = 0     # which trial to inspect (0–39)
WINDOW_IDX   = 0     # which 4s window to inspect (0–14)

SAMPLE_RATE    = 128
BASELINE_SAMP  = 3 * SAMPLE_RATE
WINDOW_SAMPLES = 4 * SAMPLE_RATE

EEG_CHANNEL_NAMES = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7",
    "PO3","O1","Oz","Pz","P4","P8","PO4","O2","T8","CP6","CP2","C4",
    "FC6","FC2","F8","F4","AF4","Fp2","Fz","Cz"
]
# ─────────────────────────────────────────────────────────────────────────────

# ── Inline quality checks (mirrors signal_quality.py logic) ─────────────────

def grade_color(g):
    return {"good": "#2ecc71", "degraded": "#f39c12", "poor": "#e74c3c"}.get(g, "gray")

def worst_grade(grades):
    order = {"poor": 2, "degraded": 1, "good": 0}
    return max(grades, key=lambda g: order[g])

def assess_eeg(eeg):
    """Returns (overall_grade, details_dict)"""
    grades = []
    details = {}

    nan_ratio = float(np.isnan(eeg).mean())
    details["NaN ratio"] = f"{nan_ratio*100:.2f}%"
    if nan_ratio > 0.05:   grades.append("poor")
    elif nan_ratio > 0.01: grades.append("degraded")
    else:                  grades.append("good")

    max_abs = float(np.nanmax(np.abs(eeg)))
    details["Max amplitude"] = f"±{max_abs:.1f} µV"
    if max_abs > 150:   grades.append("poor")
    elif max_abs > 100: grades.append("degraded")
    else:               grades.append("good")

    stds  = np.nanstd(eeg, axis=1)
    n_flat = int(np.sum(stds < 1e-4))
    details["Flat channels"] = f"{n_flat}"
    if n_flat > 2:   grades.append("poor")
    elif n_flat > 0: grades.append("degraded")
    else:            grades.append("good")

    overall = worst_grade(grades)
    return overall, details, stds

def assess_gsr(gsr):
    """Returns (overall_grade, details_dict, peaks_array)"""
    grades = []
    details = {}

    nan_ratio = float(np.isnan(gsr).mean())
    details["NaN ratio"] = f"{nan_ratio*100:.2f}%"
    if nan_ratio > 0.05:   grades.append("poor")
    elif nan_ratio > 0.01: grades.append("degraded")
    else:                  grades.append("good")

    gsr_std = float(np.nanstd(gsr))
    gsr_range = float(np.nanmax(gsr) - np.nanmin(gsr))
    details["Signal std"]   = f"{gsr_std:.2f}"
    details["Signal range"] = f"{gsr_range:.2f}"
    if gsr_std < 1e-6: grades.append("poor")
    else:              grades.append("good")

    median = float(np.nanmedian(gsr))
    peak_threshold = median + 0.5 * gsr_std
    peaks, _ = find_peaks(gsr, height=peak_threshold, distance=SAMPLE_RATE)
    details["SCR peaks"] = f"{len(peaks)}"
    if len(peaks) < 1:   grades.append("poor")
    elif len(peaks) < 2: grades.append("degraded")
    else:                grades.append("good")

    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    t_mean = t.mean(); y_mean = gsr.mean()
    slope = float(((t - t_mean) * (gsr - y_mean)).sum() / ((t - t_mean)**2).sum())
    rel_slope = abs(slope) / (gsr_std + 1e-8)
    details["Drift (rel)"] = f"{rel_slope:.4f}"
    if rel_slope > 0.5:   grades.append("poor")
    elif rel_slope > 0.1: grades.append("degraded")
    else:                 grades.append("good")

    overall = worst_grade(grades)
    return overall, details, peaks

# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {SUBJECT_FILE} ...")
    with open(SUBJECT_FILE, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    data = raw["data"]   # (40, 40, 8064)
    data_t = data[:, :, BASELINE_SAMP:]   # (40, 40, 7680)

    s = WINDOW_IDX * WINDOW_SAMPLES
    e = s + WINDOW_SAMPLES

    eeg_window = data_t[TRIAL_IDX, :32, s:e].astype(np.float32)   # (32, 512)
    gsr_window = data_t[TRIAL_IDX,  36, s:e].astype(np.float32)   # (512,)

    eeg_grade, eeg_details, ch_stds = assess_eeg(eeg_window)
    gsr_grade, gsr_details, peaks   = assess_gsr(gsr_window)

    print(f"\nTrial {TRIAL_IDX}, Window {WINDOW_IDX}")
    print(f"  EEG quality: {eeg_grade.upper()}")
    print(f"  GSR quality: {gsr_grade.upper()}")

    t_axis = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE

    # ─── FIGURE ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"Signal Quality Assessment  —  s01 · Trial {TRIAL_IDX} · Window {WINDOW_IDX}",
        fontsize=14, fontweight="bold"
    )

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── EEG heatmap ──────────────────────────────────────────────────────────
    ax_eeg_map = fig.add_subplot(gs[0:2, 0:2])
    im = ax_eeg_map.imshow(eeg_window, aspect="auto", cmap="RdBu_r",
                           vmin=-150, vmax=150,
                           extent=[0, 4, 32, 0], interpolation="nearest")
    ax_eeg_map.set_yticks(np.arange(32) + 0.5)
    ax_eeg_map.set_yticklabels(EEG_CHANNEL_NAMES, fontsize=6)
    ax_eeg_map.set_xlabel("Time (s)")
    ax_eeg_map.set_title(f"EEG Heatmap  [{eeg_grade.upper()}]",
                         fontweight="bold",
                         color=grade_color(eeg_grade))
    plt.colorbar(im, ax=ax_eeg_map, label="µV", fraction=0.03)

    # ── EEG per-channel std (flat-line detector) ──────────────────────────────
    ax_std = fig.add_subplot(gs[0:2, 2])
    colors_std = ["#e74c3c" if s < 1e-4 else "#3498db" for s in ch_stds]
    ax_std.barh(np.arange(32), ch_stds, color=colors_std, edgecolor="none")
    ax_std.set_yticks(np.arange(32))
    ax_std.set_yticklabels(EEG_CHANNEL_NAMES, fontsize=6)
    ax_std.axvline(1e-4, color="#e74c3c", linestyle="--", linewidth=1, label="flat threshold")
    ax_std.set_xlabel("Std dev (µV)")
    ax_std.set_title("Per-channel Std\n(red = flat)", fontweight="bold")
    ax_std.legend(fontsize=7)
    ax_std.invert_yaxis()

    # ── EEG quality metrics summary ───────────────────────────────────────────
    ax_eeg_info = fig.add_subplot(gs[0, 3])
    ax_eeg_info.axis("off")
    ax_eeg_info.set_title("EEG Quality Metrics", fontweight="bold")

    grade_box = mpatches.FancyBboxPatch((0.1, 0.75), 0.8, 0.18,
        boxstyle="round,pad=0.02", linewidth=2,
        edgecolor=grade_color(eeg_grade), facecolor=grade_color(eeg_grade)+"33")
    ax_eeg_info.add_patch(grade_box)
    ax_eeg_info.text(0.5, 0.84, f"Overall: {eeg_grade.upper()}",
                     ha="center", va="center", fontsize=13, fontweight="bold",
                     color=grade_color(eeg_grade))

    y = 0.62
    for metric, val in eeg_details.items():
        ax_eeg_info.text(0.05, y, f"• {metric}:", fontsize=9, va="center")
        ax_eeg_info.text(0.95, y, val, fontsize=9, va="center", ha="right",
                         fontweight="bold")
        y -= 0.14

    # ── GSR quality metrics summary ───────────────────────────────────────────
    ax_gsr_info = fig.add_subplot(gs[1, 3])
    ax_gsr_info.axis("off")
    ax_gsr_info.set_title("GSR Quality Metrics", fontweight="bold")

    grade_box2 = mpatches.FancyBboxPatch((0.1, 0.75), 0.8, 0.18,
        boxstyle="round,pad=0.02", linewidth=2,
        edgecolor=grade_color(gsr_grade), facecolor=grade_color(gsr_grade)+"33")
    ax_gsr_info.add_patch(grade_box2)
    ax_gsr_info.text(0.5, 0.84, f"Overall: {gsr_grade.upper()}",
                     ha="center", va="center", fontsize=13, fontweight="bold",
                     color=grade_color(gsr_grade))

    y = 0.62
    for metric, val in gsr_details.items():
        ax_gsr_info.text(0.05, y, f"• {metric}:", fontsize=9, va="center")
        ax_gsr_info.text(0.95, y, val, fontsize=9, va="center", ha="right",
                         fontweight="bold")
        y -= 0.14

    # ── GSR signal with peaks marked ─────────────────────────────────────────
    ax_gsr = fig.add_subplot(gs[2, 0:3])
    ax_gsr.plot(t_axis, gsr_window, color="#2980b9", linewidth=1.5, label="GSR signal")

    gsr_std  = float(np.nanstd(gsr_window))
    gsr_med  = float(np.nanmedian(gsr_window))
    peak_thr = gsr_med + 0.5 * gsr_std
    ax_gsr.axhline(peak_thr, color="#e74c3c", linestyle="--",
                   linewidth=1, label=f"SCR threshold (median + 0.5σ)")
    ax_gsr.axhline(gsr_med,  color="gray",    linestyle=":",
                   linewidth=1, label=f"Median = {gsr_med:.1f}")

    if len(peaks) > 0:
        ax_gsr.scatter(peaks / SAMPLE_RATE, gsr_window[peaks],
                       color="#e74c3c", zorder=5, s=80,
                       label=f"SCR peaks ({len(peaks)} found)", marker="^")
        for pk in peaks:
            ax_gsr.annotate(f"peak", (pk / SAMPLE_RATE, gsr_window[pk]),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=7, color="#e74c3c", ha="center")

    ax_gsr.set_xlabel("Time (s)")
    ax_gsr.set_ylabel("GSR (raw units)")
    ax_gsr.set_title(f"GSR Signal with SCR Peak Detection  [{gsr_grade.upper()}]",
                     fontweight="bold", color=grade_color(gsr_grade))
    ax_gsr.legend(fontsize=8, loc="upper right")
    ax_gsr.grid(alpha=0.2)

    # ── EEG single channel waveform ───────────────────────────────────────────
    ax_eeg_wave = fig.add_subplot(gs[2, 3])
    for ch_idx, ch_name in [(0, "Fp1"), (6, "C3"), (15, "Pz"), (31, "Cz")]:
        ax_eeg_wave.plot(t_axis, eeg_window[ch_idx],
                         linewidth=1, alpha=0.8, label=ch_name)
    ax_eeg_wave.set_xlabel("Time (s)")
    ax_eeg_wave.set_ylabel("µV")
    ax_eeg_wave.set_title("EEG — 4 key channels", fontweight="bold")
    ax_eeg_wave.legend(fontsize=7)
    ax_eeg_wave.grid(alpha=0.2)

    plt.savefig("viz_signal_quality_output.png", dpi=150, bbox_inches="tight")
    print("Saved: viz_signal_quality_output.png")
    plt.show()


if __name__ == "__main__":
    main()