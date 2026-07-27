"""
viz_deap_loader.py
===================
Visualise what DEAPLoader produces from one subject file.

Run from project root:
    python viz_deap_loader.py

Requirements: matplotlib, numpy
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Desktop\FYP\V2 MedOracle\MedOracle\data\DEAP"
SUBJECT_FILE = os.path.join(DATA_DIR, "s01.dat")

SAMPLE_RATE    = 128
BASELINE_SAMP  = 3 * SAMPLE_RATE   # 384 samples
WINDOW_SAMPLES = 4 * SAMPLE_RATE   # 512 samples per window

EMOTION_CLASSES = {"stress": 0, "calm": 1, "happy": 2, "sad": 3, "angry": 4}
COLORS = {
    "stress": "#e74c3c",
    "calm"  : "#2ecc71",
    "happy" : "#f39c12",
    "sad"   : "#3498db",
    "angry" : "#9b59b6",
}

EEG_CHANNEL_NAMES = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7",
    "PO3","O1","Oz","Pz","P4","P8","PO4","O2","T8","CP6","CP2","C4",
    "FC6","FC2","F8","F4","AF4","Fp2","Fz","Cz"
]
# ─────────────────────────────────────────────────────────────────────────────

def map_deap_to_class(v, a, d):
    v = max(1.0, min(9.0, v))
    a = max(1.0, min(9.0, a))
    d = max(1.0, min(9.0, d))
    T = 5.0
    v_hi = v >= T; a_hi = a >= T; d_hi = d >= T
    if     v_hi and     a_hi and     d_hi: return "happy"
    if     v_hi and not a_hi and     d_hi: return "calm"
    if not v_hi and     a_hi and not d_hi: return "stress"
    if not v_hi and not a_hi and not d_hi: return "sad"
    if not v_hi and     a_hi and     d_hi: return "angry"
    return None


def load_subject(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    print(f"Loading {SUBJECT_FILE} ...")
    raw    = load_subject(SUBJECT_FILE)
    data   = raw["data"]    # (40, 40, 8064)
    labels = raw["labels"]  # (40, 4)

    # Trim baseline
    data_t = data[:, :, BASELINE_SAMP:]   # (40, 40, 7680)

    # ─── Collect windows & labels ───────────────────────────────────────────
    windows_eeg, windows_gsr, window_labels = [], [], []
    skipped = 0

    for trial in range(40):
        v = float(np.clip(labels[trial, 0], 1, 9))
        a = float(np.clip(labels[trial, 1], 1, 9))
        d = float(np.clip(labels[trial, 2], 1, 9))
        emotion = map_deap_to_class(v, a, d)
        if emotion is None:
            skipped += 1
            continue

        eeg_trial = data_t[trial, :32, :]   # (32, 7680)
        gsr_trial = data_t[trial,  36, :]   # (7680,)

        n_windows = len(eeg_trial[0]) // WINDOW_SAMPLES
        for w in range(n_windows):
            s = w * WINDOW_SAMPLES
            e = s + WINDOW_SAMPLES
            windows_eeg.append(eeg_trial[:, s:e])
            windows_gsr.append(gsr_trial[s:e])
            window_labels.append(emotion)

    windows_eeg = np.array(windows_eeg)   # (N, 32, 512)
    windows_gsr = np.array(windows_gsr)   # (N, 512)

    print(f"\n  Subject: s01")
    print(f"  Total windows : {len(windows_eeg)}")
    print(f"  Skipped trials: {skipped}")
    label_counts = {em: window_labels.count(em) for em in EMOTION_CLASSES}
    for em, cnt in label_counts.items():
        print(f"  {em:8s}: {cnt} windows")

    # ─── FIGURE ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("DEAPLoader Output — Subject s01", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Class distribution (bar chart) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    emotions = list(EMOTION_CLASSES.keys())
    counts   = [label_counts.get(em, 0) for em in emotions]
    bars = ax1.bar(emotions, counts, color=[COLORS[e] for e in emotions], edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(cnt), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("Window Class Distribution", fontweight="bold")
    ax1.set_ylabel("Number of windows")
    ax1.set_xlabel("Emotion class")
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)

    # ── Panel 2: V/A/D scatter ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for trial in range(40):
        v = float(np.clip(labels[trial, 0], 1, 9))
        a = float(np.clip(labels[trial, 1], 1, 9))
        em = map_deap_to_class(v, a, float(np.clip(labels[trial, 2], 1, 9)))
        color = COLORS.get(em, "#999999") if em else "#cccccc"
        ax2.scatter(v, a, c=color, s=60, edgecolors="white", linewidths=0.5, zorder=3)

    ax2.axhline(5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axvline(5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Valence →")
    ax2.set_ylabel("Arousal →")
    ax2.set_title("V/A Space (all 40 trials)", fontweight="bold")
    ax2.set_xlim(0.5, 9.5); ax2.set_ylim(0.5, 9.5)
    ax2.grid(alpha=0.2)

    # Legend for V/A scatter
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=COLORS[e], label=e) for e in emotions]
    legend_elems.append(Patch(facecolor="#cccccc", label="unclassified"))
    ax2.legend(handles=legend_elems, fontsize=7, loc="upper left",
               framealpha=0.8, ncol=2)

    # ── Panel 3: Labels raw heatmap (V, A, D, L for all 40 trials) ──────────
    ax3 = fig.add_subplot(gs[0, 2])
    label_matrix = labels[:, :3]   # (40, 3): V, A, D
    im = ax3.imshow(label_matrix.T, aspect="auto", cmap="RdYlGn",
                    vmin=1, vmax=9, interpolation="nearest")
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Valence", "Arousal", "Dominance"])
    ax3.set_xlabel("Trial index")
    ax3.set_title("Raw V/A/D Ratings (1–9)", fontweight="bold")
    plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.04, label="Rating")

    # ── Panel 4: EEG sample (first "stress" window, all 32 channels) ────────
    ax4 = fig.add_subplot(gs[1, :])
    stress_idx = next((i for i, l in enumerate(window_labels) if l == "stress"), 0)
    eeg_sample = windows_eeg[stress_idx]   # (32, 512)
    t_axis     = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE

    offset = 0
    step   = float(np.abs(eeg_sample).max()) * 1.5
    for ch in range(32):
        ax4.plot(t_axis, eeg_sample[ch] + offset, linewidth=0.5,
                 color=plt.cm.tab20(ch % 20), alpha=0.8)
        ax4.text(-0.05, offset, EEG_CHANNEL_NAMES[ch],
                 ha="right", va="center", fontsize=5, color="gray")
        offset += step

    ax4.set_title(f'EEG — First "{window_labels[stress_idx]}" window (all 32 channels)',
                  fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Channel (stacked)")
    ax4.set_yticks([])
    ax4.set_xlim(-0.1, 4.0)
    ax4.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax4.axvline(4, color="black", linewidth=0.5, alpha=0.3)

    # ── Panel 5: GSR across multiple windows ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    for em, color in COLORS.items():
        idxs = [i for i, l in enumerate(window_labels) if l == em][:3]
        for idx in idxs:
            ax5.plot(t_axis, windows_gsr[idx], color=color, alpha=0.6,
                     linewidth=1.2, label=em)

    # deduplicate legend
    handles, labels_leg = ax5.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = h
    ax5.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    ax5.set_title("GSR (raw units) — 3 sample windows per class", fontweight="bold")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("GSR (raw amplifier units)")
    ax5.grid(alpha=0.2)

    # ── Panel 6: EEG amplitude distribution across channels ─────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    all_eeg_flat = windows_eeg.reshape(-1)
    ax6.hist(all_eeg_flat, bins=80, color="#3498db", edgecolor="none", alpha=0.8)
    ax6.axvline(0,   color="black", linewidth=1, linestyle="--", label="0 µV")
    ax6.axvline( 126, color="#e74c3c", linewidth=1, label="+126 µV")
    ax6.axvline(-126, color="#e74c3c", linewidth=1, label="−126 µV")
    ax6.set_title("EEG Amplitude Distribution", fontweight="bold")
    ax6.set_xlabel("Amplitude (µV)")
    ax6.set_ylabel("Count")
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.2)

    plt.savefig("viz_deap_loader_output.png", dpi=150, bbox_inches="tight")
    print("\nSaved: viz_deap_loader_output.png")
    plt.show()


if __name__ == "__main__":
    main()