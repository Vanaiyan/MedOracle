"""
member2_video_fusion/evaluation/ablation_study.py
=================================================
MedOracle — Ablation study (C1–C8) + Wilcoxon significance.

Evaluates the contribution of each fusion component on a PAIRED cross-dataset
eval set. Because DEAP (physio) and CREMA-D (video) share no subjects, each
"sample" pairs a real DEAP EEG/GSR window with a held-out CREMA-D video clip of
the SAME emotion. We precompute each modality's prediction once, then evaluate
all conditions over many random pairings — the pairing variability is the
natural repeated measurement for the significance test.

Conditions (CLAUDE.md RESOLVED ISSUE 8)
---------------------------------------
    C1  physio only                         (baseline)
    C2  video only                          (baseline)
    C3  equal-weight fusion (0.5 / 0.5)
    C4  confidence-weighted (no quality penalty)
    C5  FULL method: confidence × quality gating
    C6  robustness: video quality forced poor (α=0.1)
    C7  robustness: video missing → graceful degradation (physio only)
    C8  robustness: physio quality forced poor (α=0.1)

Metric: macro-averaged F1.
Significance: one-tailed Wilcoxon signed-rank, C5 vs {C1, C2, C3}, α=0.05
              (member1_physiological.utils.metrics.wilcoxon_test).

Run (needs data/DEAP + data/CREMA-D + the trained checkpoints):
    python -m member2_video_fusion.evaluation.ablation_study            # full
    python -m member2_video_fusion.evaluation.ablation_study --quick    # small/fast
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from member2_video_fusion.evaluation.build_synced_dataset import _collect_deap
from member2_video_fusion.inference import predict_video_modality
from member2_video_fusion.pipeline import _get_physio_predictor
from member2_video_fusion.fusion import (
    entropy_confidence, quality_alpha, physio_quality_of, EMOTIONS,
)
from member1_physiological.utils.metrics import wilcoxon_test

CREMAD_MANIFEST = _ROOT / "data" / "CREMA-D" / "manifest.csv"
OUT_DIR = _ROOT / "data" / "synced_samples"          # gitignored — results live beside the data
# Held-out CREMA-D test actors (never seen in video training) — from test_summary_v2.json
TEST_ACTORS = {1004, 1014, 1015, 1018, 1029, 1032, 1036, 1082, 1087}

CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
COND_LABEL = {
    "C1": "physio only", "C2": "video only", "C3": "equal 0.5/0.5",
    "C4": "confidence (no quality)", "C5": "FULL (conf × quality)",
    "C6": "robust: video poor", "C7": "robust: video missing", "C8": "robust: physio poor",
}


# ---------------------------------------------------------------------------
# Fusion variants (operate on plain {emotion: prob} dicts)
# ---------------------------------------------------------------------------

def _argmax(probs: dict) -> str:
    return max(probs, key=probs.get)


def _fuse(pp: dict, vp: dict, pq: str, vq: str, use_quality: bool) -> dict:
    """Confidence-weighted fusion; quality penalty applied only if use_quality."""
    c_p, c_v = entropy_confidence(pp), entropy_confidence(vp)
    a_p = quality_alpha(pq) if use_quality else 1.0
    a_v = quality_alpha(vq) if use_quality else 1.0
    g_p, g_v = c_p * a_p, c_v * a_v
    total = g_p + g_v
    if total < 1e-9:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    w_p, w_v = g_p / total, g_v / total
    return {e: w_p * pp[e] + w_v * vp[e] for e in EMOTIONS}


def _predict_conditions(pp: dict, pq: str, vp: dict, vq: str) -> dict:
    """Return the predicted label under every condition for one paired sample."""
    return {
        "C1": _argmax(pp),
        "C2": _argmax(vp),
        "C3": _argmax({e: 0.5 * pp[e] + 0.5 * vp[e] for e in EMOTIONS}),
        "C4": _argmax(_fuse(pp, vp, pq, vq, use_quality=False)),
        "C5": _argmax(_fuse(pp, vp, pq, vq, use_quality=True)),
        "C6": _argmax(_fuse(pp, vp, pq, "poor", use_quality=True)),   # video poor
        "C7": _argmax(pp),                                            # video missing → physio only
        "C8": _argmax(_fuse(pp, vp, "poor", vq, use_quality=True)),   # physio poor
    }


# ---------------------------------------------------------------------------
# Build the per-modality prediction pools (each model runs ONCE per item)
# ---------------------------------------------------------------------------

def _build_video_pool(k: int) -> dict:
    """emotion -> list of (probs dict, video_quality) from held-out CREMA-D clips."""
    with open(CREMAD_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    by_emo = {e: [] for e in EMOTIONS}
    for r in rows:
        if int(r["actor_id"]) in TEST_ACTORS and r["emotion"] in by_emo:
            by_emo[r["emotion"]].append(r)

    rng = random.Random(0)
    pool = {}
    for e in EMOTIONS:
        clips = rng.sample(by_emo[e], min(k, len(by_emo[e])))
        preds = []
        print(f"  video[{e}]: running model on {len(clips)} held-out clips…", flush=True)
        for c in clips:
            vp, vq = predict_video_modality(_ROOT / c["path"])
            preds.append(({em: vp["class_probabilities"][em] for em in EMOTIONS}, vq))
        pool[e] = preds
    return pool


def _build_physio_pool(k: int) -> dict:
    """emotion -> list of (probs dict, physio_quality) from DEAP windows."""
    cand = _collect_deap()                       # emotion -> clearest-first candidates
    predictor = _get_physio_predictor()
    pool = {}
    for e in EMOTIONS:
        top = cand[e][:k]
        eeg_b = np.stack([c[6] for c in top])    # (k, 32, 512)
        gsr_b = np.stack([c[7] for c in top])    # (k, 512)
        preds = predictor.predict_batch(eeg_b, gsr_b)
        pool[e] = [
            ({em: p["class_probabilities"][em] for em in EMOTIONS},
             physio_quality_of(p["signal_quality"]["eeg"], p["signal_quality"]["gsr"]))
            for p in preds
        ]
        print(f"  physio[{e}]: {len(preds)} DEAP windows scored", flush=True)
    return pool


# ---------------------------------------------------------------------------
# One pairing → macro-F1 per condition
# ---------------------------------------------------------------------------

def _run_pairing(physio_pool, video_pool, rng):
    y_true, preds = [], {c: [] for c in CONDITIONS}
    for e in EMOTIONS:
        phys = physio_pool[e][:]
        rng.shuffle(phys)
        vids = video_pool[e]
        for (pp, pq), (vp, vq) in zip(phys, vids):
            cp = _predict_conditions(pp, pq, vp, vq)
            y_true.append(e)
            for c in CONDITIONS:
                preds[c].append(cp[c])
    f1 = {c: f1_score(y_true, preds[c], labels=EMOTIONS, average="macro", zero_division=0)
          for c in CONDITIONS}
    return f1, y_true, preds


# ---------------------------------------------------------------------------
# Confusion-matrix figure (C1 / C2 / C5)
# ---------------------------------------------------------------------------

def _save_confusion_figure(y_true, preds, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cond in zip(axes, ["C1", "C2", "C5"]):
        cm = confusion_matrix(y_true, preds[cond], labels=EMOTIONS).astype(np.float32)
        cm /= np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right"); ax.set_yticklabels(EMOTIONS)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_title(f"{cond} — {COND_LABEL[cond]}", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.suptitle("Ablation — normalised confusion matrices (representative pairing)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="MedOracle C1–C8 ablation study")
    ap.add_argument("--k", type=int, default=20, help="samples per emotion (default 20)")
    ap.add_argument("--n", type=int, default=50, help="random pairings (default 50)")
    ap.add_argument("--quick", action="store_true", help="fast run: k=6, n=20")
    args = ap.parse_args()
    k, n = (6, 20) if args.quick else (args.k, args.n)

    print(f"\n{'='*64}\n  MedOracle Ablation Study (C1–C8)\n{'='*64}")
    print(f"  samples/emotion = {k}   pairings = {n}\n")

    print("Precomputing per-modality predictions (models run once per item)…")
    video_pool  = _build_video_pool(k)
    physio_pool = _build_physio_pool(k)

    print(f"\nEvaluating {n} random cross-dataset pairings…")
    rng = random.Random(123)
    per_cond = {c: [] for c in CONDITIONS}
    first_pairing = None
    for i in range(n):
        f1, y_true, preds = _run_pairing(physio_pool, video_pool, rng)
        for c in CONDITIONS:
            per_cond[c].append(f1[c])
        if i == 0:
            first_pairing = (y_true, preds)

    # --- results table ---
    print(f"\n{'Cond':<5} {'Description':<26} {'Macro-F1 (mean ± std)'}")
    print("  " + "-" * 60)
    summary = {}
    for c in CONDITIONS:
        arr = np.array(per_cond[c])
        summary[c] = {"label": COND_LABEL[c], "mean_macro_f1": float(arr.mean()),
                      "std_macro_f1": float(arr.std())}
        star = "  ← full method" if c == "C5" else ""
        print(f"  {c:<4} {COND_LABEL[c]:<26} {arr.mean():.4f} ± {arr.std():.4f}{star}")

    # --- Wilcoxon: C5 vs C1 / C2 / C3 (one-tailed, greater) ---
    print(f"\n  Wilcoxon signed-rank (one-tailed, C5 > baseline, α=0.05):")
    sig = {}
    for base in ["C1", "C2", "C3"]:
        try:
            stat, p = wilcoxon_test(per_cond["C5"], per_cond[base], alternative="greater")
            verdict = "significant ✓" if p < 0.05 else "not significant"
            print(f"    C5 vs {base} ({COND_LABEL[base]:<22}): p = {p:.4g}  {verdict}")
            sig[f"C5_vs_{base}"] = {"statistic": stat, "p_value": p, "significant": bool(p < 0.05)}
        except Exception as exc:
            print(f"    C5 vs {base}: n/a ({exc})")
            sig[f"C5_vs_{base}"] = {"error": str(exc)}

    # --- persist ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = OUT_DIR / "ablation_confusion_matrices.png"
    _save_confusion_figure(*first_pairing, cm_path)
    results = {
        "k_per_emotion": k, "n_pairings": n,
        "conditions": summary,
        "wilcoxon_C5_vs_baselines": sig,
        "per_pairing_macro_f1": {c: per_cond[c] for c in CONDITIONS},
    }
    json_path = OUT_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {cm_path}")
    print(f"\n{'='*64}\n  Ablation complete.\n{'='*64}")


if __name__ == "__main__":
    main()
