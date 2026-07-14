"""
member1_physiological/utils/metrics.py
========================================
Evaluation metrics for the physiological emotion classification module.

Responsibilities:
  1. compute_fold_metrics()    — per-fold scores: macro-F1, per-class F1,
                                  accuracy, confusion matrix, precision, recall
  2. aggregate_loso_metrics()  — mean ± std across all 32 LOSO folds
  3. compute_class_weights()   — balanced class weights for weighted cross-entropy
  4. wilcoxon_test()           — one-tailed Wilcoxon signed-rank test for ablation
  5. format_fold_report()      — human-readable per-fold log string
  6. format_loso_summary()     — final summary table across all folds

Primary metric: Macro-averaged F1 (not accuracy).
Rationale: class imbalance is present in DEAP 5-class mapping;
           macro-F1 weights each class equally regardless of support.

Author: Suhira Balarajan (214206G)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import wilcoxon as _scipy_wilcoxon

import os, sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.models.physiological_net import EMOTION_CLASSES, N_CLASSES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class index → class name, in label-integer order (0=stress … 4=angry)
CLASS_NAMES: List[str] = [
    name for name, _ in sorted(EMOTION_CLASSES.items(), key=lambda x: x[1])
]
# = ["stress", "calm", "happy", "sad", "angry"]

ALL_LABELS: List[int] = list(range(N_CLASSES))   # [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 1. Per-fold metrics
# ---------------------------------------------------------------------------

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute all evaluation metrics for a single LOSO fold.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)   — ground-truth integer class labels (0–4)
    y_pred : np.ndarray, shape (N,)   — predicted integer class labels (0–4)

    Returns
    -------
    dict with keys:
        macro_f1          : float     — primary metric
        accuracy          : float
        per_class_f1      : Dict[str, float]   — keyed by class name
        per_class_precision: Dict[str, float]
        per_class_recall  : Dict[str, float]
        per_class_support : Dict[str, int]     — number of true samples
        confusion_matrix  : np.ndarray (5, 5) — rows=true, cols=pred
        n_samples         : int
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        raise ValueError("y_true is empty — cannot compute metrics.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # --- Primary metric: macro-averaged F1 ---
    macro_f1 = float(f1_score(
        y_true, y_pred,
        labels    = ALL_LABELS,
        average   = "macro",
        zero_division = 0,
    ))

    # --- Accuracy ---
    accuracy = float(np.mean(y_true == y_pred))

    # --- Per-class precision / recall / F1 / support ---
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred,
        labels        = ALL_LABELS,
        average       = None,        # per-class
        zero_division = 0,
    )

    per_class_f1        = {CLASS_NAMES[i]: float(f1_arr[i])        for i in range(N_CLASSES)}
    per_class_precision = {CLASS_NAMES[i]: float(precision_arr[i]) for i in range(N_CLASSES)}
    per_class_recall    = {CLASS_NAMES[i]: float(recall_arr[i])    for i in range(N_CLASSES)}
    per_class_support   = {CLASS_NAMES[i]: int(support_arr[i])     for i in range(N_CLASSES)}

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)

    return {
        "macro_f1"          : macro_f1,
        "accuracy"          : accuracy,
        "per_class_f1"      : per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall"  : per_class_recall,
        "per_class_support" : per_class_support,
        "confusion_matrix"  : cm,                 # np.ndarray (5, 5)
        "n_samples"         : int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# 2. LOSO aggregation
# ---------------------------------------------------------------------------

def aggregate_loso_metrics(
    fold_results: List[Dict],
) -> Dict:
    """
    Aggregate per-fold metrics across all 32 LOSO folds.

    Parameters
    ----------
    fold_results : list of dicts, each from compute_fold_metrics()
                  Length must be 1–32.

    Returns
    -------
    dict with keys:
        mean_macro_f1         : float
        std_macro_f1          : float
        all_macro_f1s         : List[float]   — one per fold (for Wilcoxon)
        mean_accuracy         : float
        std_accuracy          : float
        per_class_mean_f1     : Dict[str, float]
        per_class_std_f1      : Dict[str, float]
        summed_confusion_matrix: np.ndarray (5, 5) — sum across folds
        n_folds               : int
        total_samples         : int
    """
    if not fold_results:
        raise ValueError("fold_results is empty — need at least one fold.")

    all_macro   = [r["macro_f1"] for r in fold_results]
    all_acc     = [r["accuracy"]  for r in fold_results]

    # Per-class F1: collect across folds then average
    per_class_all: Dict[str, List[float]] = {name: [] for name in CLASS_NAMES}
    for r in fold_results:
        for name in CLASS_NAMES:
            per_class_all[name].append(r["per_class_f1"][name])

    per_class_mean = {name: float(np.mean(vals)) for name, vals in per_class_all.items()}
    per_class_std  = {name: float(np.std(vals))  for name, vals in per_class_all.items()}

    # Summed confusion matrix across all folds
    cm_sum = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for r in fold_results:
        cm_sum += r["confusion_matrix"].astype(np.int64)

    return {
        "mean_macro_f1"          : float(np.mean(all_macro)),
        "std_macro_f1"           : float(np.std(all_macro)),
        "all_macro_f1s"          : all_macro,         # keep raw for Wilcoxon
        "mean_accuracy"          : float(np.mean(all_acc)),
        "std_accuracy"           : float(np.std(all_acc)),
        "per_class_mean_f1"      : per_class_mean,
        "per_class_std_f1"       : per_class_std,
        "summed_confusion_matrix": cm_sum,
        "n_folds"                : len(fold_results),
        "total_samples"          : sum(r["n_samples"] for r in fold_results),
    }


# ---------------------------------------------------------------------------
# 3. Class weights for weighted cross-entropy loss
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute balanced class weights for use in nn.CrossEntropyLoss(weight=...).

    Uses sklearn's 'balanced' strategy:
        weight_c = n_samples / (n_classes × n_samples_in_class_c)

    Parameters
    ----------
    labels : np.ndarray, shape (N,) — integer class labels from the TRAINING set
    device : torch.device — where to place the returned tensor (default: cpu)

    Returns
    -------
    torch.FloatTensor, shape (N_CLASSES,) — one weight per class, in label order

    IMPORTANT: Only call this on the TRAINING set labels, never on test labels.
    """
    labels = np.asarray(labels, dtype=int)

    # Identify which classes are actually present in training data
    present = np.unique(labels)

    weights = compute_class_weight(
        class_weight = "balanced",
        classes      = present,
        y            = labels,
    )

    # Fill a (N_CLASSES,) array — absent classes get weight 1.0
    weight_tensor = np.ones(N_CLASSES, dtype=np.float32)
    for cls_idx, w in zip(present, weights):
        weight_tensor[cls_idx] = float(w)

    tensor = torch.tensor(weight_tensor, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# 4. Wilcoxon signed-rank test (ablation study significance)
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = "greater",
) -> Tuple[float, float]:
    """
    One-tailed Wilcoxon signed-rank test between two sets of fold scores.

    Used in ablation study to test whether condition A (e.g. C5 full method)
    is significantly better than condition B (e.g. C1 EEG-only).

    H0: median(scores_a - scores_b) = 0
    H1: median(scores_a - scores_b) > 0   (alternative='greater')

    Parameters
    ----------
    scores_a    : list of per-fold macro-F1 scores for method A
    scores_b    : list of per-fold macro-F1 scores for method B
    alternative : 'greater' | 'less' | 'two-sided'

    Returns
    -------
    (statistic, p_value) — p < 0.05 means A is significantly better than B
                            at the alpha=0.05 level (Suhira et al. report this)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length: "
            f"{len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 fold scores for Wilcoxon test.")

    stat, p_val = _scipy_wilcoxon(
        np.array(scores_a) - np.array(scores_b),
        alternative = alternative,
    )
    return float(stat), float(p_val)


# ---------------------------------------------------------------------------
# 5. Formatted log strings
# ---------------------------------------------------------------------------

def format_fold_report(
    fold_metrics : Dict,
    fold_idx     : int,
    subject_id   : str,
) -> str:
    """
    Return a human-readable per-fold report string for console/file logging.

    Example output:
        [Fold 01 | s01] macro-F1: 0.6234 | acc: 0.6512 | n=240
          stress : F1=0.71  P=0.69  R=0.73  (n=48)
          calm   : F1=0.58  P=0.61  R=0.56  (n=52)
          ...
    """
    m    = fold_metrics
    lines = [
        f"[Fold {fold_idx:02d} | {subject_id}] "
        f"macro-F1: {m['macro_f1']:.4f} | "
        f"acc: {m['accuracy']:.4f} | "
        f"n={m['n_samples']}",
    ]
    for name in CLASS_NAMES:
        f1  = m["per_class_f1"][name]
        pr  = m["per_class_precision"][name]
        rec = m["per_class_recall"][name]
        sup = m["per_class_support"][name]
        lines.append(
            f"  {name:<8}: F1={f1:.4f}  P={pr:.4f}  R={rec:.4f}  (n={sup})"
        )
    return "\n".join(lines)


def format_loso_summary(agg_metrics: Dict) -> str:
    """
    Return a final summary table string after all LOSO folds complete.

    Example output:
        ============================================================
        LOSO RESULTS (32 folds, 11520 total windows)
        ============================================================
        Macro-F1 : 0.6134 +/- 0.0412    [PRIMARY METRIC]
        Accuracy : 0.6298 +/- 0.0388
        ------------------------------------------------------------
        Per-class F1 (mean +/- std):
          stress : 0.6801 +/- 0.0521
          calm   : 0.5912 +/- 0.0634
          happy  : 0.6423 +/- 0.0498
          sad    : 0.5734 +/- 0.0711
          angry  : 0.5700 +/- 0.0589
        ============================================================
    """
    a = agg_metrics
    sep  = "=" * 60
    dash = "-" * 60

    lines = [
        sep,
        f"LOSO RESULTS ({a['n_folds']} folds, {a['total_samples']} total windows)",
        sep,
        f"Macro-F1 : {a['mean_macro_f1']:.4f} +/- {a['std_macro_f1']:.4f}    [PRIMARY METRIC]",
        f"Accuracy : {a['mean_accuracy']:.4f} +/- {a['std_accuracy']:.4f}",
        dash,
        "Per-class F1 (mean +/- std):",
    ]
    for name in CLASS_NAMES:
        mean = a["per_class_mean_f1"][name]
        std  = a["per_class_std_f1"][name]
        lines.append(f"  {name:<8}: {mean:.4f} +/- {std:.4f}")

    lines += [
        dash,
        "Confusion Matrix (summed across all folds):",
        "             " + "  ".join(f"{n[:5]:>5}" for n in CLASS_NAMES),
    ]
    cm = a["summed_confusion_matrix"]
    for i, row_name in enumerate(CLASS_NAMES):
        row_str = "  ".join(f"{cm[i, j]:>5}" for j in range(N_CLASSES))
        lines.append(f"  {row_name:<8}   {row_str}")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (no DEAP data needed — uses synthetic labels)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running metrics.py self-test...\n")
    rng = np.random.default_rng(42)

    # --- Test 1: compute_fold_metrics ---
    N = 240
    y_true = rng.integers(0, N_CLASSES, N)
    y_pred = rng.integers(0, N_CLASSES, N)

    fm = compute_fold_metrics(y_true, y_pred)

    assert "macro_f1"           in fm
    assert "accuracy"           in fm
    assert "per_class_f1"       in fm
    assert "confusion_matrix"   in fm
    assert fm["confusion_matrix"].shape == (N_CLASSES, N_CLASSES)
    assert set(fm["per_class_f1"].keys()) == set(CLASS_NAMES)
    assert 0.0 <= fm["macro_f1"] <= 1.0
    assert fm["n_samples"] == N

    print("Test 1 — compute_fold_metrics: PASSED")
    print(format_fold_report(fm, fold_idx=1, subject_id="s01"))
    print()

    # --- Test 2: aggregate_loso_metrics ---
    fold_list = []
    for fold in range(32):
        yt = rng.integers(0, N_CLASSES, 240)
        yp = rng.integers(0, N_CLASSES, 240)
        fold_list.append(compute_fold_metrics(yt, yp))

    agg = aggregate_loso_metrics(fold_list)

    assert agg["n_folds"] == 32
    assert agg["total_samples"] == 32 * 240
    assert agg["summed_confusion_matrix"].shape == (N_CLASSES, N_CLASSES)
    assert len(agg["all_macro_f1s"]) == 32
    assert 0.0 <= agg["mean_macro_f1"] <= 1.0

    print("Test 2 — aggregate_loso_metrics: PASSED")
    print(format_loso_summary(agg))
    print()

    # --- Test 3: compute_class_weights ---
    labels_train = rng.integers(0, N_CLASSES, 9000)
    w = compute_class_weights(labels_train)

    assert w.shape == (N_CLASSES,)
    assert w.dtype == torch.float32
    assert (w > 0).all()

    print(f"Test 3 — compute_class_weights: PASSED")
    print(f"  Class weights: {[round(float(x), 4) for x in w]}")
    print(f"  (Higher weight = fewer training samples = more penalty for misclassification)")
    print()

    # --- Test 4: wilcoxon_test ---
    scores_strong = [0.65, 0.68, 0.70, 0.67, 0.72, 0.63, 0.69, 0.71]
    scores_weak   = [0.58, 0.60, 0.62, 0.59, 0.63, 0.57, 0.61, 0.60]

    stat, p = wilcoxon_test(scores_strong, scores_weak, alternative="greater")
    print(f"Test 4 — wilcoxon_test: PASSED")
    print(f"  statistic={stat:.2f}  p-value={p:.4f}")
    print(f"  Interpretation: {'significant (p<0.05)' if p < 0.05 else 'not significant'}")
    print()

    # --- Test 5: edge case — perfect predictions ---
    y_perfect = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    fm_perfect = compute_fold_metrics(y_perfect, y_perfect)
    assert fm_perfect["macro_f1"] == 1.0, "Perfect predictions should give macro-F1=1.0"
    assert fm_perfect["accuracy"] == 1.0

    print("Test 5 — perfect predictions: PASSED (macro-F1=1.0, acc=1.0)")
    print()

    print("All metrics.py self-tests PASSED")

    print()
    print("CLASS_NAMES in label-index order:", CLASS_NAMES)
    print("Run: python -m member1_physiological.utils.metrics")
