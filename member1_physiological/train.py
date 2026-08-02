"""
member1_physiological/train.py
================================
Leave-One-Subject-Out (LOSO) training loop for PhysiologicalNet.

Pipeline per fold:
  1. Hold out one subject as test set
  2. Fit EEGPreprocessor + GSRPreprocessor on the remaining 31 subjects (training only)
  3. Build DEAPWindowDataset with augmentation for train, no augmentation for test
  4. Compute class weights from training labels (sklearn balanced strategy)
  5. Train PhysiologicalNet with weighted cross-entropy + Adam + ReduceLROnPlateau
  6. Evaluate after every epoch; save best checkpoint by macro-F1; early-stop if no
     improvement for `patience` epochs
  7. Repeat for all 32 subjects; aggregate macro-F1 mean +/- std

Usage:
    # Full 32-fold LOSO (recommended — run overnight):
    python -m member1_physiological.train --data_dir data/DEAP

    # Single fold for quick debugging:
    python -m member1_physiological.train --data_dir data/DEAP --test_subject s01

    # Custom hyperparameters:
    python -m member1_physiological.train --data_dir data/DEAP --epochs 100 --lr 5e-4

    # CPU-only explicit:
    python -m member1_physiological.train --data_dir data/DEAP --device cpu

Author: Suhira Balarajan (214206G)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure repo root is on path regardless of working directory
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.preprocessing.deap_loader import DEAPLoader, SubjectData
from member1_physiological.models.physiological_net import PhysiologicalNet
from member1_physiological.utils.dataset import build_loso_datasets, build_kfold_datasets
from member1_physiological.utils.metrics import (
    compute_fold_metrics,
    aggregate_loso_metrics,
    compute_class_weights,
    format_fold_report,
    format_loso_summary,
)


# ---------------------------------------------------------------------------
# Focal Loss — down-weights easy/dominant examples, focuses on hard ones
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with per-class weighting.
    gamma=2 is the standard choice — reduces loss for well-classified examples.
    """
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt      = torch.exp(-ce_loss)                   # probability of correct class
        focal   = ((1 - pt) ** self.gamma) * ce_loss    # down-weight easy examples
        return focal.mean()


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters in one place. Serialised into every checkpoint."""

    # --- Model architecture (must match encoders.py / physiological_net.py) ---
    d_model   : int   = 128
    n_heads   : int   = 4
    n_layers  : int   = 2
    dropout   : float = 0.3

    # --- Optimiser ---
    lr            : float = 1e-3
    weight_decay  : float = 1e-4    # L2 regularisation

    # --- LR scheduler (ReduceLROnPlateau on val macro-F1) ---
    lr_patience : int   = 7         # epochs without improvement before reducing LR
    lr_factor   : float = 0.5       # multiply LR by this on plateau
    min_lr      : float = 1e-6      # floor — never reduce below this

    # --- Training loop ---
    batch_size    : int  = 64
    n_epochs      : int  = 80
    patience      : int  = 15       # early-stop patience (epochs without improvement)
    grad_clip     : float = 1.0     # max gradient norm (prevents exploding gradients)

    # --- Data ---
    augment_train : bool = True
    num_workers   : int  = 0        # 0 is required on Windows (no fork support)

    # --- Paths ---
    checkpoint_dir : str = "checkpoints"
    log_dir        : str = "logs"


# ---------------------------------------------------------------------------
# Core training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model     : PhysiologicalNet,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    device    : torch.device,
    grad_clip : float,
) -> Tuple[float, float]:
    """
    Run one full pass over the training DataLoader.

    Returns
    -------
    (avg_loss, avg_accuracy)  — both averaged over all batches
    """
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for eeg, gsr, labels in loader:
        eeg    = eeg.to(device)       # (B, 32, 512)
        gsr    = gsr.to(device)       # (B, 512)
        labels = labels.to(device)    # (B,) — LongTensor

        optimizer.zero_grad()

        logits = model(eeg, gsr)      # (B, 5)
        loss   = criterion(logits, labels)

        loss.backward()

        # Gradient clipping — especially important for attention layers
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss    += loss.item() * eeg.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += eeg.size(0)

    avg_loss = total_loss    / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model  : PhysiologicalNet,
    loader : DataLoader,
    device : torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader without gradient tracking.

    Returns
    -------
    y_true : np.ndarray (N,) — ground-truth integer labels
    y_pred : np.ndarray (N,) — predicted integer labels (argmax of logits)
    """
    model.eval()

    all_true = []
    all_pred = []

    for eeg, gsr, labels in loader:
        eeg    = eeg.to(device)
        gsr    = gsr.to(device)

        logits = model(eeg, gsr)           # (B, 5)
        preds  = logits.argmax(dim=-1)     # (B,)

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

    return np.array(all_true, dtype=int), np.array(all_pred, dtype=int)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model       : PhysiologicalNet,
    eeg_prep_mean: np.ndarray,
    eeg_prep_std : np.ndarray,
    gsr_prep_mean: float,
    gsr_prep_std : float,
    config      : TrainConfig,
    fold_metrics: Dict,
    subject_id  : str,
    fold_idx    : int,
    epoch       : int,
    checkpoint_dir: str,
) -> str:
    """
    Save model weights + preprocessor statistics + metadata to a .pt file.

    The checkpoint contains everything predict.py needs to run inference
    without re-fitting the preprocessors.

    Returns the saved file path.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    fname = f"fold_{fold_idx:02d}_{subject_id}_best.pt"
    fpath = os.path.join(checkpoint_dir, fname)

    torch.save({
        # Model
        "model_state_dict" : model.state_dict(),

        # Preprocessor statistics (fitted on training subjects only)
        "eeg_mean"         : eeg_prep_mean,        # np.ndarray (32,)
        "eeg_std"          : eeg_prep_std,          # np.ndarray (32,)
        "gsr_mean"         : float(gsr_prep_mean),  # scalar float
        "gsr_std"          : float(gsr_prep_std),   # scalar float

        # Metadata
        "config"           : dataclasses.asdict(config),
        "fold_metrics"     : fold_metrics,
        "subject_id"       : subject_id,
        "fold_idx"         : fold_idx,
        "best_epoch"       : epoch,
    }, fpath)

    return fpath


def load_checkpoint(fpath: str, device: torch.device) -> Dict:
    """Load a checkpoint saved by save_checkpoint(). Returns the raw dict."""
    return torch.load(fpath, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    subject_data : Dict[str, SubjectData],
    test_subject : str,
    fold_idx     : int,
    config       : TrainConfig,
    device       : torch.device,
    log_file,               # open file handle or None
) -> Dict:
    """
    Run the full training loop for a single LOSO fold.

    Returns the best fold metrics dict (from compute_fold_metrics).
    """
    sep  = "=" * 78
    dash = "-" * 78

    _log(f"\n{sep}", log_file)
    _log(f"LOSO Fold {fold_idx + 1:02d}/32 — Test subject: {test_subject}", log_file)

    # --- Build datasets ---
    train_ds, test_ds, eeg_prep, gsr_prep = build_loso_datasets(
        subject_data,
        test_subject     = test_subject,
        augment_train    = config.augment_train,
    )

    _log(f"  Train windows : {len(train_ds):,}  |  Test windows: {len(test_ds):,}", log_file)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        drop_last   = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = config.batch_size * 2,   # no grad → can use larger batch
        shuffle     = False,
        num_workers = config.num_workers,
    )

    # --- Class weights (training labels only — never touch test) ---
    class_weights = compute_class_weights(train_ds.labels, device=device)
    weight_str = "  ".join(
        f"{n}={float(class_weights[i]):.3f}"
        for i, n in enumerate(["stress", "calm", "happy", "sad", "angry"])
    )
    _log(f"  Class weights : {weight_str}", log_file)
    _log(sep, log_file)

    # --- Model ---
    model = PhysiologicalNet(
        d_model  = config.d_model,
        n_heads  = config.n_heads,
        n_layers = config.n_layers,
        dropout  = config.dropout,
    ).to(device)

    # --- Optimiser + loss ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config.lr,
        weight_decay = config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- LR scheduler (monitors validation macro-F1 — higher is better) ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "max",
        patience  = config.lr_patience,
        factor    = config.lr_factor,
        min_lr    = config.min_lr,
    )

    # --- Training loop with early stopping ---
    best_macro_f1   = -1.0
    best_fold_metrics = None
    best_epoch      = 0
    best_ckpt_path  = None
    patience_counter = 0

    eeg_mean, eeg_std = eeg_prep.get_stats()    # (32,), (32,)
    gsr_mean, gsr_std = gsr_prep.get_stats()    # float, float

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.grad_clip
        )

        # Evaluate
        y_true, y_pred = evaluate(model, test_loader, device)
        fold_metrics   = compute_fold_metrics(y_true, y_pred)
        val_f1         = fold_metrics["macro_f1"]

        # Step scheduler
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        is_best = val_f1 > best_macro_f1
        marker  = "  [BEST]" if is_best else ""

        _log(
            f"  Epoch {epoch:3d}/{config.n_epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_F1={val_f1:.4f} | lr={current_lr:.3e} | "
            f"{elapsed:.1f}s{marker}",
            log_file,
        )

        # Save checkpoint on improvement
        if is_best:
            best_macro_f1    = val_f1
            best_fold_metrics = fold_metrics
            best_epoch       = epoch
            patience_counter = 0

            best_ckpt_path = save_checkpoint(
                model         = model,
                eeg_prep_mean = eeg_mean,
                eeg_prep_std  = eeg_std,
                gsr_prep_mean = gsr_mean,
                gsr_prep_std  = gsr_std,
                config        = config,
                fold_metrics  = fold_metrics,
                subject_id    = test_subject,
                fold_idx      = fold_idx,
                epoch         = epoch,
                checkpoint_dir= config.checkpoint_dir,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                _log(
                    f"  Early stop at epoch {epoch} "
                    f"(no improvement for {config.patience} epochs)",
                    log_file,
                )
                break

    _log(dash, log_file)
    _log(
        f"  Fold result: best macro-F1 = {best_macro_f1:.4f} @ epoch {best_epoch}",
        log_file,
    )
    if best_ckpt_path:
        _log(f"  Checkpoint : {best_ckpt_path}", log_file)

    # Full per-class breakdown
    _log("", log_file)
    _log(format_fold_report(best_fold_metrics, fold_idx + 1, test_subject), log_file)

    return best_fold_metrics


# ---------------------------------------------------------------------------
# Full LOSO training
# ---------------------------------------------------------------------------

def train_loso(
    subject_data : Dict[str, SubjectData],
    config       : TrainConfig,
    device       : torch.device,
    test_subject : Optional[str] = None,
) -> Dict:
    """
    Run LOSO training over all subjects (or a single subject for debugging).

    Parameters
    ----------
    subject_data  : loaded from DEAPLoader.load_all()
    config        : TrainConfig instance
    device        : torch.device
    test_subject  : if given, only run this one fold (debug mode)

    Returns
    -------
    agg_metrics : dict from aggregate_loso_metrics()
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    log_path = os.path.join(config.log_dir, "loso_training.log")
    log_file = open(log_path, "a", encoding="utf-8")

    _log(f"\n{'#' * 78}", log_file)
    _log(f"# MedOracle — PhysiologicalNet LOSO Training", log_file)
    _log(f"# Device : {device}", log_file)
    _log(f"# Config : {dataclasses.asdict(config)}", log_file)
    _log(f"{'#' * 78}", log_file)

    subject_ids = sorted(subject_data.keys())
    if test_subject is not None:
        # Debug mode — single fold
        if test_subject not in subject_ids:
            raise ValueError(f"test_subject '{test_subject}' not in loaded subjects: {subject_ids}")
        subject_ids = [test_subject]
        _log(f"\n[DEBUG] Running single fold for subject: {test_subject}", log_file)

    fold_results = []

    for fold_idx, subj_id in enumerate(subject_ids):
        fold_metrics = train_one_fold(
            subject_data = subject_data,
            test_subject = subj_id,
            fold_idx     = fold_idx,
            config       = config,
            device       = device,
            log_file     = log_file,
        )
        fold_results.append(fold_metrics)

    # Aggregate across folds
    agg = aggregate_loso_metrics(fold_results)
    summary = format_loso_summary(agg)

    _log("\n" + summary, log_file)

    # Save summary JSON
    summary_path = os.path.join(config.log_dir, "loso_summary.json")
    _save_summary_json(agg, fold_results, subject_ids, summary_path)
    _log(f"\nSummary JSON saved to: {summary_path}", log_file)

    log_file.close()
    print(summary)
    print(f"\nFull log: {log_path}")
    print(f"Summary : {summary_path}")

    return agg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str, log_file) -> None:
    """Print to stdout and write to log file simultaneously."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _save_summary_json(
    agg          : Dict,
    fold_results : List[Dict],
    subject_ids  : List[str],
    path         : str,
) -> None:
    """Serialise aggregated results to JSON for downstream analysis."""
    # Convert numpy types to Python native for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    per_fold = []
    for subj_id, fm in zip(subject_ids, fold_results):
        per_fold.append({
            "subject_id"   : subj_id,
            "macro_f1"     : float(fm["macro_f1"]),
            "accuracy"     : float(fm["accuracy"]),
            "per_class_f1" : {k: float(v) for k, v in fm["per_class_f1"].items()},
            "n_samples"    : int(fm["n_samples"]),
        })

    output = {
        "mean_macro_f1"          : float(agg["mean_macro_f1"]),
        "std_macro_f1"           : float(agg["std_macro_f1"]),
        "mean_accuracy"          : float(agg["mean_accuracy"]),
        "std_accuracy"           : float(agg["std_accuracy"]),
        "n_folds"                : int(agg["n_folds"]),
        "total_samples"          : int(agg["total_samples"]),
        "per_class_mean_f1"      : {k: float(v) for k, v in agg["per_class_mean_f1"].items()},
        "per_class_std_f1"       : {k: float(v) for k, v in agg["per_class_std_f1"].items()},
        "summed_confusion_matrix": _convert(agg["summed_confusion_matrix"]),
        "per_fold"               : per_fold,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def get_device(device_str: str) -> torch.device:
    """Resolve device string: 'auto' picks CUDA if available, else CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Random split training (Phase 1 — better demo accuracy)
# ---------------------------------------------------------------------------

def train_random_split(
    subject_data : Dict[str, SubjectData],
    config       : TrainConfig,
    device       : torch.device,
    test_ratio   : float = 0.20,
    seed         : int   = 42,
) -> Dict:
    """
    Train on 80% of all windows (randomly sampled across all subjects),
    test on remaining 20%. Single training run — no folds.

    Simulates a deployment scenario where some calibration data is available
    from each user, giving higher accuracy than LOSO.

    Returns metrics dict compatible with compute_fold_metrics().
    """
    from member1_physiological.utils.dataset import DEAPWindowDataset
    from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
    from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor
    from sklearn.model_selection import train_test_split

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    log_path = os.path.join(config.log_dir, "random_split_training.log")
    log_file = open(log_path, "a", encoding="utf-8")

    _log(f"\n{'#' * 78}", log_file)
    _log(f"# MedOracle — PhysiologicalNet Random Split Training (80/20)", log_file)
    _log(f"# Device : {device}", log_file)
    _log(f"{'#' * 78}", log_file)

    # --- Collect all windows across all subjects using to_arrays() ---
    eeg_list, gsr_list, label_list = [], [], []
    subject_ids = sorted(subject_data.keys())

    for subj_id in subject_ids:
        eeg, gsr, labels = subject_data[subj_id].get_arrays()
        eeg_list.append(eeg)
        gsr_list.append(gsr)
        label_list.append(labels)

    all_eeg    = np.concatenate(eeg_list,   axis=0).astype(np.float32)
    all_gsr    = np.concatenate(gsr_list,   axis=0).astype(np.float32)
    all_labels = np.concatenate(label_list, axis=0).astype(np.int64)

    total = len(all_labels)
    _log(f"Total windows: {total:,}", log_file)

    # --- Random 80/20 split ---
    idx = np.arange(total)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_ratio, random_state=seed, stratify=all_labels
    )

    # --- Fit preprocessors on training data only ---
    eeg_prep = EEGPreprocessor()
    gsr_prep = GSRPreprocessor()
    eeg_prep.fit(all_eeg[train_idx])
    gsr_prep.fit(all_gsr[train_idx])

    train_eeg = eeg_prep.transform(all_eeg[train_idx])
    train_gsr = gsr_prep.transform(all_gsr[train_idx])
    test_eeg  = eeg_prep.transform(all_eeg[test_idx])
    test_gsr  = gsr_prep.transform(all_gsr[test_idx])

    train_labels = all_labels[train_idx]
    test_labels  = all_labels[test_idx]

    _log(f"Train: {len(train_labels):,} | Test: {len(test_labels):,}", log_file)

    # --- Build datasets ---
    import torch
    from torch.utils.data import TensorDataset

    def _make_loader(eeg, gsr, labels, shuffle, batch_size):
        ds = TensorDataset(
            torch.tensor(eeg,    dtype=torch.float32),
            torch.tensor(gsr,    dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )

    train_loader = _make_loader(train_eeg, train_gsr, train_labels, True,  config.batch_size)
    test_loader  = _make_loader(test_eeg,  test_gsr,  test_labels,  False, config.batch_size * 2)

    # --- Class weights ---
    class_weights = compute_class_weights(train_labels, device=device)
    weight_str = "  ".join(
        f"{n}={float(class_weights[i]):.3f}"
        for i, n in enumerate(["stress", "calm", "happy", "sad", "angry"])
    )
    _log(f"Class weights: {weight_str}", log_file)

    # --- Model, optimiser, loss ---
    model = PhysiologicalNet(
        d_model  = config.d_model,
        n_heads  = config.n_heads,
        n_layers = config.n_layers,
        dropout  = config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.lr_patience,
        factor=config.lr_factor, min_lr=config.min_lr,
    )

    # --- Training loop ---
    best_f1, best_metrics, best_epoch = -1.0, None, 0
    patience_counter = 0
    eeg_mean, eeg_std = eeg_prep.get_stats()
    gsr_mean, gsr_std = gsr_prep.get_stats()

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.grad_clip
        )
        y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_fold_metrics(y_true, y_pred)
        val_f1  = metrics["macro_f1"]

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        is_best    = val_f1 > best_f1
        marker     = "  [BEST]" if is_best else ""

        _log(
            f"  Epoch {epoch:3d}/{config.n_epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_F1={val_f1:.4f} | val_acc={metrics['accuracy']:.4f} | "
            f"lr={current_lr:.3e} | {time.time()-t0:.1f}s{marker}",
            log_file,
        )

        if is_best:
            best_f1, best_metrics, best_epoch = val_f1, metrics, epoch
            patience_counter = 0
            ckpt_path = os.path.join(config.checkpoint_dir, "random_split_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "eeg_mean": eeg_mean, "eeg_std": eeg_std,
                "gsr_mean": float(gsr_mean), "gsr_std": float(gsr_std),
                "config":   dataclasses.asdict(config),
                "metrics":  metrics,
                "epoch":    epoch,
                "eval_mode": "random_split",
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                _log(f"  Early stop at epoch {epoch}", log_file)
                break

    _log(f"\nBest macro-F1 = {best_f1:.4f} @ epoch {best_epoch}", log_file)
    _log(format_fold_report(best_metrics, 1, "random_split"), log_file)

    summary = (
        f"\n{'=' * 60}\n"
        f"RANDOM SPLIT RESULTS (80/20, seed={seed})\n"
        f"{'=' * 60}\n"
        f"Macro-F1 : {best_f1:.4f}    [PRIMARY METRIC]\n"
        f"Accuracy : {best_metrics['accuracy']:.4f}\n"
        f"{'=' * 60}\n"
        f"Per-class F1:\n"
    )
    for cls, f1 in best_metrics["per_class_f1"].items():
        summary += f"  {cls:<8}: {f1:.4f}\n"
    summary += f"{'=' * 60}"

    _log(summary, log_file)
    log_file.close()
    print(summary)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Log       : {log_path}")

    return best_metrics


# ---------------------------------------------------------------------------
# 5-Fold Subject-wise GroupKFold training (cross-subject, rigorous)
# ---------------------------------------------------------------------------

def train_5fold_group(
    subject_data : Dict[str, SubjectData],
    config       : TrainConfig,
    device       : torch.device,
) -> Dict:
    """
    Train with 5-fold subject-wise GroupKFold.

    Groups are defined by subject ID — no subject appears in both train and
    test within the same fold. This is cross-subject evaluation, sitting
    between LOSO (hardest) and random split (easiest).

    5 folds × ~6–7 test subjects each → mean ± std macro-F1 reported.

    Returns aggregated metrics dict (same format as train_loso).
    """
    from sklearn.model_selection import GroupKFold
    from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
    from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    log_path = os.path.join(config.log_dir, "5fold_group_training.log")
    log_file = open(log_path, "a", encoding="utf-8")

    _log(f"\n{'#' * 78}", log_file)
    _log(f"# MedOracle — PhysiologicalNet 5-Fold Subject GroupKFold Training", log_file)
    _log(f"# Device : {device}", log_file)
    _log(f"# Config : {dataclasses.asdict(config)}", log_file)
    _log(f"{'#' * 78}", log_file)

    # Collect all windows and assign group = subject index (integer)
    subject_ids = sorted(subject_data.keys())
    all_windows = []
    all_groups  = []

    for grp_idx, sid in enumerate(subject_ids):
        windows = subject_data[sid].windows
        all_windows.extend(windows)
        all_groups.extend([grp_idx] * len(windows))

    all_groups = np.array(all_groups)
    _log(f"Total windows: {len(all_windows):,} across {len(subject_ids)} subjects", log_file)

    gkf          = GroupKFold(n_splits=5)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(all_windows, groups=all_groups)):
        train_windows = [all_windows[i] for i in train_idx]
        test_windows  = [all_windows[i] for i in test_idx]

        test_subj_indices = sorted(set(all_groups[test_idx]))
        test_subj_ids     = [subject_ids[i] for i in test_subj_indices]

        sep  = "=" * 78
        dash = "-" * 78
        _log(f"\n{sep}", log_file)
        _log(
            f"5-Fold GroupKFold — Fold {fold_idx + 1}/5 | "
            f"Test subjects: {', '.join(test_subj_ids)}", log_file
        )
        _log(
            f"  Train windows: {len(train_windows):,}  |  "
            f"Test windows: {len(test_windows):,}", log_file
        )

        train_ds, test_ds, eeg_prep, gsr_prep = build_kfold_datasets(
            train_windows, test_windows, augment_train=config.augment_train
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, drop_last=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=config.batch_size * 2, shuffle=False,
            num_workers=config.num_workers,
        )

        class_weights = compute_class_weights(train_ds.labels, device=device)
        weight_str    = "  ".join(
            f"{n}={float(class_weights[i]):.3f}"
            for i, n in enumerate(["stress", "calm", "happy", "sad", "angry"])
        )
        _log(f"  Class weights: {weight_str}", log_file)
        _log(sep, log_file)

        model = PhysiologicalNet(
            d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.n_layers, dropout=config.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=config.lr_patience,
            factor=config.lr_factor, min_lr=config.min_lr,
        )

        best_f1, best_metrics, best_epoch = -1.0, None, 0
        patience_counter = 0
        eeg_mean, eeg_std = eeg_prep.get_stats()
        gsr_mean, gsr_std = gsr_prep.get_stats()

        for epoch in range(1, config.n_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, config.grad_clip
            )
            y_true, y_pred = evaluate(model, test_loader, device)
            metrics        = compute_fold_metrics(y_true, y_pred)
            val_f1         = metrics["macro_f1"]

            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]
            is_best    = val_f1 > best_f1
            marker     = "  [BEST]" if is_best else ""

            _log(
                f"  Epoch {epoch:3d}/{config.n_epochs} | "
                f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_F1={val_f1:.4f} | lr={current_lr:.3e} | "
                f"{time.time() - t0:.1f}s{marker}",
                log_file,
            )

            if is_best:
                best_f1, best_metrics, best_epoch = val_f1, metrics, epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "eeg_mean": eeg_mean, "eeg_std": eeg_std,
                    "gsr_mean": float(gsr_mean), "gsr_std": float(gsr_std),
                    "config":   dataclasses.asdict(config),
                    "metrics":  metrics,
                    "epoch":    epoch,
                    "eval_mode":  "5fold_group",
                    "test_subjects": test_subj_ids,
                }, os.path.join(config.checkpoint_dir, f"5fold_group_fold{fold_idx + 1:02d}_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    _log(f"  Early stop at epoch {epoch}", log_file)
                    break

        _log(dash, log_file)
        _log(
            f"  Fold {fold_idx + 1} result: best macro-F1 = {best_f1:.4f} @ epoch {best_epoch}",
            log_file,
        )
        _log(format_fold_report(best_metrics, fold_idx + 1, "+".join(test_subj_ids)), log_file)
        fold_results.append(best_metrics)

    agg = aggregate_loso_metrics(fold_results)

    summary = (
        f"\n{'=' * 60}\n"
        f"5-FOLD SUBJECT GROUPKFOLD RESULTS\n"
        f"(cross-subject — no subject leakage)\n"
        f"{'=' * 60}\n"
        f"Mean Macro-F1 : {agg['mean_macro_f1']:.4f} ± {agg['std_macro_f1']:.4f}  [PRIMARY]\n"
        f"Mean Accuracy : {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}\n"
        f"{'=' * 60}\n"
        f"Per-class mean F1:\n"
    )
    for cls, f1 in agg["per_class_mean_f1"].items():
        summary += f"  {cls:<8}: {f1:.4f}\n"
    summary += f"{'=' * 60}"

    _log(summary, log_file)

    summary_path = os.path.join(config.log_dir, "5fold_group_summary.json")
    _save_summary_json(agg, fold_results, [f"fold{i+1}" for i in range(5)], summary_path)
    _log(f"\nSummary JSON: {summary_path}", log_file)
    log_file.close()

    print(summary)
    print(f"\nLog     : {log_path}")
    print(f"Summary : {summary_path}")
    return agg


# ---------------------------------------------------------------------------
# 10-Fold Stratified KFold training (within-subject, AlgoRidge-style)
# ---------------------------------------------------------------------------

def train_10fold_stratified(
    subject_data : Dict[str, SubjectData],
    config       : TrainConfig,
    device       : torch.device,
    seed         : int = 42,
) -> Dict:
    """
    Train with 10-fold stratified KFold on the full window pool.

    Windows from all subjects are pooled and split randomly into 10 folds,
    stratified by class label. The same subject may appear in both train and
    test within a fold (within-subject leakage — intentional for comparison).

    This mirrors AlgoRidge's protocol and produces inflated results compared
    to LOSO or 5-fold GroupKFold. Use only for comparison, not as primary
    metric.

    Returns aggregated metrics dict.
    """
    from sklearn.model_selection import StratifiedKFold

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    log_path = os.path.join(config.log_dir, "10fold_stratified_training.log")
    log_file = open(log_path, "a", encoding="utf-8")

    _log(f"\n{'#' * 78}", log_file)
    _log(f"# MedOracle — PhysiologicalNet 10-Fold Stratified KFold (AlgoRidge-style)", log_file)
    _log(f"# NOTE: within-subject leakage — for comparison only, not primary metric", log_file)
    _log(f"# Device : {device}", log_file)
    _log(f"{'#' * 78}", log_file)

    # Pool all windows across all subjects
    subject_ids = sorted(subject_data.keys())
    all_windows = []
    for sid in subject_ids:
        all_windows.extend(subject_data[sid].windows)

    all_labels = np.array([w.label_int for w in all_windows])
    _log(f"Total windows: {len(all_windows):,} across {len(subject_ids)} subjects", log_file)

    skf          = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_windows, all_labels)):
        train_windows = [all_windows[i] for i in train_idx]
        test_windows  = [all_windows[i] for i in test_idx]

        sep  = "=" * 78
        dash = "-" * 78
        _log(f"\n{sep}", log_file)
        _log(
            f"10-Fold Stratified — Fold {fold_idx + 1}/10 | "
            f"Train: {len(train_windows):,}  Test: {len(test_windows):,}", log_file
        )
        _log(sep, log_file)

        train_ds, test_ds, eeg_prep, gsr_prep = build_kfold_datasets(
            train_windows, test_windows, augment_train=config.augment_train
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, drop_last=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=config.batch_size * 2, shuffle=False,
            num_workers=config.num_workers,
        )

        class_weights = compute_class_weights(train_ds.labels, device=device)
        weight_str    = "  ".join(
            f"{n}={float(class_weights[i]):.3f}"
            for i, n in enumerate(["stress", "calm", "happy", "sad", "angry"])
        )
        _log(f"  Class weights: {weight_str}", log_file)

        model = PhysiologicalNet(
            d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.n_layers, dropout=config.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=config.lr_patience,
            factor=config.lr_factor, min_lr=config.min_lr,
        )

        best_f1, best_metrics, best_epoch = -1.0, None, 0
        patience_counter = 0
        eeg_mean, eeg_std = eeg_prep.get_stats()
        gsr_mean, gsr_std = gsr_prep.get_stats()

        for epoch in range(1, config.n_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, config.grad_clip
            )
            y_true, y_pred = evaluate(model, test_loader, device)
            metrics        = compute_fold_metrics(y_true, y_pred)
            val_f1         = metrics["macro_f1"]

            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]
            is_best    = val_f1 > best_f1
            marker     = "  [BEST]" if is_best else ""

            _log(
                f"  Epoch {epoch:3d}/{config.n_epochs} | "
                f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_F1={val_f1:.4f} | lr={current_lr:.3e} | "
                f"{time.time() - t0:.1f}s{marker}",
                log_file,
            )

            if is_best:
                best_f1, best_metrics, best_epoch = val_f1, metrics, epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "eeg_mean": eeg_mean, "eeg_std": eeg_std,
                    "gsr_mean": float(gsr_mean), "gsr_std": float(gsr_std),
                    "config":   dataclasses.asdict(config),
                    "metrics":  metrics,
                    "epoch":    epoch,
                    "eval_mode": "10fold_stratified",
                    "fold_idx":  fold_idx,
                }, os.path.join(config.checkpoint_dir, f"10fold_strat_fold{fold_idx + 1:02d}_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    _log(f"  Early stop at epoch {epoch}", log_file)
                    break

        _log(dash, log_file)
        _log(
            f"  Fold {fold_idx + 1} result: best macro-F1 = {best_f1:.4f} @ epoch {best_epoch}",
            log_file,
        )
        _log(format_fold_report(best_metrics, fold_idx + 1, f"stratified_fold{fold_idx+1}"), log_file)
        fold_results.append(best_metrics)

    agg = aggregate_loso_metrics(fold_results)

    summary = (
        f"\n{'=' * 60}\n"
        f"10-FOLD STRATIFIED KFOLD RESULTS (AlgoRidge-style)\n"
        f"WARNING: within-subject leakage — inflated scores\n"
        f"Use only for protocol comparison, not as primary metric\n"
        f"{'=' * 60}\n"
        f"Mean Macro-F1 : {agg['mean_macro_f1']:.4f} ± {agg['std_macro_f1']:.4f}  [COMPARISON]\n"
        f"Mean Accuracy : {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}\n"
        f"{'=' * 60}\n"
        f"Per-class mean F1:\n"
    )
    for cls, f1 in agg["per_class_mean_f1"].items():
        summary += f"  {cls:<8}: {f1:.4f}\n"
    summary += f"{'=' * 60}"

    _log(summary, log_file)

    summary_path = os.path.join(config.log_dir, "10fold_stratified_summary.json")
    _save_summary_json(agg, fold_results, [f"fold{i+1}" for i in range(10)], summary_path)
    _log(f"\nSummary JSON: {summary_path}", log_file)
    log_file.close()

    print(summary)
    print(f"\nLog     : {log_path}")
    print(f"Summary : {summary_path}")
    return agg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PhysiologicalNet with LOSO cross-validation on DEAP"
    )
    parser.add_argument(
        "--data_dir",
        type    = str,
        default = "data/DEAP",
        help    = "Path to folder containing s01.dat … s32.dat",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type    = str,
        default = "member1_physiological/checkpoints",
        help    = "Directory where per-fold .pt checkpoints are saved",
    )
    parser.add_argument(
        "--log_dir",
        type    = str,
        default = "member1_physiological/logs",
        help    = "Directory for training logs and summary JSON",
    )
    parser.add_argument(
        "--test_subject",
        type    = str,
        default = None,
        help    = "Run a single fold for this subject only (e.g. s01). "
                  "Omit to run all 32 folds.",
    )
    parser.add_argument("--epochs",        type=int,   default=80,    help="Max epochs per fold")
    parser.add_argument("--batch_size",    type=int,   default=64,    help="Training batch size")
    parser.add_argument("--lr",            type=float, default=1e-3,  help="Adam learning rate")
    parser.add_argument("--weight_decay",  type=float, default=1e-4,  help="L2 regularisation")
    parser.add_argument("--patience",      type=int,   default=15,    help="Early stopping patience")
    parser.add_argument("--lr_patience",   type=int,   default=7,     help="LR scheduler patience")
    parser.add_argument("--dropout",       type=float, default=0.3,   help="Dropout rate")
    parser.add_argument("--device",        type=str,   default="auto",
                        help="Device: 'auto' | 'cuda' | 'cpu'")
    parser.add_argument("--no_augment",    action="store_true",
                        help="Disable training data augmentation")
    parser.add_argument(
        "--eval_mode",
        type    = str,
        default = "loso",
        choices = ["loso", "random_split", "5fold", "10fold"],
        help    = (
            "loso         : Leave-One-Subject-Out, 32 folds (most rigorous) | "
            "5fold        : Subject-wise GroupKFold, 5 folds (cross-subject, faster) | "
            "10fold       : Stratified KFold, 10 folds (within-subject, AlgoRidge-style) | "
            "random_split : 80/20 random split (demo model)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = get_device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Build config from CLI args
    config = TrainConfig(
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        batch_size     = args.batch_size,
        n_epochs       = args.epochs,
        patience       = args.patience,
        lr_patience    = args.lr_patience,
        dropout        = args.dropout,
        augment_train  = not args.no_augment,
        checkpoint_dir = args.checkpoint_dir,
        log_dir        = args.log_dir,
    )

    # Load DEAP dataset
    print(f"\nLoading DEAP data from: {args.data_dir}")
    loader       = DEAPLoader(data_dir=args.data_dir, verbose=True)
    subject_data = loader.load_all()

    # Route to selected evaluation mode
    if args.eval_mode == "random_split":
        print("\n[Eval mode: RANDOM SPLIT 80/20 — demo model]")
        train_random_split(
            subject_data = subject_data,
            config       = config,
            device       = device,
        )
    elif args.eval_mode == "5fold":
        print("\n[Eval mode: 5-FOLD SUBJECT GROUPKFOLD — cross-subject]")
        train_5fold_group(
            subject_data = subject_data,
            config       = config,
            device       = device,
        )
    elif args.eval_mode == "10fold":
        print("\n[Eval mode: 10-FOLD STRATIFIED — within-subject (AlgoRidge-style)]")
        train_10fold_stratified(
            subject_data = subject_data,
            config       = config,
            device       = device,
        )
    else:
        print("\n[Eval mode: LOSO — rigorous evaluation]")
        train_loso(
            subject_data = subject_data,
            config       = config,
            device       = device,
            test_subject = args.test_subject,
        )


if __name__ == "__main__":
    main()

"""
Commands to run:

Single fold (fast debug — tests the full pipeline on s01 only):
    python -m member1_physiological.train --data_dir data/DEAP --test_subject s01

Full LOSO (run overnight — saves 32 checkpoints):
    python -m member1_physiological.train --data_dir data/DEAP

With GPU (if available):
    python -m member1_physiological.train --data_dir data/DEAP --device cuda

Checkpoints saved to: member1_physiological/checkpoints/fold_XX_sXX_best.pt
Training log saved to: member1_physiological/logs/loso_training.log
Summary JSON saved to: member1_physiological/logs/loso_summary.json
"""
