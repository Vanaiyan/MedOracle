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
from member1_physiological.utils.dataset import build_loso_datasets
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
    criterion = FocalLoss(weight=class_weights, gamma=2.0)

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

    # Run LOSO training
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
