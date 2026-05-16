"""
member2_video_fusion/training/kaggle_retrain_v2.py
===================================================
MedOracle — Member 2 (Vanaiyan Kirupagaran, 214215H)

Retrain v2: CREMA-D + RAVDESS unified 5-fold training
======================================================

Prerequisite: run kaggle_ravdess_extract.py first and commit its output
as a Kaggle dataset, then update RAVDESS_MANIFEST_PATH below.

What changed vs the v1 training (0.6403 mean F1)
-------------------------------------------------
  Fix 1  encoder_drop=0.3        — regularise the largest sub-network
  Fix 2  early stopping p=5      — stop before overfitting, not after
  Fix 3  ReduceLROnPlateau p=3   — decay LR when val_F1 stagnates
  Fix 4  augmentation            — flip + color jitter (with saturation) + rotation ±10°
  Fix 5  CREMA-D + RAVDESS       — 115 actors across two recording styles
  Fix 6  mixed precision (AMP)   — same as before, kept from v1

Kaggle dataset paths (update these to match your actual dataset slugs)
----------------------------------------------------------------------
  CREMA-D .npy :  /kaggle/input/datasets/vanaiyan/cremad-npy-frames/
  RAVDESS .npy :  /kaggle/input/datasets/vanaiyan/ravdess-npy-frames/
                  (after committing kaggle_ravdess_extract.py output)
  FYP code     :  /kaggle/input/datasets/vanaiyan/medracle-fyp-code/FYP/

Training setup
--------------
  Dataset    : CREMA-D (91 actors) + RAVDESS (24 actors) = 115 actors
  Folds      : 5-fold actor-independent GroupKFold
  Batch size : 16
  Max epochs : 30  (early stopping will likely trigger sooner)
  LR         : 1e-4  (AdamW with weight decay 1e-4)
  Scheduler  : ReduceLROnPlateau (patience=3, factor=0.5, mode=max)
  Early stop : patience=5 on val_macro_f1 (no improvement → stop)
  Metric     : Macro-averaged F1 (primary, handles class imbalance)
  Loss       : Weighted cross-entropy (balanced class weights per fold)
  AMP        : torch.amp.autocast + GradScaler (CUDA only)

Checkpoints
-----------
  /kaggle/working/checkpoints_v2/fold_{k}_best.pt   — best model per fold
  /kaggle/working/checkpoints_v2/training_summary_v2.json

Author: Vanaiyan Kirupagaran (214215H)
"""

# ============================================================
# CELL 1 — Setup: copy FYP code into working directory
# ============================================================

import os
import sys
import shutil

# Kaggle input filesystem does not support Python import directly.
# Copy the code to /kaggle/working/ first.
_FYP_SRC = "/kaggle/input/datasets/vanaiyan/medracle-fyp-code/FYP"

for _subdir in ("member2_video_fusion", "shared"):
    _dst = f"/kaggle/working/{_subdir}"
    if os.path.exists(_dst):
        shutil.rmtree(_dst)
    shutil.copytree(f"{_FYP_SRC}/{_subdir}", _dst)
    print(f"✓ Copied {_subdir} → {_dst}")

sys.path.insert(0, "/kaggle/working")
print(f"\nsys.path[0] = /kaggle/working")


# ============================================================
# CELL 2 — Imports
# ============================================================

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from member2_video_fusion.models.video_model import VideoEmotionModel
from member2_video_fusion.preprocessing.multi_corpus_dataset import (
    MultiCorpusDataset,
    get_unified_folds,
    load_unified_manifest,
    multicorpus_collate_fn,
)
from shared.data_contracts import EMOTION_CLASSES

print("✓ All imports successful")
print(f"  PyTorch version : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3 — Configuration
# ============================================================

# ── Kaggle dataset paths ───────────────────────────────────────────────────────
CREMAD_NPY_DIR       = Path("/kaggle/input/datasets/vanaiyan/cremad-npy-frames")
CREMAD_MANIFEST_PATH = Path("/kaggle/input/datasets/vanaiyan/cremad-npy-frames/manifest.csv")

# Update this path after committing kaggle_ravdess_extract.py output
RAVDESS_MANIFEST_PATH = Path("/kaggle/input/datasets/vanaiyan/ravdess-npy-frames/ravdess_manifest.csv")

# ── Output ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints_v2")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_MANIFEST_PATH = Path("/kaggle/working/unified_manifest.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE       = 16
MAX_EPOCHS       = 30         # early stopping will fire before this in most folds
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
NUM_WORKERS      = 2          # Kaggle T4 can handle 2 workers
ENCODER_DROPOUT  = 0.3        # Fix 1: was 0.0 in v1
EARLY_STOP_PAT   = 5          # Fix 2: patience in epochs
SCHED_PATIENCE   = 3          # Fix 3: ReduceLROnPlateau patience
SCHED_FACTOR     = 0.5        # halve LR on plateau
LABEL_SMOOTHING  = 0.1        # Fix 7: prevents overconfident hard-label learning
GRAD_CLIP_NORM   = 1.0        # Fix 8: caps gradient magnitude, stabilises updates
N_FOLDS          = 5

USE_AMP = torch.cuda.is_available()   # AMP only on CUDA

EMOTION_LABELS = list(EMOTION_CLASSES.keys())   # ["stress","calm","happy","sad","angry"]
IDX_TO_EMOTION = {v: k for k, v in EMOTION_CLASSES.items()}

print(f"\n{'='*60}")
print(f"  MedOracle — VideoEmotionModel Retrain v2")
print(f"{'='*60}")
print(f"  MAX_EPOCHS      : {MAX_EPOCHS} (early-stopping patience={EARLY_STOP_PAT})")
print(f"  BATCH_SIZE      : {BATCH_SIZE}")
print(f"  LR              : {LR}  WD={WEIGHT_DECAY}")
print(f"  ENCODER_DROP    : {ENCODER_DROPOUT}  (was 0.0 in v1)")
print(f"  SCHEDULER       : ReduceLROnPlateau(patience={SCHED_PATIENCE}, factor={SCHED_FACTOR})")
print(f"  LABEL_SMOOTHING : {LABEL_SMOOTHING}   (prevents overconfident predictions)")
print(f"  GRAD_CLIP_NORM  : {GRAD_CLIP_NORM}     (max gradient norm before clipping)")
print(f"  MIXED PRECISION : {USE_AMP}")
print(f"  CHECKPOINT DIR  : {CHECKPOINT_DIR}")


# ============================================================
# CELL 4 — Build unified manifest (CREMA-D + RAVDESS)
# ============================================================

def build_unified_manifest(
    cremad_manifest_path: Path,
    ravdess_manifest_path: Path,
    output_path: Path,
) -> list:
    """Merge CREMA-D and RAVDESS manifests into a single unified CSV.

    CREMA-D manifest CSV is expected to have columns:
        path (video path), actor_id, emotion, emotion_int, raw_label, ...
    The .npy path is derived by replacing the video file suffix with .npy
    and the directory with CREMAD_NPY_DIR.

    RAVDESS manifest CSV (from kaggle_ravdess_extract.py) has columns:
        npy_path, actor_id, source, emotion, emotion_int

    Unified manifest columns (written to output_path):
        npy_path, actor_id, source, emotion, emotion_int

    Returns
    -------
    List of unified row dicts.
    """
    all_rows = []

    # ── 1. CREMA-D rows ────────────────────────────────────────────────────────
    cremad_skipped = 0
    with open(cremad_manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            emotion_str = row.get("emotion", "").strip().lower()
            # Skip disgust rows (already filtered in original manifest, but guard)
            if emotion_str not in EMOTION_CLASSES:
                cremad_skipped += 1
                continue

            # Derive .npy path from original video path
            video_path = Path(row["path"])
            npy_path   = CREMAD_NPY_DIR / (video_path.stem + ".npy")

            if not npy_path.exists():
                cremad_skipped += 1
                continue

            all_rows.append({
                "npy_path":    str(npy_path),
                "actor_id":    int(row["actor_id"]),
                "source":      "cremad",
                "emotion":     emotion_str,
                "emotion_int": int(EMOTION_CLASSES[emotion_str]),
            })

    cremad_total = len([r for r in all_rows if r["source"] == "cremad"])
    print(f"  CREMA-D : {cremad_total} rows loaded, {cremad_skipped} skipped")

    # ── 2. RAVDESS rows ────────────────────────────────────────────────────────
    ravdess_total   = 0
    ravdess_skipped = 0
    with open(ravdess_manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            npy_path   = Path(row["npy_path"])
            emotion_str = row["emotion"].strip().lower()

            if not npy_path.exists() or emotion_str not in EMOTION_CLASSES:
                ravdess_skipped += 1
                continue

            all_rows.append({
                "npy_path":    str(npy_path),
                "actor_id":    int(row["actor_id"]),    # already offset 2001–2024
                "source":      "ravdess",
                "emotion":     emotion_str,
                "emotion_int": int(EMOTION_CLASSES[emotion_str]),
            })
            ravdess_total += 1

    print(f"  RAVDESS : {ravdess_total} rows loaded, {ravdess_skipped} skipped")

    # ── 3. Write unified manifest ──────────────────────────────────────────────
    fieldnames = ["npy_path", "actor_id", "source", "emotion", "emotion_int"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Total   : {len(all_rows)} rows → {output_path}")
    return all_rows


print("\nBuilding unified manifest...")
unified_rows = build_unified_manifest(
    CREMAD_MANIFEST_PATH,
    RAVDESS_MANIFEST_PATH,
    UNIFIED_MANIFEST_PATH,
)

# ── Summary stats ──────────────────────────────────────────────────────────────
src_counts   = Counter(r["source"]  for r in unified_rows)
emo_counts   = Counter(r["emotion"] for r in unified_rows)
actor_counts = Counter(r["actor_id"] for r in unified_rows)

print(f"\nUnified dataset summary:")
print(f"  Clips per source  : {dict(src_counts)}")
print(f"  Unique actors     : {len(actor_counts)} "
      f"(CREMA-D: {sum(1 for a in actor_counts if a < 2000)}, "
      f"RAVDESS: {sum(1 for a in actor_counts if a >= 2000)})")
print(f"  Emotion breakdown :")
for emo in EMOTION_LABELS:
    print(f"    {emo:8s}: {emo_counts.get(emo, 0)}")


# ============================================================
# CELL 5 — Generate 5-fold GroupKFold splits
# ============================================================

print(f"\nGenerating {N_FOLDS}-fold actor-independent GroupKFold splits...")
folds = get_unified_folds(UNIFIED_MANIFEST_PATH, n_splits=N_FOLDS)

for i, (train_rows, val_rows) in enumerate(folds):
    train_actors = set(r["actor_id"] for r in train_rows)
    val_actors   = set(r["actor_id"] for r in val_rows)
    overlap      = train_actors & val_actors
    train_src    = Counter(r["source"] for r in train_rows)
    val_src      = Counter(r["source"] for r in val_rows)
    assert len(overlap) == 0, f"Actor overlap in fold {i+1}!"
    print(f"  Fold {i+1}: train={len(train_rows):>5} clips / {len(train_actors):>3} actors "
          f"{dict(train_src)} | "
          f"val={len(val_rows):>4} clips / {len(val_actors):>3} actors "
          f"{dict(val_src)}")

print(f"\n✓ All {N_FOLDS} folds: no actor overlap")


# ============================================================
# CELL 6 — Training helper functions
# ============================================================

def get_class_weights(rows: list) -> torch.Tensor:
    """Compute balanced class weights from a list of manifest rows.

    Uses sklearn's 'balanced' mode: w_k = n_samples / (n_classes * n_k).
    Passed to CrossEntropyLoss to handle class imbalance without SMOTE.
    """
    labels  = np.array([r["emotion_int"] for r in rows])
    classes = np.array(sorted(EMOTION_CLASSES.values()))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    scaler,
) -> tuple[float, float]:
    """Run one training epoch with mixed-precision support.

    Parameters
    ----------
    scaler : torch.amp.GradScaler instance (or None if not using AMP)

    Returns
    -------
    (avg_loss, macro_f1) for the epoch.
    """
    model.train()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    for batch in loader:
        clips  = batch["clip"].to(device)    # (B, T, 3, 224, 224)
        labels = batch["label"].to(device)   # (B,)

        optimizer.zero_grad()

        if USE_AMP and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                out  = model(clips)
                loss = criterion(out["logits"], labels)
            scaler.scale(loss).backward()
            # Fix 8: unscale before clipping so the clip threshold is in
            # the original gradient space, not the scaled one.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(clips)
            loss = criterion(out["logits"], labels)
            loss.backward()
            # Fix 8: gradient clipping — caps update magnitude, prevents
            # a single bad batch from blowing up learned weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item() * clips.size(0)
        all_preds.extend(out["predicted_class"].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, list, list]:
    """Run validation (no gradients, no AMP).

    Returns
    -------
    (avg_loss, macro_f1, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        clips  = batch["clip"].to(device)
        labels = batch["label"].to(device)

        out  = model(clips)
        loss = criterion(out["logits"], labels)

        total_loss += loss.item() * clips.size(0)
        all_preds.extend(out["predicted_class"].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels


class EarlyStopping:
    """Track val_macro_f1; signal stop after `patience` epochs with no improvement.

    Usage
    -----
        es = EarlyStopping(patience=5)
        for epoch in ...:
            val_f1 = ...
            if es.step(val_f1):
                break   # early stop triggered

    Attributes
    ----------
    best_f1   : float — best val_macro_f1 seen so far
    best_epoch: int   — epoch at which best_f1 was achieved
    counter   : int   — consecutive epochs without improvement
    """

    def __init__(self, patience: int = 5):
        self.patience   = patience
        self.best_f1    = 0.0
        self.best_epoch = 0
        self.counter    = 0

    def step(self, val_f1: float, epoch: int) -> bool:
        """Update state. Returns True if training should stop."""
        if val_f1 > self.best_f1:
            self.best_f1    = val_f1
            self.best_epoch = epoch
            self.counter    = 0
            return False   # improvement — continue
        else:
            self.counter += 1
            return self.counter >= self.patience   # stop if patience exhausted


print("✓ Training helpers defined")


# ============================================================
# CELL 7 — Main 5-fold training loop
# ============================================================

device = VideoEmotionModel.get_device()
print(f"\nDevice: {device}")

fold_results = []

for fold_idx, (train_rows, val_rows) in enumerate(folds):
    fold_num = fold_idx + 1

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num}/{N_FOLDS}   "
          f"train={len(train_rows)} clips   val={len(val_rows)} clips")
    print(f"{'='*60}")

    # ── Class weights from training split ──────────────────────────────────────
    class_weights = get_class_weights(train_rows).to(device)
    print(f"  Class weights: { {e: round(float(class_weights[i]),3) for e,i in EMOTION_CLASSES.items()} }")

    # ── Datasets ────────────────────────────────────────────────────────────────
    # Fix 4: augment=True for training only
    train_ds = MultiCorpusDataset(train_rows, augment=True,  skip_errors=True)
    val_ds   = MultiCorpusDataset(val_rows,   augment=False, skip_errors=True)

    print(f"  Dataset sizes  : train={len(train_ds)}  val={len(val_ds)}")

    # ── DataLoaders ─────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=multicorpus_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=multicorpus_collate_fn,
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    # Fix 1: encoder_drop=0.3 (was 0.0 in v1)
    model = VideoEmotionModel(
        pretrained=True,
        encoder_drop=ENCODER_DROPOUT,
        lstm_drop=0.3,
        fc_drop=0.4,
    ).to(device)

    params = model.param_summary()
    print(f"  Model params   : {params['total']:,} total / "
          f"{params['trainable']:,} trainable ({params['trainable_pct']}%)")

    # ── Loss, optimiser, scheduler ──────────────────────────────────────────────
    # Fix 7: label_smoothing=0.1 — soft targets prevent the model from
    # learning overconfident outputs (e.g. 0.99 for one class), which is
    # one of the primary causes of the v1 overfitting pattern.
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = optim.AdamW(   # AdamW = Adam + decoupled weight decay (slight improvement)
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Fix 3: ReduceLROnPlateau on val_macro_f1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=SCHED_PATIENCE,
        factor=SCHED_FACTOR,
        min_lr=1e-6,
    )

    # Fix 6: AMP GradScaler (CUDA only)
    scaler = torch.amp.GradScaler() if USE_AMP else None

    # Fix 2: Early stopping
    early_stop = EarlyStopping(patience=EARLY_STOP_PAT)

    ckpt_path  = CHECKPOINT_DIR / f"fold_{fold_num}_best.pt"
    best_epoch = 0

    print(f"\n  {'Ep':>3}  {'Train Loss':>10}  {'Train F1':>8}  "
          f"{'Val Loss':>8}  {'Val F1':>6}  {'LR':>8}  {'ES':>5}  {'Δ'}")
    print(f"  {'─'*72}")

    # ── Epoch loop ──────────────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler step (Fix 3)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping check (Fix 2)
        stop = early_stop.step(val_f1, epoch)

        # Checkpoint if improved
        improved = (val_f1 >= early_stop.best_f1 and early_stop.counter == 0)
        marker   = "✓" if improved else " "

        if improved:
            best_epoch = epoch
            torch.save({
                "fold":                fold_num,
                "epoch":               epoch,
                "val_macro_f1":        val_f1,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_weights":       class_weights.cpu(),
                "config": {
                    "encoder_drop": ENCODER_DROPOUT,
                    "batch_size":   BATCH_SIZE,
                    "lr":           LR,
                    "weight_decay": WEIGHT_DECAY,
                },
            }, ckpt_path)

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {train_loss:>10.4f}  {train_f1:>8.4f}  "
              f"{val_loss:>8.4f}  {val_f1:>6.4f}  {current_lr:>8.2e}  "
              f"{early_stop.counter:>2}/{EARLY_STOP_PAT}  "
              f"{marker}  ({elapsed:.0f}s)")

        if stop:
            print(f"\n  → Early stopping at epoch {epoch} "
                  f"(best was epoch {early_stop.best_epoch}, "
                  f"F1={early_stop.best_f1:.4f})")
            break

    # ── Per-fold classification report ──────────────────────────────────────────
    print(f"\n  Best Fold {fold_num} — Epoch {best_epoch}, "
          f"Val Macro-F1: {early_stop.best_f1:.4f}")

    # Reload best checkpoint for final report
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, val_preds_best, val_labels_best = evaluate(
        model, val_loader, criterion, device
    )

    report = classification_report(
        val_labels_best, val_preds_best,
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0,
    )
    print(f"\n  Classification Report — Fold {fold_num} (best epoch {best_epoch}):\n")
    print(report)

    fold_results.append({
        "fold":         fold_num,
        "best_epoch":   best_epoch,
        "val_macro_f1": early_stop.best_f1,
        "checkpoint":   str(ckpt_path),
        "train_clips":  len(train_ds),
        "val_clips":    len(val_ds),
    })


# ============================================================
# CELL 8 — Final summary and save JSON
# ============================================================

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE — 5-Fold Summary (v2)")
print(f"{'='*60}")

f1_scores = [r["val_macro_f1"] for r in fold_results]
for r in fold_results:
    print(f"  Fold {r['fold']} | Best Epoch {r['best_epoch']:>2} | "
          f"Val Macro-F1: {r['val_macro_f1']:.4f}")

mean_f1 = float(np.mean(f1_scores))
std_f1  = float(np.std(f1_scores))

print(f"\n  ─────────────────────────────────────────")
print(f"  Mean Macro-F1 : {mean_f1:.4f}")
print(f"  Std  Macro-F1 : {std_f1:.4f}")
print(f"  v1 baseline   : 0.6403 ± 0.0328  (CREMA-D only, no augmentation)")
delta = mean_f1 - 0.6403
print(f"  Δ vs baseline : {delta:+.4f}")
print(f"  ─────────────────────────────────────────")
print(f"\n  Checkpoints   : {CHECKPOINT_DIR}/")

summary = {
    "version":       "v2",
    "mean_macro_f1": mean_f1,
    "std_macro_f1":  std_f1,
    "v1_baseline":   0.6403,
    "delta_vs_v1":   round(delta, 4),
    "folds":         fold_results,
    "config": {
        "datasets":       ["cremad", "ravdess"],
        "n_actors":       115,
        "batch_size":     BATCH_SIZE,
        "max_epochs":     MAX_EPOCHS,
        "lr":             LR,
        "weight_decay":   WEIGHT_DECAY,
        "encoder_drop":   ENCODER_DROPOUT,
        "early_stop_pat": EARLY_STOP_PAT,
        "sched_patience": SCHED_PATIENCE,
        "sched_factor":   SCHED_FACTOR,
        "augmentation":   "flip+colorjitter+rotation10deg",
        "label_smoothing": LABEL_SMOOTHING,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "use_amp":        USE_AMP,
        "device":         str(device),
    },
}

summary_path = CHECKPOINT_DIR / "training_summary_v2.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved → {summary_path}")


# ============================================================
# CELL 9 — Identify the best overall checkpoint
# ============================================================

best_fold   = max(fold_results, key=lambda r: r["val_macro_f1"])
best_ckpt   = Path(best_fold["checkpoint"])

print(f"\nBest overall checkpoint:")
print(f"  Fold         : {best_fold['fold']}")
print(f"  Best epoch   : {best_fold['best_epoch']}")
print(f"  Val Macro-F1 : {best_fold['val_macro_f1']:.4f}")
print(f"  Checkpoint   : {best_ckpt}")

print(f"\n  Use this checkpoint for:")
print(f"    - Cross-dataset evaluation on SAVEE")
print(f"    - SHAP explainability (Member 3 integration)")
print(f"    - The full run_full_pipeline() export")


# ============================================================
# CELL 10 — Cross-dataset eval scaffold (SAVEE placeholder)
# ============================================================

print("\n" + "="*60)
print("  Cross-Dataset Evaluation Scaffold")
print("="*60)
print("""
  To run cross-dataset evaluation on SAVEE (or any held-out corpus):

  1. Add SAVEE as a Kaggle dataset input.
  2. Extract SAVEE frames with the same YOLO pipeline used for RAVDESS
     (see kaggle_ravdess_extract.py for the extraction template).
  3. Load the best checkpoint above:

     ckpt = torch.load(best_ckpt, map_location=device)
     model = VideoEmotionModel(pretrained=False, encoder_drop=0.3).to(device)
     model.load_state_dict(ckpt["model_state_dict"])

  4. Build a SAVEE manifest CSV with the same columns as unified_manifest.csv.
  5. Create a MultiCorpusDataset(savee_rows, augment=False).
  6. Run evaluate() on the DataLoader.
  7. Report macro-F1 — compare to v1's ~0.29 on RAVDESS.

  Expected improvement rationale:
    - RAVDESS actors in training should directly improve generalisation
      to RAVDESS-style recordings.
    - encoder_drop=0.3 reduces CREMA-D-specific texture overfitting.
    - Color+saturation jitter reduces dataset-specific colour shift.
    - Rotation ±10° increases robustness to camera angle variation.
""")

print("✓ Retrain v2 complete. All checkpoints saved to /kaggle/working/checkpoints_v2/")
