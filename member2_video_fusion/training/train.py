"""
member2_video_fusion/training/train.py
=======================================
MedOracle — Member 2 (Vanaiyan)

5-fold GroupKFold training script for VideoEmotionModel on CREMA-D.

Usage (local):
    python member2_video_fusion/training/train.py

Usage (Google Colab — paste as a single cell):
    See the Colab block version at the bottom of this file.

Training setup
--------------
    Optimizer  : Adam (lr=1e-4, weight_decay=1e-4)
    Scheduler  : ReduceLROnPlateau (patience=3, factor=0.5)
    Loss       : Weighted cross-entropy (class weights from sklearn)
    Epochs     : 20 per fold
    Batch size : 16
    Folds      : 5 (actor-independent GroupKFold)
    Metric     : Macro-averaged F1 (primary), per-class F1, accuracy

Checkpoints
-----------
    Best model per fold saved to: checkpoints/fold_{k}_best.pt
    Final summary saved to:       checkpoints/training_summary.json

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from member2_video_fusion.models.video_model import VideoEmotionModel
from member2_video_fusion.preprocessing.dataset import CREMADDataset, get_folds
from shared.data_contracts import EMOTION_CLASSES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MANIFEST_PATH = _PROJECT_ROOT / "data" / "CREMA-D" / "manifest.csv"
CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 16
NUM_EPOCHS  = 20
LR          = 1e-4
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 2        # set to 0 on Windows or if multiprocessing errors occur

EMOTION_LABELS = list(EMOTION_CLASSES.keys())   # ["stress","calm","happy","sad","angry"]
IDX_TO_EMOTION = {v: k for k, v in EMOTION_CLASSES.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_class_weights(dataset: CREMADDataset, indices: list[int]) -> torch.Tensor:
    """Compute class weights for weighted cross-entropy from training indices."""
    labels = [dataset[i]["label"] for i in indices]
    classes = np.array(sorted(EMOTION_CLASSES.values()))
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, macro_f1)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        clips  = batch["clip"].to(device)     # (B, T, 3, 224, 224)
        labels = batch["label"].to(device)    # (B,)

        optimizer.zero_grad()
        out    = model(clips)
        loss   = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = out["predicted_class"].cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """Run validation. Returns (avg_loss, macro_f1, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        clips  = batch["clip"].to(device)
        labels = batch["label"].to(device)

        out  = model(clips)
        loss = criterion(out["logits"], labels)

        total_loss += loss.item() * clips.size(0)
        preds = out["predicted_class"].cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    device = VideoEmotionModel.get_device()
    print(f"\n{'='*60}")
    print(f"  MedOracle — VideoEmotionModel Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {NUM_EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print(f"{'='*60}\n")

    # Load full dataset (augmentation OFF — we apply per split)
    dataset = CREMADDataset(
        manifest_path=str(MANIFEST_PATH),
        augment=False,
    )
    print(f"Dataset loaded: {len(dataset)} clips, {len(EMOTION_LABELS)} classes\n")

    # Get 5-fold splits
    folds = get_folds(dataset)

    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_num}/5   |  train={len(train_indices)}  val={len(val_indices)}")
        print(f"{'─'*60}")

        # ── Class weights from training split ──────────────────────────────
        class_weights = get_class_weights(dataset, train_indices).to(device)

        # ── DataLoaders ────────────────────────────────────────────────────
        # Enable augmentation on training subset
        train_dataset = CREMADDataset(
            manifest_path=str(MANIFEST_PATH),
            augment=True,
        )
        train_subset = Subset(train_dataset, train_indices)
        val_subset   = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=(device.type != "cpu"),
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(device.type != "cpu"),
        )

        # ── Model, loss, optimiser ─────────────────────────────────────────
        model = VideoEmotionModel(pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=True
        )

        # ── Training loop ──────────────────────────────────────────────────
        best_val_f1  = 0.0
        best_epoch   = 0
        ckpt_path    = CHECKPOINT_DIR / f"fold_{fold_num}_best.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()

            train_loss, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_f1, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step(val_f1)

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint for this fold
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch  = epoch
                torch.save({
                    "fold":       fold_num,
                    "epoch":      epoch,
                    "val_f1":     val_f1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_weights": class_weights.cpu(),
                }, ckpt_path)
                print(f"    ✓ Best checkpoint saved (val F1={val_f1:.4f})")

        # ── Per-fold report ────────────────────────────────────────────────
        print(f"\n  Best Fold {fold_num} — Epoch {best_epoch}, Val Macro-F1: {best_val_f1:.4f}")

        # Reload best checkpoint for final classification report
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        report = classification_report(
            val_labels, val_preds,
            target_names=EMOTION_LABELS,
            digits=4,
            zero_division=0,
        )
        print(f"\n  Classification Report — Fold {fold_num}:\n")
        print(report)

        fold_results.append({
            "fold":        fold_num,
            "best_epoch":  best_epoch,
            "val_macro_f1": best_val_f1,
            "checkpoint":  str(ckpt_path),
        })

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — 5-Fold Summary")
    print(f"{'='*60}")

    f1_scores = [r["val_macro_f1"] for r in fold_results]
    for r in fold_results:
        print(f"  Fold {r['fold']} | Best Epoch {r['best_epoch']} | Val Macro-F1: {r['val_macro_f1']:.4f}")
    print(f"\n  Mean Macro-F1 : {np.mean(f1_scores):.4f}")
    print(f"  Std  Macro-F1 : {np.std(f1_scores):.4f}")
    print(f"\n  Checkpoints saved to: {CHECKPOINT_DIR}/\n")

    # Save summary JSON
    summary = {
        "mean_macro_f1": float(np.mean(f1_scores)),
        "std_macro_f1":  float(np.std(f1_scores)),
        "folds":         fold_results,
        "config": {
            "batch_size":   BATCH_SIZE,
            "num_epochs":   NUM_EPOCHS,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "device":       str(device),
        },
    }
    summary_path = CHECKPOINT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
