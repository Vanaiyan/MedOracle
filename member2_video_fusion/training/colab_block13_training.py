# ============================================================
# BLOCK 13 — Training (5-Fold GroupKFold)
# Run this in Google Colab after Blocks 1–12 pass.
# Expected time: ~2–3 hours on T4 GPU for all 5 folds.
# ============================================================

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from google.colab import drive

# ── Mount Drive to save checkpoints (won't be lost if session ends) ────────
drive.mount('/content/drive')
DRIVE_CKPT_DIR = "/content/drive/MyDrive/MedOracle/checkpoints"
import os; os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

# ── Colab paths (set these to match your Colab setup) ──────────────────────
import sys
sys.path.insert(0, "/content/MedOracle")           # project root in Colab

from member2_video_fusion.models.video_model import VideoEmotionModel
from member2_video_fusion.preprocessing.dataset import CREMADDataset, get_folds
from shared.data_contracts import EMOTION_CLASSES

MANIFEST_PATH = "/content/MedOracle/data/CREMA-D/manifest.csv"
EMOTION_LABELS = list(EMOTION_CLASSES.keys())

# ── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE   = 16
NUM_EPOCHS   = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2

device = VideoEmotionModel.get_device()
print(f"Device: {device}")

# ── Helper: class weights ───────────────────────────────────────────────────
def get_class_weights(dataset, indices):
    labels = [dataset[i]["label"] for i in indices]
    classes = np.array(sorted(EMOTION_CLASSES.values()))
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float32)

# ── Helper: one training epoch ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        clips  = batch["clip"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(clips)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        all_preds.extend(out["predicted_class"].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1

# ── Helper: validation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        clips  = batch["clip"].to(device)
        labels = batch["label"].to(device)
        out    = model(clips)
        loss   = criterion(out["logits"], labels)
        total_loss += loss.item() * clips.size(0)
        all_preds.extend(out["predicted_class"].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels

# ── Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset     = CREMADDataset(manifest_path=MANIFEST_PATH, augment=False)
dataset_aug = CREMADDataset(manifest_path=MANIFEST_PATH, augment=True)
folds       = get_folds(dataset)
print(f"  Total clips : {len(dataset)}")
print(f"  Folds       : {len(folds)}")

# ── 5-Fold Training Loop ────────────────────────────────────────────────────
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    fold_num = fold_idx + 1
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num}/5   train={len(train_idx)}   val={len(val_idx)}")
    print(f"{'='*60}")

    # Class weights
    class_weights = get_class_weights(dataset, train_idx).to(device)

    # DataLoaders
    train_loader = DataLoader(
        Subset(dataset_aug, train_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Model
    model     = VideoEmotionModel(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_val_f1 = 0.0
    best_epoch  = 0
    ckpt_path   = f"{DRIVE_CKPT_DIR}/fold_{fold_num}_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(f"  Ep {epoch:02d}/{NUM_EPOCHS} | "
              f"Train L={train_loss:.4f} F1={train_f1:.4f} | "
              f"Val L={val_loss:.4f} F1={val_f1:.4f} | {elapsed:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save({
                "fold": fold_num, "epoch": epoch, "val_f1": val_f1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_weights": class_weights.cpu(),
            }, ckpt_path)
            print(f"    ✓ Saved best (F1={val_f1:.4f}) → {ckpt_path}")

    # Classification report on best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print(f"\nClassification Report — Fold {fold_num} (best epoch {best_epoch}):")
    print(classification_report(val_labels, val_preds, target_names=EMOTION_LABELS, digits=4, zero_division=0))

    fold_results.append({
        "fold": fold_num, "best_epoch": best_epoch,
        "val_macro_f1": best_val_f1, "checkpoint": ckpt_path
    })

# ── Final summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE — 5-Fold Summary")
print(f"{'='*60}")
f1_scores = [r["val_macro_f1"] for r in fold_results]
for r in fold_results:
    print(f"  Fold {r['fold']} | Best Epoch {r['best_epoch']:02d} | Val Macro-F1: {r['val_macro_f1']:.4f}")
print(f"\n  Mean Macro-F1 : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

summary = {
    "mean_macro_f1": float(np.mean(f1_scores)),
    "std_macro_f1":  float(np.std(f1_scores)),
    "folds": fold_results,
    "config": {"batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
               "lr": LR, "weight_decay": WEIGHT_DECAY}
}
summary_path = f"{DRIVE_CKPT_DIR}/training_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved → {summary_path}")
