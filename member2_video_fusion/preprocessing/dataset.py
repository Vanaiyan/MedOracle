"""
member2_video_fusion/preprocessing/dataset.py
==============================================
MedOracle — Member 2 (Vanaiyan)

CREMADDataset: loads the manifest CSV, samples T=16 frames uniformly from
each clip, assesses video signal quality, and returns tensors ready for
the ResNet50 → BiLSTM pipeline.

Actor-independent 5-fold GroupKFold splits are generated here so every
experiment uses the exact same folds.

Usage
-----
    from member2_video_fusion.preprocessing.dataset import CREMADDataset, get_folds

    folds = get_folds(manifest_path="data/CREMA-D/manifest.csv", n_splits=5)
    for fold_idx, (train_df, val_df) in enumerate(folds):
        train_ds = CREMADDataset(train_df, augment=True)
        val_ds   = CREMADDataset(val_df,   augment=False)

Architecture note
-----------------
    Input video → Sample T=16 frames (uniform temporal stride)
               → YOLO face detection (added in face_detector.py)
               → Crop + resize to 224×224
               → ResNet50 feature extraction (models/resnet_encoder.py)
               → BiLSTM temporal modelling (models/bilstm.py)
               → Softmax → P_video[5]

Signal quality thresholds (from CLAUDE.md)
-------------------------------------------
    Face detection rate : good ≥80%, degraded 50–80%, poor <50%
    Laplacian variance  : good ≥100,  degraded 50–100,  poor <50
    Face bounding box   : good >10% frame area, degraded 5–10%, poor <5%
    Rule: worst metric wins (any poor → video = poor)

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FRAMES       = 16          # temporal depth fed into BiLSTM
FRAME_SIZE     = 224         # spatial size expected by ResNet50
CREMA_FPS      = 30.0        # all CREMA-D clips are 30 fps

# ImageNet normalisation (applied when convert_to_tensor=True)
IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Signal quality thresholds
_FACE_RATE_GOOD      = 0.80
_FACE_RATE_DEGRADED  = 0.50
_LAP_VAR_GOOD        = 100.0
_LAP_VAR_DEGRADED    = 50.0
_FACE_AREA_GOOD      = 0.10
_FACE_AREA_DEGRADED  = 0.05


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> List[dict]:
    """
    Load the CREMA-D manifest CSV produced by the dataset audit step.

    Returns a list of dicts with keys:
        path, filename, actor_id, emotion, emotion_int,
        raw_label, sentence, intensity
    """
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            row["actor_id"]    = int(row["actor_id"])
            row["emotion_int"] = int(row["emotion_int"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Actor-independent 5-fold GroupKFold
# ---------------------------------------------------------------------------

def get_folds(
    manifest_path: str | Path,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[dict], List[dict]]]:
    """
    Create actor-independent GroupKFold splits.

    Grouping by actor_id guarantees that no actor appears in both the
    training and validation sets of any fold — this is essential because
    CREMA-D actors appear in many clips and a naive random split would leak
    actor identity into the validation set.

    Parameters
    ----------
    manifest_path : path to data/CREMA-D/manifest.csv
    n_splits      : number of folds (default 5)
    seed          : random seed for reproducibility

    Returns
    -------
    List of (train_rows, val_rows) tuples, one per fold.
    """
    rows   = load_manifest(manifest_path)
    groups = np.array([r["actor_id"] for r in rows])
    X_idx  = np.arange(len(rows))          # dummy X — we only need indices

    gkf    = GroupKFold(n_splits=n_splits)
    folds  = []

    for train_idx, val_idx in gkf.split(X_idx, groups=groups):
        train_rows = [rows[i] for i in train_idx]
        val_rows   = [rows[i] for i in val_idx]
        folds.append((train_rows, val_rows))

    return folds


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames_uniform(video_path: str, n_frames: int = N_FRAMES) -> Optional[np.ndarray]:
    """
    Sample n_frames frames uniformly across the full duration of a clip.

    Parameters
    ----------
    video_path : path to the .flv (or .mp4) video file
    n_frames   : number of frames to sample (T=16 by default)

    Returns
    -------
    np.ndarray of shape (n_frames, H, W, 3) in RGB uint8, or None if the
    clip cannot be opened or has fewer frames than requested.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < n_frames:
        cap.release()
        return None

    # Uniform temporal stride: pick n_frames evenly spaced indices
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return np.stack(frames)   # (T, H, W, 3)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def resize_frames(frames: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """
    Resize all frames to (size × size).

    Parameters
    ----------
    frames : (T, H, W, 3) uint8
    size   : target spatial size

    Returns
    -------
    np.ndarray of shape (T, size, size, 3) uint8
    """
    resized = np.stack([
        cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        for f in frames
    ])
    return resized


def normalise_frames(frames: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet mean/std normalisation.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 in [0, 255]

    Returns
    -------
    np.ndarray of shape (T, H, W, 3) float32, normalised
    """
    frames_f = frames.astype(np.float32) / 255.0
    frames_f = (frames_f - IMAGENET_MEAN) / IMAGENET_STD
    return frames_f


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Convert (T, H, W, 3) float32 → (T, 3, H, W) float32 torch.Tensor.
    The channel-first format is required by PyTorch's Conv2d layers.
    """
    # (T, H, W, 3) → (T, 3, H, W)
    tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2))
    return tensor


# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------

def augment_frames(frames: np.ndarray) -> np.ndarray:
    """
    Apply random spatial augmentation to all frames in a clip consistently.

    Augmentations applied identically to every frame in the clip:
    - Horizontal flip (50% probability)
    - Random brightness/contrast jitter
    - Random crop (80–100% of original size) then resize back

    Parameters
    ----------
    frames : (T, H, W, 3) uint8

    Returns
    -------
    np.ndarray of same shape (T, H, W, 3) uint8
    """
    T, H, W, C = frames.shape

    # Horizontal flip
    if random.random() < 0.5:
        frames = frames[:, :, ::-1, :].copy()

    # Brightness/contrast jitter (applied uniformly to whole clip)
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.randint(-20, 20)    # brightness
    frames = np.clip(alpha * frames.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Random crop
    crop_ratio = random.uniform(0.80, 1.00)
    crop_h     = int(H * crop_ratio)
    crop_w     = int(W * crop_ratio)
    y0 = random.randint(0, H - crop_h)
    x0 = random.randint(0, W - crop_w)
    frames = frames[:, y0:y0+crop_h, x0:x0+crop_w, :]
    frames = np.stack([
        cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        for f in frames
    ])

    return frames


# ---------------------------------------------------------------------------
# Signal quality assessment (no YOLO — uses Laplacian sharpness only)
# ---------------------------------------------------------------------------

def assess_video_quality_basic(frames: np.ndarray) -> str:
    """
    Assess video signal quality using Laplacian variance (sharpness).

    This is the fallback quality check used before YOLO face detection
    is available. Once YOLO is integrated, use assess_video_quality_with_yolo()
    in face_detector.py instead.

    Thresholds (from CLAUDE.md):
        Laplacian variance ≥ 100  → good
                           50–100 → degraded
                           < 50   → poor

    Reference: Pech-Pacheco et al. (2000)

    Parameters
    ----------
    frames : (T, H, W, 3) uint8

    Returns
    -------
    "good" | "degraded" | "poor"
    """
    lap_vars = []
    for frame in frames:
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_vars.append(lap_var)

    mean_lap = np.mean(lap_vars)

    if mean_lap >= _LAP_VAR_GOOD:
        return "good"
    elif mean_lap >= _LAP_VAR_DEGRADED:
        return "degraded"
    else:
        return "poor"


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CREMADDataset(Dataset):
    """
    PyTorch Dataset for CREMA-D video clips.

    Each item returns:
        frames_tensor : torch.Tensor of shape (T, 3, 224, 224) float32
        label         : int — emotion class index (0–4)
        actor_id      : int — for GroupKFold bookkeeping
        video_quality : str — "good" | "degraded" | "poor"
        path          : str — original file path (for debugging)

    Parameters
    ----------
    rows        : list of manifest dicts (from load_manifest or get_folds)
    augment     : apply random augmentation (True for training, False for val)
    n_frames    : number of frames to sample per clip (default 16)
    frame_size  : spatial size after resize (default 224)
    skip_errors : if True, bad clips are skipped and replaced silently;
                  if False, an IOError is raised on bad clips
    """

    def __init__(
        self,
        rows:        List[dict],
        augment:     bool = False,
        n_frames:    int  = N_FRAMES,
        frame_size:  int  = FRAME_SIZE,
        skip_errors: bool = True,
    ):
        self.augment     = augment
        self.n_frames    = n_frames
        self.frame_size  = frame_size
        self.skip_errors = skip_errors

        # Filter out any rows whose video file doesn't exist on this machine
        self.rows = [r for r in rows if Path(r["path"]).exists()]
        missing   = len(rows) - len(self.rows)
        if missing:
            print(f"[CREMADDataset] Warning: {missing} video files not found on disk — skipped")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row  = self.rows[idx]
        path = row["path"]

        # --- 1. Sample frames ---
        frames = sample_frames_uniform(path, self.n_frames)
        if frames is None:
            if self.skip_errors:
                # Return a black-frame placeholder so DataLoader doesn't crash
                frames = np.zeros((self.n_frames, self.frame_size, self.frame_size, 3), dtype=np.uint8)
                quality = "poor"
            else:
                raise IOError(f"Cannot read video: {path}")
        else:
            quality = assess_video_quality_basic(frames)

        # --- 2. Augmentation (training only) ---
        if self.augment:
            frames = augment_frames(frames)

        # --- 3. Resize + normalise ---
        frames = resize_frames(frames, self.frame_size)
        frames = normalise_frames(frames)

        # --- 4. Convert to tensor (T, 3, H, W) ---
        tensor = frames_to_tensor(frames)

        return {
            "frames":       tensor,                   # (T, 3, 224, 224) float32
            "label":        int(row["emotion_int"]),  # 0–4
            "emotion":      row["emotion"],           # "stress" | "calm" | ...
            "actor_id":     row["actor_id"],          # int
            "video_quality": quality,                 # "good"|"degraded"|"poor"
            "path":         path,
        }


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def cremad_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function that stacks tensors and keeps metadata as lists.

    Usage:
        DataLoader(dataset, batch_size=8, collate_fn=cremad_collate_fn)
    """
    return {
        "frames":        torch.stack([b["frames"] for b in batch]),   # (B, T, 3, 224, 224)
        "label":         torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "actor_id":      [b["actor_id"] for b in batch],
        "emotion":       [b["emotion"] for b in batch],
        "video_quality": [b["video_quality"] for b in batch],
        "path":          [b["path"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Quick smoke test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    MANIFEST = "data/CREMA-D/manifest.csv"

    print("=== Loading manifest ===")
    rows = load_manifest(MANIFEST)
    print(f"Total rows: {len(rows)}")

    print("\n=== Generating 5-fold GroupKFold splits ===")
    folds = get_folds(MANIFEST, n_splits=5)
    for i, (train, val) in enumerate(folds):
        train_actors = set(r["actor_id"] for r in train)
        val_actors   = set(r["actor_id"] for r in val)
        overlap      = train_actors & val_actors
        print(f"  Fold {i+1}: train={len(train)} clips / {len(train_actors)} actors | "
              f"val={len(val)} clips / {len(val_actors)} actors | "
              f"actor overlap={len(overlap)} (must be 0)")
        assert len(overlap) == 0, f"Actor overlap detected in fold {i+1}!"

    print("\n=== Testing CREMADDataset (fold 0 val, 3 clips) ===")
    _, val_rows = folds[0]
    dataset = CREMADDataset(val_rows[:3], augment=False)
    for i in range(len(dataset)):
        item = dataset[i]
        print(f"  Clip {i}: frames={tuple(item['frames'].shape)} "
              f"label={item['label']} ({item['emotion']}) "
              f"actor={item['actor_id']} quality={item['video_quality']}")
    print("\n✓ All checks passed")
