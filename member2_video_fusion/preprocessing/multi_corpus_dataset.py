"""
member2_video_fusion/preprocessing/multi_corpus_dataset.py
===========================================================
MedOracle — Member 2 (Vanaiyan Kirupagaran, 214215H)

MultiCorpusDataset: unified PyTorch Dataset for CREMA-D + RAVDESS,
loading from pre-extracted .npy frame files.

Why .npy instead of decoding video at runtime
----------------------------------------------
Both datasets store each clip as a (16, 224, 224, 3) uint8 numpy array
on disk. Compared to video-decode + YOLO detection at training time,
.npy loading is ~10× faster, removes the GPU/CPU contention from YOLO,
and makes training speed predictable.

Manifest format
---------------
A unified CSV with columns:
    npy_path    : absolute path to the .npy file
    actor_id    : int — actor identifier (namespace below)
    source      : "cremad" | "ravdess"
    emotion     : str — one of stress|calm|happy|sad|angry
    emotion_int : int — 0–4 from EMOTION_CLASSES

Actor ID namespacing (avoids GroupKFold collision)
---------------------------------------------------
    CREMA-D actors  : 1001–1091  (original 4-digit IDs)
    RAVDESS actors  : 2001–2024  (original 1–24 offset by +2000)
    No overlap guaranteed.

Augmentation (training only, applied identically to all T=16 frames)
---------------------------------------------------------------------
    1. Random horizontal flip          (p=0.5)
    2. Color jitter: brightness ±30%, contrast ±30%, saturation ±30%
       (drawn once per clip, applied to every frame via HSV transform)
    3. Random rotation ±10°
       (BORDER_REFLECT fill avoids black corner artefacts)
    4. Random erasing / cutout         (p=0.5, 2–20% of frame area)
       Occludes a random patch (same box across all frames) to break
       actor-identity shortcuts and force the model onto expression cues.
       Reference: Zhong et al. (2020), "Random Erasing Data Augmentation".

Key: __getitem__ returns dict with key "clip" (not "frames") to match
     the training loop convention used throughout train.py and
     colab_block13_training.py.

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

# ── Shared emotion map (locked across all members) ────────────────────────────
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.data_contracts import EMOTION_CLASSES

# ── Constants ─────────────────────────────────────────────────────────────────

N_FRAMES      = 16
FRAME_SIZE    = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RAVDESS_ACTOR_OFFSET = 2000   # RAVDESS actor 1 → stored as 2001

# Reverse map for convenience
IDX_TO_EMOTION = {v: k for k, v in EMOTION_CLASSES.items()}

# Label harmonisation maps (mirrors shared/label_harmonization.py)
CREMAD_LABEL_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "calm",
    "FEA": "stress",
    # DIS → DROP (excluded at manifest build time)
}

RAVDESS_EMOTION_MAP = {
    "01": "calm",    # neutral → calm
    "02": "calm",    # calm
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "stress",  # fearful → stress (Kreibig, 2010)
    # "07": DROP (disgust)
    # "08": DROP (surprised)
}


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _flip_frames(frames: np.ndarray) -> np.ndarray:
    """Random horizontal flip — same direction applied to all T frames.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB

    Returns
    -------
    (T, H, W, 3) uint8 — flipped with probability 0.5
    """
    if random.random() < 0.5:
        return frames[:, :, ::-1, :].copy()
    return frames


def _color_jitter_frames(
    frames:     np.ndarray,
    brightness: float = 0.2,
    contrast:   float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    """Apply brightness, contrast, and saturation jitter uniformly to a clip.

    Jitter factors are sampled once and applied identically to all frames,
    preserving temporal consistency (no flickering artefacts between frames).

    Strategy
    --------
    1. Brightness + contrast: linear rescale in float32 → clamp → back to uint8.
       alpha (contrast scale): uniform in [1−contrast, 1+contrast]
       beta  (brightness bias): uniform in [−brightness×255, +brightness×255]
    2. Saturation: convert to HSV, scale the S channel by sat_factor,
       convert back to RGB.
       sat_factor: uniform in [1−saturation, 1+saturation]

    Parameters
    ----------
    frames     : (T, H, W, 3) uint8 RGB
    brightness : fractional brightness shift range (default 0.2 → ±20%)
    contrast   : contrast scale range (default 0.2 → ×0.8–×1.2)
    saturation : HSV S-channel scale range (default 0.2 → ×0.8–×1.2)

    Returns
    -------
    (T, H, W, 3) uint8 RGB
    """
    # Draw factors once per clip
    alpha   = random.uniform(1.0 - contrast,      1.0 + contrast)
    beta    = random.uniform(-brightness * 255.0, brightness * 255.0)
    sat_fac = random.uniform(1.0 - saturation,    1.0 + saturation)

    result = []
    for frame in frames:
        # 1. Brightness + contrast (in float32 to avoid uint8 wrap-around)
        f = np.clip(alpha * frame.astype(np.float32) + beta, 0.0, 255.0).astype(np.uint8)

        # 2. Saturation via HSV colour space
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_fac, 0.0, 255.0)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        result.append(f)

    return np.stack(result)   # (T, H, W, 3) uint8


def _rotate_frames(frames: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Rotate all frames by the same random angle in [−max_angle, +max_angle].

    Uses BORDER_REFLECT (mirror padding) to fill corner areas created by
    the rotation, avoiding the black-corner artefact that BORDER_CONSTANT
    would introduce and that the network could learn to exploit.

    Reference: common practice in affine augmentation for face recognition.

    Parameters
    ----------
    frames    : (T, H, W, 3) uint8 RGB
    max_angle : rotation limit in degrees (default ±10°)

    Returns
    -------
    (T, H, W, 3) uint8 — rotated by the same angle
    """
    angle = random.uniform(-max_angle, max_angle)
    _, H, W, _ = frames.shape
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = np.stack([
        cv2.warpAffine(
            f, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        for f in frames
    ])
    return rotated


def _random_erase_frames(
    frames:       np.ndarray,
    p:            float = 0.5,
    area_range:   tuple = (0.02, 0.20),
    aspect_range: tuple = (0.3, 3.3),
) -> np.ndarray:
    """Random erasing / cutout — occlude one random rectangle in every frame.

    The same rectangle (location + size) is erased from all T frames to keep
    temporal consistency, and is filled with random RGB noise. This forces the
    model to rely on distributed expression cues rather than memorising a fixed
    region of a particular actor's face.

    Reference: Zhong et al. (2020), "Random Erasing Data Augmentation".

    Parameters
    ----------
    frames       : (T, H, W, 3) uint8 RGB
    p            : probability of applying erasing (default 0.5)
    area_range   : erased area as a fraction of the frame (default 2–20%)
    aspect_range : aspect ratio range of the erased box (default 0.3–3.3)

    Returns
    -------
    (T, H, W, 3) uint8 — with one patch erased (or unchanged with prob 1−p)
    """
    if random.random() > p:
        return frames

    _, H, W, _ = frames.shape
    img_area = H * W

    for _ in range(10):   # retry until a valid box fits inside the frame
        target_area = random.uniform(*area_range) * img_area
        aspect      = random.uniform(*aspect_range)
        h = int(round((target_area * aspect) ** 0.5))
        w = int(round((target_area / aspect) ** 0.5))
        if 0 < h < H and 0 < w < W:
            top  = random.randint(0, H - h)
            left = random.randint(0, W - w)
            frames = frames.copy()
            noise  = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
            frames[:, top:top + h, left:left + w, :] = noise   # same box, all frames
            return frames

    return frames   # no valid box found in 10 tries — return unchanged


def augment_clip(frames: np.ndarray) -> np.ndarray:
    """Apply full training augmentation pipeline to a single clip.

    Augmentation order (all operations applied identically to every frame):
        1. Random horizontal flip   (p=0.5)
        2. Color jitter             (brightness, contrast, saturation ±30%)
        3. Random rotation          (±10°, BORDER_REFLECT fill)
        4. Random erasing / cutout  (p=0.5, 2–20% of frame area)

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB — already resized to 224×224

    Returns
    -------
    (T, H, W, 3) uint8 — augmented clip
    """
    frames = _flip_frames(frames)
    frames = _color_jitter_frames(frames, brightness=0.3, contrast=0.3, saturation=0.3)
    frames = _rotate_frames(frames)
    frames = _random_erase_frames(frames)
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# Normalisation + tensor conversion
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_and_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 → (T, 3, H, W) float32 normalised tensor.

    Applies ImageNet mean/std normalisation, then transposes to channel-first
    layout expected by PyTorch's Conv2d layers.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB in [0, 255]

    Returns
    -------
    torch.Tensor of shape (T, 3, H, W) float32, ImageNet-normalised
    """
    frames_f = frames.astype(np.float32) / 255.0           # [0, 1]
    frames_f = (frames_f - IMAGENET_MEAN) / IMAGENET_STD   # ImageNet normalised
    tensor   = torch.from_numpy(
        frames_f.transpose(0, 3, 1, 2)                     # (T, H, W, 3) → (T, 3, H, W)
    )
    return tensor.contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_unified_manifest(manifest_path: str | Path) -> List[dict]:
    """Load the unified multi-corpus manifest CSV.

    Required columns: npy_path, actor_id, source, emotion, emotion_int

    Parameters
    ----------
    manifest_path : path to the unified manifest CSV

    Returns
    -------
    List of dicts, one per clip.
    """
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["actor_id"]    = int(row["actor_id"])
            row["emotion_int"] = int(row["emotion_int"])
            rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Actor-independent GroupKFold splits
# ═══════════════════════════════════════════════════════════════════════════════

def get_unified_folds(
    manifest_path: str | Path,
    n_splits: int = 5,
) -> List[Tuple[List[dict], List[dict]]]:
    """Generate actor-independent GroupKFold splits for the unified dataset.

    Grouping is by actor_id. Because CREMA-D actors are 1001–1091 and
    RAVDESS actors are 2001–2024, there is no numeric overlap, so GroupKFold
    will never place the same actor in both train and val.

    This means:
    - No actor identity leakage between splits.
    - Both CREMA-D and RAVDESS actors are distributed across folds.
    - The model is evaluated on actors it has never seen during training.

    Parameters
    ----------
    manifest_path : path to unified manifest CSV
    n_splits      : number of folds (default 5)

    Returns
    -------
    List of (train_rows, val_rows) tuples — one per fold.
    """
    rows   = load_unified_manifest(manifest_path)
    groups = np.array([r["actor_id"] for r in rows])
    X_idx  = np.arange(len(rows))

    gkf   = GroupKFold(n_splits=n_splits)
    folds = []

    for train_idx, val_idx in gkf.split(X_idx, groups=groups):
        train_rows = [rows[i] for i in train_idx]
        val_rows   = [rows[i] for i in val_idx]
        folds.append((train_rows, val_rows))

    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset class
# ═══════════════════════════════════════════════════════════════════════════════

class MultiCorpusDataset(Dataset):
    """Unified PyTorch Dataset for CREMA-D + RAVDESS pre-extracted .npy files.

    Each .npy file is expected to be shape (16, 224, 224, 3) uint8 RGB,
    produced by the RAVDESS extraction notebook (kaggle_ravdess_extract.py)
    for RAVDESS, or the existing pre-extraction pipeline for CREMA-D.

    Parameters
    ----------
    rows        : list of manifest dicts — from load_unified_manifest()
                  or the train/val split from get_unified_folds()
    augment     : if True, apply training augmentations (flip, jitter, rotation)
                  Set True for training splits, False for validation/test.
    skip_errors : if True, missing or corrupt .npy files are silently replaced
                  with a black-frame placeholder so training never crashes.
                  If False, an IOError is raised on any bad file.

    __getitem__ returns a dict
    --------------------------
    "clip"      : (T, 3, 224, 224) float32 tensor    — model input
    "label"     : int                                 — emotion class 0–4
    "emotion"   : str                                 — e.g. "stress"
    "actor_id"  : int                                 — actor identifier
    "source"    : str                                 — "cremad" | "ravdess"
    "npy_path"  : str                                 — path to source file
    """

    def __init__(
        self,
        rows:        List[dict],
        augment:     bool = False,
        skip_errors: bool = True,
    ):
        self.augment     = augment
        self.skip_errors = skip_errors

        # Filter to rows whose .npy files actually exist
        self.rows = [r for r in rows if Path(r["npy_path"]).exists()]
        n_missing = len(rows) - len(self.rows)
        if n_missing > 0:
            print(f"[MultiCorpusDataset] Warning: {n_missing}/{len(rows)} "
                  f".npy files not found on disk — skipped")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # ── 1. Load pre-extracted frames ──────────────────────────────────────
        try:
            frames = np.load(row["npy_path"])   # expected: (16, 224, 224, 3) uint8
        except Exception as exc:
            if self.skip_errors:
                frames = np.zeros(
                    (N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8
                )
            else:
                raise IOError(f"Cannot load .npy: {row['npy_path']}") from exc

        # Guard: ensure correct dtype (in case of legacy float .npy)
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        # Guard: ensure correct shape (T, H, W, 3)
        if frames.shape != (N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3):
            if self.skip_errors:
                frames = np.zeros(
                    (N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8
                )
            else:
                raise ValueError(
                    f"Unexpected shape {frames.shape} in {row['npy_path']}"
                )

        # ── 2. Training augmentation ──────────────────────────────────────────
        if self.augment:
            frames = augment_clip(frames)

        # ── 3. Normalise + to tensor ─────────────────────────────────────────
        clip_tensor = normalise_and_to_tensor(frames)   # (T, 3, 224, 224) float32

        return {
            "clip":     clip_tensor,
            "label":    int(row["emotion_int"]),
            "emotion":  row["emotion"],
            "actor_id": int(row["actor_id"]),
            "source":   row["source"],
            "npy_path": row["npy_path"],
        }


def multicorpus_collate_fn(batch: List[dict]) -> dict:
    """Custom collate: stack tensors, keep metadata as lists.

    Usage
    -----
        DataLoader(dataset, batch_size=16, collate_fn=multicorpus_collate_fn)
    """
    return {
        "clip":     torch.stack([b["clip"] for b in batch]),          # (B, T, 3, 224, 224)
        "label":    torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "actor_id": [b["actor_id"] for b in batch],
        "emotion":  [b["emotion"] for b in batch],
        "source":   [b["source"] for b in batch],
        "npy_path": [b["npy_path"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    MANIFEST = "data/unified_manifest.csv"
    print("=== MultiCorpusDataset smoke test ===\n")

    # ── 1. Fold validation ────────────────────────────────────────────────────
    print("Generating 5-fold splits...")
    folds = get_unified_folds(MANIFEST, n_splits=5)
    for i, (train, val) in enumerate(folds):
        train_actors = set(r["actor_id"] for r in train)
        val_actors   = set(r["actor_id"] for r in val)
        overlap      = train_actors & val_actors
        train_src    = set(r["source"] for r in train)
        val_src      = set(r["source"] for r in val)
        print(f"  Fold {i+1}: train={len(train):>5} clips / {len(train_actors):>3} actors "
              f"{sorted(train_src)} | "
              f"val={len(val):>4} clips / {len(val_actors):>3} actors "
              f"{sorted(val_src)} | overlap={len(overlap)} (must be 0)")
        assert len(overlap) == 0, f"Actor overlap detected in fold {i+1}!"
    print(f"\n✓ All {len(folds)} folds: no actor overlap\n")

    # ── 2. Dataset loading ────────────────────────────────────────────────────
    print("Testing MultiCorpusDataset (first fold, val, 3 clips)...")
    _, val_rows = folds[0]
    ds = MultiCorpusDataset(val_rows[:3], augment=False)
    for k in range(len(ds)):
        item = ds[k]
        assert item["clip"].shape == (16, 3, 224, 224), f"Bad shape: {item['clip'].shape}"
        assert 0 <= item["label"] <= 4
        assert item["emotion"] in EMOTION_CLASSES
        print(f"  [{k}] clip={tuple(item['clip'].shape)} "
              f"label={item['label']} ({item['emotion']}) "
              f"actor={item['actor_id']} source={item['source']}")

    # ── 3. Augmentation sanity ────────────────────────────────────────────────
    print("\nTesting augmentation (same clip, 3 augmented versions)...")
    ds_aug = MultiCorpusDataset(val_rows[:1], augment=True)
    tensors = [ds_aug[0]["clip"] for _ in range(3)]
    are_different = not all(torch.allclose(tensors[0], t) for t in tensors[1:])
    print(f"  Augmented clips differ: {are_different} (should be True)")

    print("\n✓ MultiCorpusDataset checks passed")
