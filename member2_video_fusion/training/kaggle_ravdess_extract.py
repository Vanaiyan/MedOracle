"""
member2_video_fusion/training/kaggle_ravdess_extract.py
========================================================
MedOracle — Member 2 (Vanaiyan Kirupagaran, 214215H)

Kaggle notebook: RAVDESS video → pre-extracted .npy frames
===========================================================

Run this notebook ONCE on Kaggle BEFORE the main retraining notebook
(kaggle_retrain_v2.py). Its output must then be committed as a Kaggle
dataset so the training notebook can access the .npy files.

What this does
--------------
1. Installs ultralytics (YOLOv8) and downloads the face-detection model.
2. Walks all .mp4 files under the RAVDESS input directory.
3. For each valid clip (emotion code not in DROP set):
     a. Decode video → sample T=16 frames uniformly.
     b. Run YOLO face detection on each frame.
     c. Crop the largest detected face → resize to 224×224.
        Fallback: if YOLO detects no face, use the full frame.
     d. Stack frames into (16, 224, 224, 3) uint8 RGB array.
     e. Save as <actor_id>_<emotion>_<clip_idx>.npy
4. Builds ravdess_manifest.csv with columns:
     npy_path, actor_id (offset +2000), source, emotion, emotion_int

Output directory: /kaggle/working/ravdess_npy/
Manifest path  : /kaggle/working/ravdess_manifest.csv

After this runs
---------------
  "Save & Run All" on Kaggle → commit output → create new Kaggle dataset
  from the output, e.g. "ravdess-npy-frames".
  Then set RAVDESS_NPY_DIR in kaggle_retrain_v2.py to point to it.

RAVDESS filename format
-----------------------
  01-01-06-01-02-01-12.mp4
  parts (split by "-"):
    [0] modality          01 = AV speech
    [1] vocal channel     01 = speech, 02 = song
    [2] emotion code      01–08
    [3] emotional intensity
    [4] statement
    [5] repetition
    [6] actor ID          01–24

Emotion mapping (emotion code at parts[2])
------------------------------------------
  "01" → calm     (neutral → calm, low-arousal neutral state)
  "02" → calm     (calm)
  "03" → happy
  "04" → sad
  "05" → angry
  "06" → stress   (fearful → stress; Kreibig, 2010)
  "07" → DROP     (disgust — no clean 5-class mapping)
  "08" → DROP     (surprised — no clean 5-class mapping)

Actor ID convention
-------------------
  Original RAVDESS IDs: 1–24
  Stored in manifest as: original_id + 2000  (→ 2001–2024)
  Avoids collision with CREMA-D IDs (1001–1091) in GroupKFold.

Dataset paths (Kaggle)
----------------------
  RAVDESS videos  : /kaggle/input/datasets/adrivg/ravdess-emotional-speech-video/
  YOLO model URL  : https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt

Author: Vanaiyan Kirupagaran (214215H)
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================

import subprocess
subprocess.run(["pip", "install", "ultralytics", "--quiet"], check=True)

print("✓ ultralytics installed")


# ============================================================
# CELL 2 — Imports and configuration
# ============================================================

import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
RAVDESS_ROOT   = Path("/kaggle/input/datasets/adrivg/ravdess-emotional-speech-video")
OUTPUT_NPY_DIR = Path("/kaggle/working/ravdess_npy")
MANIFEST_PATH  = Path("/kaggle/working/ravdess_manifest.csv")
YOLO_MODEL_PATH = Path("/kaggle/working/yolo_face.pt")

OUTPUT_NPY_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_FRAMES   = 16
FRAME_SIZE = 224

# ── Emotion map (code → MedOracle class) ──────────────────────────────────────
# Locked: matches CLAUDE.md and label_harmonization.py
RAVDESS_EMOTION_MAP = {
    "01": "calm",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "stress",
}
DROP_CODES = {"07", "08"}   # disgust, surprised — no clean 5-class mapping

EMOTION_CLASSES = {
    "stress": 0,
    "calm":   1,
    "happy":  2,
    "sad":    3,
    "angry":  4,
}

RAVDESS_ACTOR_OFFSET = 2000   # stored actor_id = original (1–24) + 2000

print("Configuration:")
print(f"  RAVDESS root  : {RAVDESS_ROOT}")
print(f"  Output .npy   : {OUTPUT_NPY_DIR}")
print(f"  Manifest      : {MANIFEST_PATH}")
print(f"  N_FRAMES      : {N_FRAMES}")
print(f"  FRAME_SIZE    : {FRAME_SIZE}×{FRAME_SIZE}")
print(f"  Actor offset  : +{RAVDESS_ACTOR_OFFSET}")
print(f"  Emotion map   : {RAVDESS_EMOTION_MAP}")
print(f"  DROP codes    : {DROP_CODES}")


# ============================================================
# CELL 3 — Download YOLO face-detection model
# ============================================================

YOLO_URL = ("https://huggingface.co/arnabdhar/YOLOv8-Face-Detection"
            "/resolve/main/model.pt")

if not YOLO_MODEL_PATH.exists():
    print(f"Downloading YOLO face model from HuggingFace...")
    urllib.request.urlretrieve(YOLO_URL, YOLO_MODEL_PATH)
    print(f"✓ Saved to {YOLO_MODEL_PATH} ({YOLO_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"✓ YOLO model already exists: {YOLO_MODEL_PATH}")

yolo_model = YOLO(str(YOLO_MODEL_PATH))
yolo_model.fuse()   # fuse Conv+BN layers for faster inference
print("✓ YOLO model loaded and fused")


# ============================================================
# CELL 4 — Frame sampling and face-crop helpers
# ============================================================

def sample_frames_uniform(video_path: Path, n_frames: int = N_FRAMES):
    """Sample n_frames uniformly from a video file.

    Parameters
    ----------
    video_path : path to .mp4 file
    n_frames   : number of frames to sample (T=16)

    Returns
    -------
    np.ndarray of shape (n_frames, H, W, 3) uint8 RGB, or None if the
    video cannot be opened or has fewer frames than requested.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.stack(frames)   # (T, H, W, 3) uint8 RGB


def detect_and_crop_face(
    frame: np.ndarray,
    yolo: YOLO,
    target_size: int = FRAME_SIZE,
) -> np.ndarray:
    """Detect the largest face in a frame and return a square crop at target_size.

    If YOLO detects no face, returns the full frame resized to target_size.
    This graceful fallback ensures no clip is silently dropped due to detection
    failure on individual frames.

    Parameters
    ----------
    frame       : (H, W, 3) uint8 RGB
    yolo        : loaded YOLO face-detection model
    target_size : output crop size (default 224)

    Returns
    -------
    (target_size, target_size, 3) uint8 RGB
    """
    results = yolo.predict(frame, verbose=False, conf=0.3)

    best_box  = None
    best_area = 0

    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        # Add a small margin (10%) around the detected face
        H, W = frame.shape[:2]
        margin_x = int((x2 - x1) * 0.10)
        margin_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(W, x2 + margin_x)
        y2 = min(H, y2 + margin_y)
        crop = frame[y1:y2, x1:x2]
    else:
        # Fallback: use the full frame
        crop = frame

    # Resize to target_size × target_size
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def process_clip(
    video_path: Path,
    yolo: YOLO,
    n_frames:   int = N_FRAMES,
    frame_size: int = FRAME_SIZE,
) -> tuple[np.ndarray | None, int, int]:
    """Full pipeline for one clip: decode → YOLO crop → stack.

    Parameters
    ----------
    video_path : path to .mp4 file
    yolo       : loaded YOLO model
    n_frames   : T (default 16)
    frame_size : spatial size (default 224)

    Returns
    -------
    Tuple of:
        frames        : (16, 224, 224, 3) uint8 RGB, or None on failure
        faces_detected : number of frames where YOLO found a face
        total_frames   : n_frames (for computing detection rate)
    """
    raw_frames = sample_frames_uniform(video_path, n_frames)
    if raw_frames is None:
        return None, 0, n_frames

    processed    = []
    faces_found  = 0

    for frame in raw_frames:
        # Run YOLO — check if face was actually detected (not fallback)
        results = yolo.predict(frame, verbose=False, conf=0.3)
        has_face = (results and results[0].boxes is not None
                    and len(results[0].boxes) > 0)
        if has_face:
            faces_found += 1

        # Crop (with fallback to full frame if no detection)
        cropped = detect_and_crop_face(frame, yolo, frame_size)
        processed.append(cropped)

    return np.stack(processed), faces_found, n_frames


# ============================================================
# CELL 5 — Parse RAVDESS filename → (actor_id, emotion_code)
# ============================================================

def parse_ravdess_filename(stem: str) -> tuple[int | None, str | None]:
    """Extract actor ID and emotion code from a RAVDESS filename stem.

    Parameters
    ----------
    stem : filename without extension, e.g. "01-01-06-01-02-01-12"

    Returns
    -------
    (actor_id, emotion_code) or (None, None) if parsing fails.
    """
    parts = stem.split("-")
    if len(parts) != 7:
        return None, None
    try:
        actor_id     = int(parts[6])   # 01–24
        emotion_code = parts[2]        # "01"–"08"
        return actor_id, emotion_code
    except (ValueError, IndexError):
        return None, None


# Quick validation
_test_cases = [
    ("01-01-06-01-02-01-12", 12, "06"),
    ("01-01-01-01-01-01-01",  1, "01"),
    ("01-01-08-02-01-02-24", 24, "08"),
]
for stem, exp_actor, exp_code in _test_cases:
    actor, code = parse_ravdess_filename(stem)
    assert actor == exp_actor and code == exp_code, \
        f"Parsing failed for {stem}: got ({actor}, {code})"
print("✓ Filename parsing validated")


# ============================================================
# CELL 6 — Discover all RAVDESS .mp4 files
# ============================================================

all_mp4_files = sorted(RAVDESS_ROOT.rglob("*.mp4"))
print(f"Found {len(all_mp4_files)} .mp4 files under {RAVDESS_ROOT}")

# Categorise: valid, drop, unrecognised
valid_files   = []
drop_count    = 0
unknown_count = 0

for mp4 in all_mp4_files:
    actor_id, emotion_code = parse_ravdess_filename(mp4.stem)
    if actor_id is None:
        unknown_count += 1
        continue
    if emotion_code in DROP_CODES:
        drop_count += 1
        continue
    if emotion_code not in RAVDESS_EMOTION_MAP:
        unknown_count += 1
        continue
    valid_files.append((mp4, actor_id, emotion_code))

print(f"\nCategorisation:")
print(f"  Valid (to extract) : {len(valid_files)}")
print(f"  Dropped (DIS/SUR)  : {drop_count}")
print(f"  Unrecognised       : {unknown_count}")

# Emotion distribution
from collections import Counter
emo_counts = Counter(RAVDESS_EMOTION_MAP[code] for _, _, code in valid_files)
print(f"\nEmotion distribution:")
for emo, cnt in sorted(emo_counts.items()):
    print(f"  {emo:8s} : {cnt}")


# ============================================================
# CELL 7 — Main extraction loop
# ============================================================

manifest_rows  = []
n_success      = 0
n_failed       = 0
n_low_detection = 0   # clips where face detected in <50% frames

t_start = time.time()
print(f"\nExtracting {len(valid_files)} clips...")
print(f"{'Clip':>5}  {'Actor':>5}  {'Emotion':>8}  {'FaceRate':>9}  {'Status'}")
print("-" * 55)

for clip_idx, (mp4_path, actor_id, emotion_code) in enumerate(valid_files):
    emotion_str = RAVDESS_EMOTION_MAP[emotion_code]
    emotion_int = EMOTION_CLASSES[emotion_str]

    # Stored actor_id (offset)
    stored_actor_id = actor_id + RAVDESS_ACTOR_OFFSET   # 2001–2024

    # Output .npy filename: actor{id:04d}_emo{code}_{clipidx:05d}.npy
    npy_name = f"actor{stored_actor_id:04d}_{emotion_str}_{clip_idx:05d}.npy"
    npy_path = OUTPUT_NPY_DIR / npy_name

    # Skip if already extracted (allows resuming interrupted runs)
    if npy_path.exists():
        # Read manifest entry from existing file
        manifest_rows.append({
            "npy_path":    str(npy_path),
            "actor_id":    stored_actor_id,
            "source":      "ravdess",
            "emotion":     emotion_str,
            "emotion_int": emotion_int,
        })
        n_success += 1
        if (clip_idx + 1) % 100 == 0:
            print(f"  {clip_idx+1:>5} / {len(valid_files)} — {n_success} done "
                  f"({time.time()-t_start:.0f}s elapsed, skipped existing)")
        continue

    # Extract frames
    frames, faces_found, total_frames = process_clip(
        mp4_path, yolo_model
    )

    if frames is None:
        status = "FAILED (decode)"
        n_failed += 1
        if (clip_idx + 1) % 50 == 0 or clip_idx < 5:
            print(f"  {clip_idx+1:>5}  {actor_id:>5}  {emotion_str:>8}  "
                  f"{'N/A':>9}  {status}")
        continue

    face_rate = faces_found / total_frames
    if face_rate < 0.5:
        n_low_detection += 1
        status = f"low_face ({face_rate:.0%})"
    else:
        status = f"ok ({face_rate:.0%})"

    # Save .npy
    np.save(str(npy_path), frames)   # (16, 224, 224, 3) uint8

    manifest_rows.append({
        "npy_path":    str(npy_path),
        "actor_id":    stored_actor_id,
        "source":      "ravdess",
        "emotion":     emotion_str,
        "emotion_int": emotion_int,
    })
    n_success += 1

    # Progress every 100 clips
    if (clip_idx + 1) % 100 == 0:
        elapsed = time.time() - t_start
        eta     = elapsed / (clip_idx + 1) * (len(valid_files) - clip_idx - 1)
        print(f"  {clip_idx+1:>5} / {len(valid_files)} — "
              f"{n_success} ok, {n_failed} failed, {n_low_detection} low-face | "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

print(f"\n{'─'*55}")
print(f"Extraction complete: {n_success} ok / {n_failed} failed / "
      f"{n_low_detection} low face-detection rate")
print(f"Total time: {time.time()-t_start:.1f}s")


# ============================================================
# CELL 8 — Save manifest CSV
# ============================================================

fieldnames = ["npy_path", "actor_id", "source", "emotion", "emotion_int"]

with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Manifest saved: {MANIFEST_PATH}")
print(f"  Rows written  : {len(manifest_rows)}")

# Verify counts
from collections import Counter
src_counts = Counter(r["source"] for r in manifest_rows)
emo_counts_final = Counter(r["emotion"] for r in manifest_rows)
actor_counts = Counter(r["actor_id"] for r in manifest_rows)

print(f"\n  Sources       : {dict(src_counts)}")
print(f"  Emotions      :")
for emo in ["stress", "calm", "happy", "sad", "angry"]:
    print(f"    {emo:8s}: {emo_counts_final.get(emo, 0)}")
print(f"  Unique actors : {len(actor_counts)} (expected 24, IDs 2001–2024)")
print(f"  Actor ID range: {min(actor_counts.keys())}–{max(actor_counts.keys())}")


# ============================================================
# CELL 9 — Sanity check: load a few .npy files back
# ============================================================

print("\nSanity check: loading first 5 extracted .npy files...")
npy_files = sorted(OUTPUT_NPY_DIR.glob("*.npy"))[:5]

all_ok = True
for npy_file in npy_files:
    arr = np.load(str(npy_file))
    shape_ok = arr.shape == (N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3)
    dtype_ok  = arr.dtype == np.uint8
    range_ok  = arr.min() >= 0 and arr.max() <= 255

    status = "✓" if (shape_ok and dtype_ok and range_ok) else "✗"
    if not (shape_ok and dtype_ok and range_ok):
        all_ok = False
    print(f"  {status} {npy_file.name}: shape={arr.shape} dtype={arr.dtype} "
          f"range=[{arr.min()},{arr.max()}]")

print(f"\n{'✓ All checks passed' if all_ok else '✗ Some files failed — review output above'}")

print(f"\n{'='*55}")
print(f"  RAVDESS extraction COMPLETE")
print(f"  .npy files : {OUTPUT_NPY_DIR}")
print(f"  Manifest   : {MANIFEST_PATH}")
print(f"{'='*55}")
print(f"\nNext steps:")
print(f"  1. 'Save & Run All' on Kaggle to commit this notebook output.")
print(f"  2. From the output tab, create a new Kaggle dataset,")
print(f"     e.g. 'vanaiyan/ravdess-npy-frames'.")
print(f"  3. Set RAVDESS_NPY_DIR and RAVDESS_MANIFEST_PATH in")
print(f"     kaggle_retrain_v2.py to point to the new dataset.")
