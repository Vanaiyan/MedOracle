"""
One-time script: re-saves random_split_best.pt with fitted preprocessor stats.

The checkpoint was saved as a raw state_dict (no config/eeg_mean/gsr_mean).
This script loads all DEAP data, fits the preprocessors, then re-saves the
checkpoint in the structured format that predict.py expects.

Run once from the MedOracle root:
    python -m member1_physiological.fix_checkpoint
"""

import os
import sys
import pickle

import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor

CHECKPOINT_IN  = "member1_physiological/checkpoints/random_split_best.pt"
CHECKPOINT_OUT = "member1_physiological/checkpoints/random_split_best.pt"
DEAP_DIR       = "data/DEAP"

_BASELINE_SAMPLES = 384   # 3s × 128 Hz
_WINDOW_SAMPLES   = 512   # 4s × 128 Hz
_WINDOWS_PER_TRIAL = 15


def load_all_windows(deap_dir: str):
    """Load EEG and GSR windows from all available DEAP .dat files."""
    all_eeg, all_gsr = [], []

    dat_files = sorted([
        f for f in os.listdir(deap_dir) if f.endswith(".dat")
    ])
    print(f"Found {len(dat_files)} DEAP .dat files")

    for fname in dat_files:
        fpath = os.path.join(deap_dir, fname)
        with open(fpath, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        data = raw["data"]  # (40, 40, 8064)

        for trial_idx in range(40):
            trial_data = data[trial_idx, :, _BASELINE_SAMPLES:]  # (40, 7680)
            for w in range(_WINDOWS_PER_TRIAL):
                start = w * _WINDOW_SAMPLES
                end   = start + _WINDOW_SAMPLES
                eeg = trial_data[:32, start:end].astype(np.float32)  # (32, 512)
                gsr = trial_data[36,  start:end].astype(np.float32)  # (512,)
                all_eeg.append(eeg)
                all_gsr.append(gsr)

        print(f"  Loaded {fname}")

    eeg_arr = np.stack(all_eeg, axis=0)  # (N, 32, 512)
    gsr_arr = np.stack(all_gsr, axis=0)  # (N, 512)
    print(f"\nTotal windows: {len(all_eeg)}")
    return eeg_arr, gsr_arr


def main():
    print("=== Fixing checkpoint: adding preprocessor stats ===\n")

    # 1. Load raw checkpoint
    print(f"Loading: {CHECKPOINT_IN}")
    ckpt_raw = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)

    if "config" in ckpt_raw:
        print("Checkpoint already in structured format — nothing to fix.")
        return

    print("Detected raw state_dict format. Fixing...\n")

    # 2. Fit preprocessors on all DEAP data
    print("Loading DEAP data to fit preprocessors...")
    eeg_arr, gsr_arr = load_all_windows(DEAP_DIR)

    eeg_prep = EEGPreprocessor()
    eeg_prep.fit(eeg_arr)
    eeg_mean, eeg_std = eeg_prep.get_stats()
    print(f"\nEEG stats: mean range [{eeg_mean.min():.4f}, {eeg_mean.max():.4f}]")
    print(f"EEG stats: std  range [{eeg_std.min():.4f}, {eeg_std.max():.4f}]")

    gsr_prep = GSRPreprocessor()
    gsr_prep.fit(gsr_arr)
    gsr_mean, gsr_std = gsr_prep.get_stats()
    print(f"GSR stats: mean={gsr_mean:.4f}, std={gsr_std:.4f}")

    # 3. Build structured checkpoint
    structured = {
        "config": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.3,
        },
        "model_state_dict": ckpt_raw,
        "eeg_mean": eeg_mean,
        "eeg_std":  eeg_std,
        "gsr_mean": gsr_mean,
        "gsr_std":  gsr_std,
        "subject_id": "random_split",
        "best_epoch": -1,
        "fold_metrics": {"macro_f1": 0.4171},
    }

    # 4. Save
    torch.save(structured, CHECKPOINT_OUT)
    print(f"\nSaved structured checkpoint to: {CHECKPOINT_OUT}")
    print("Done. Run predict.py again — predictions will now vary across emotions.")


if __name__ == "__main__":
    main()
