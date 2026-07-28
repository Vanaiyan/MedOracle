"""
Model Soup — average weights of all 32 LOSO fold checkpoints into one model.

Reference: Wortsman et al. (2022) "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference time."

Run after downloading all 32 LOSO fold checkpoints from Kaggle:
    python -m member1_physiological.model_soup \
        --checkpoint_dir member1_physiological/checkpoints/loso \
        --output        member1_physiological/checkpoints/model_soup_best.pt

Author: Suhira Balarajan (214206G)
"""

import argparse
import os
import sys

import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.models.physiological_net import PhysiologicalNet


def load_fold_checkpoints(checkpoint_dir: str):
    """Load all .pt files from a directory. Returns list of (filename, ckpt_dict)."""
    pt_files = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
    ])
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")

    checkpoints = []
    print(f"Found {len(pt_files)} checkpoint(s)\n")
    print(f"{'File':<40} {'Subject':<8} {'Macro-F1':>10}")
    print("-" * 62)

    for fname in pt_files:
        fpath = os.path.join(checkpoint_dir, fname)
        ckpt  = torch.load(fpath, map_location="cpu", weights_only=False)
        subj  = ckpt.get("subject_id", "?")
        f1    = ckpt.get("fold_metrics", {}).get("macro_f1", -1.0)
        print(f"{fname:<40} {subj:<8} {f1:>10.4f}")
        checkpoints.append((fname, ckpt))

    print("-" * 62)
    f1_scores = [c.get("fold_metrics", {}).get("macro_f1", 0) for _, c in checkpoints]
    print(f"Mean Macro-F1 across folds: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
    return checkpoints


def average_weights(checkpoints, weighted=True):
    """
    Average model weights across all fold checkpoints.

    weighted=True  → weight each fold by its Macro-F1 score (better folds contribute more)
    weighted=False → uniform average (all folds contribute equally)
    """
    n = len(checkpoints)

    # Compute per-fold weights
    f1_scores = np.array([
        c.get("fold_metrics", {}).get("macro_f1", 1.0) for _, c in checkpoints
    ], dtype=np.float32)

    if weighted and f1_scores.sum() > 0:
        fold_weights = f1_scores / f1_scores.sum()   # normalise so they sum to 1
        method = "F1-weighted"
    else:
        fold_weights = np.ones(n, dtype=np.float32) / n
        method = "uniform"

    print(f"Averaging weights of {n} models ({method})...")
    for i, ((fname, _), w) in enumerate(zip(checkpoints, fold_weights)):
        print(f"  {fname:<40} weight: {w:.4f}")

    # Weighted sum
    avg_state = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in checkpoints[0][1]["model_state_dict"].items()
    }
    for (_, ckpt), w in zip(checkpoints, fold_weights):
        for k, v in ckpt["model_state_dict"].items():
            avg_state[k] += v.float() * w

    return avg_state, method


def average_preprocessor_stats(checkpoints, weighted=True):
    """Average EEG and GSR preprocessor stats, optionally weighted by F1."""
    f1_scores = np.array([
        c.get("fold_metrics", {}).get("macro_f1", 1.0) for _, c in checkpoints
    ], dtype=np.float32)
    weights = f1_scores / f1_scores.sum() if weighted else np.ones(len(checkpoints)) / len(checkpoints)

    eeg_means = np.stack([c["eeg_mean"] for _, c in checkpoints], axis=0)
    eeg_stds  = np.stack([c["eeg_std"]  for _, c in checkpoints], axis=0)
    gsr_means = np.array([float(c["gsr_mean"]) for _, c in checkpoints])
    gsr_stds  = np.array([float(c["gsr_std"])  for _, c in checkpoints])

    return (
        (eeg_means * weights[:, None]).sum(axis=0).astype(np.float32),
        (eeg_stds  * weights[:, None]).sum(axis=0).astype(np.float32),
        float((gsr_means * weights).sum()),
        float((gsr_stds  * weights).sum()),
    )


def verify_model_soup(avg_state: dict, config: dict) -> None:
    """Load averaged weights into PhysiologicalNet and run a quick forward pass."""
    model = PhysiologicalNet(**config)
    model.load_state_dict(avg_state)
    model.eval()

    fake_eeg = torch.randn(2, 32, 512)
    fake_gsr = torch.randn(2, 512)
    with torch.no_grad():
        logits = model(fake_eeg, fake_gsr)

    assert logits.shape == (2, 5), f"Wrong output shape: {logits.shape}"
    print("Model soup forward pass: OK")
    print(f"Output shape: {logits.shape}  (batch=2, classes=5)")


def main():
    parser = argparse.ArgumentParser(description="Model Soup — average LOSO fold weights")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing all LOSO fold .pt files")
    parser.add_argument("--output", default="member1_physiological/checkpoints/model_soup_best.pt",
                        help="Output path for the averaged model")
    args = parser.parse_args()

    print("=" * 62)
    print("Model Soup — Wortsman et al. (2022)")
    print("=" * 62 + "\n")

    # 1. Load all checkpoints
    checkpoints = load_fold_checkpoints(args.checkpoint_dir)

    # 2. Weighted average weights (better folds contribute more)
    avg_state, method = average_weights(checkpoints, weighted=True)

    # 3. Weighted average preprocessor stats
    eeg_mean, eeg_std, gsr_mean, gsr_std = average_preprocessor_stats(checkpoints, weighted=True)

    # Use config from first checkpoint
    cfg = checkpoints[0][1]["config"]
    model_cfg = {
        "d_model":  cfg["d_model"],
        "n_heads":  cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "dropout":  cfg["dropout"],
    }

    # 4. Verify
    print()
    verify_model_soup(avg_state, model_cfg)

    # 5. Compute expected F1 improvement
    f1_scores = [c.get("fold_metrics", {}).get("macro_f1", 0) for _, c in checkpoints]
    mean_f1   = float(np.mean(f1_scores))

    # 6. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    soup_ckpt = {
        "config":           model_cfg,
        "model_state_dict": avg_state,
        "eeg_mean":         eeg_mean,
        "eeg_std":          eeg_std,
        "gsr_mean":         gsr_mean,
        "gsr_std":          gsr_std,
        "subject_id":       "model_soup_32folds",
        "fold_metrics":     {"macro_f1": mean_f1},
        "n_folds_averaged": len(checkpoints),
        "method":           f"{method}_weight_averaging",
        "reference":        "Wortsman et al. (2022) Model Soups",
    }
    torch.save(soup_ckpt, args.output)

    print(f"\nSaved model soup → {args.output}")
    print(f"Folds averaged  : {len(checkpoints)}")
    print(f"Mean LOSO F1    : {mean_f1:.4f}")
    print("\nTest it with:")
    print(f"  python -m member1_physiological.predict \\")
    print(f"      --checkpoint {args.output} \\")
    print(f"      --dat_file data/DEAP/s01.dat --trial 0 --window 0")


if __name__ == "__main__":
    main()
