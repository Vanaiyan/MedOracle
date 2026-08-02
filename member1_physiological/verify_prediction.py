"""
Check: load a DEAP subject, show true label vs predicted label
for each window so you can verify the model is working correctly.

Usage:
    python -m member1_physiological.verify_prediction \
        --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt \
        --dat_file   data/DEAP/s01.dat \
        --n_samples  10
"""

import argparse
import os
import sys

import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.preprocessing.deap_loader import DEAPLoader
from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor
from member1_physiological.models.physiological_net import PhysiologicalNet
from member1_physiological.train import load_checkpoint, TrainConfig

EMOTION_CLASSES = ["stress", "calm", "happy", "sad", "angry"]


def run_verify(checkpoint_path: str, dat_file: str, n_samples: int, device: torch.device):
    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path, device)
    cfg  = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                          if k in TrainConfig.__dataclass_fields__})

    model = PhysiologicalNet(
        d_model  = cfg.d_model,
        n_heads  = cfg.n_heads,
        n_layers = cfg.n_layers,
        dropout  = cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Preprocessors (from checkpoint stats) ────────────────────────────────
    eeg_prep = EEGPreprocessor()
    eeg_prep._mean = ckpt["eeg_mean"]
    eeg_prep._std  = ckpt["eeg_std"]
    eeg_prep._fitted = True

    gsr_prep = GSRPreprocessor()
    gsr_prep._mean   = ckpt["gsr_mean"]
    gsr_prep._std    = ckpt["gsr_std"]
    gsr_prep._fitted = True

    # ── Load subject data ─────────────────────────────────────────────────────
    data_dir = os.path.dirname(dat_file)
    subj_id  = os.path.basename(dat_file).replace(".dat", "")

    print(f"Loading subject: {subj_id} from {dat_file}")
    loader       = DEAPLoader(data_dir=data_dir, verbose=False)
    loader.load_all()
    subject_data = loader.get_subject(subj_id)
    windows      = subject_data.windows

    print(f"Total windows available: {len(windows)}")
    print(f"Checking {min(n_samples, len(windows))} samples\n")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"{'#':<5} {'Trial':<7} {'Win':<5} {'TRUE LABEL':<12} {'PREDICTED':<12} {'CORRECT':<8} {'CONFIDENCE'}")
    print("-" * 68)

    correct = 0
    total   = min(n_samples, len(windows))

    with torch.no_grad():
        for i, w in enumerate(windows[:total]):
            # Preprocess
            eeg = eeg_prep.transform(w.eeg)   # (32, 512)
            gsr = gsr_prep.transform(w.gsr)   # (512,)

            eeg_t = torch.from_numpy(eeg).float().unsqueeze(0).to(device)  # (1, 32, 512)
            gsr_t = torch.from_numpy(gsr).float().unsqueeze(0).to(device)  # (1, 512)

            logits = model(eeg_t, gsr_t)                        # (1, 5)
            probs  = torch.softmax(logits, dim=-1)[0]           # (5,)
            pred_idx  = probs.argmax().item()
            confidence = probs[pred_idx].item()

            true_label = w.label_str
            pred_label = EMOTION_CLASSES[pred_idx]
            is_correct = true_label == pred_label

            if is_correct:
                correct += 1

            marker = "✓" if is_correct else "✗"
            print(
                f"{i+1:<5} {w.trial_idx:<7} {w.window_idx:<5} "
                f"{true_label:<12} {pred_label:<12} {marker:<8} {confidence:.3f}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 68)
    print(f"\nAccuracy on {total} samples: {correct}/{total} = {correct/total:.1%}")
    print(f"Checkpoint trained on fold: {ckpt.get('subject_id', 'unknown')} "
          f"(test subject was held out during training)")

    if ckpt.get("subject_id") == subj_id:
        print(f"\n⚠  NOTE: You are testing on {subj_id} using the checkpoint where "
              f"{subj_id} was the TEST subject.")
        print("   This is a valid test — the model never trained on this subject.")
    else:
        print(f"\n⚠  NOTE: Checkpoint was trained with {ckpt.get('subject_id')} as test subject.")
        print(f"   You are predicting on {subj_id} — this subject WAS in the training set.")
        print("   For a fair test, use the checkpoint where the subject was held out.")


def main():
    parser = argparse.ArgumentParser(
        description="Verify model predictions against true DEAP labels"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--dat_file",   required=True,
                        help="Path to DEAP .dat file (e.g. data/DEAP/s01.dat)")
    parser.add_argument("--n_samples",  type=int, default=20,
                        help="Number of windows to check (default: 20)")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_verify(args.checkpoint, args.dat_file, args.n_samples, device)


if __name__ == "__main__":
    main()
