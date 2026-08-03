"""
Inference module for PhysiologicalNet.

Loads a trained checkpoint and predicts the emotion class for a single
4-second EEG + GSR window. Produces the physiological_prediction_dict
that is passed to M2 (fusion module) via the shared interface contract.

Interface contract output (M1 -> M2):
    {
        "predicted_emotion":   str,     # "stress"|"calm"|"happy"|"sad"|"angry"
        "confidence":          float,   # entropy-based, 0.0-1.0
        "class_probabilities": dict,    # {emotion: probability} sums to 1.0
        "signal_quality": {
            "eeg": str,                 # "good"|"degraded"|"poor"
            "gsr": str
        }
    }

Usage:
    # Predict on a single window from a DEAP .dat file (demo):
    python -m member1_physiological.predict ^
        --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt ^
        --dat_file data/DEAP/s01.dat ^
        --trial 0 --window 0

    # Predict on all 15 windows of one trial:
    python -m member1_physiological.predict ^
        --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt ^
        --dat_file data/DEAP/s01.dat ^
        --trial 0

    # Find and use the best checkpoint automatically:
    python -m member1_physiological.predict ^
        --checkpoint_dir member1_physiological/checkpoints ^
        --dat_file data/DEAP/s01.dat ^
        --trial 0 --window 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.models.physiological_net import PhysiologicalNet
from member1_physiological.models.signal_quality import (
    assess_eeg_quality,
    assess_gsr_quality,
)
from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor
from shared.data_contracts import (
    EMOTION_CLASSES,
    make_physiological_prediction,
)

# Label index -> name (0=stress, 1=calm, 2=happy, 3=sad, 4=angry)
IDX_TO_EMOTION = {v: k for k, v in EMOTION_CLASSES.items()}

# DEAP constants (must match deap_loader.py)
_BASELINE_SAMPLES = 384    # 3s × 128 Hz
_WINDOW_SAMPLES   = 512    # 4s × 128 Hz
_WINDOWS_PER_TRIAL = 15


# ---------------------------------------------------------------------------
# Entropy-based confidence  (Guo et al., 2017)
# ---------------------------------------------------------------------------

def entropy_confidence(probs: np.ndarray) -> float:
    """
    Compute entropy-based confidence score.

        c = 1 - H(P) / log(K)
        H(P) = -sum(p * log(p))
        K = number of classes (5)

    Returns 1.0 for a perfectly certain prediction (all mass on one class)
    and 0.0 for a perfectly uniform distribution (maximum uncertainty).

    This formula is used in the M2 gated fusion gate score:
        g_physio = c_physio x alpha_physio
    """
    K   = len(probs)
    H   = -np.sum(probs * np.log(probs + 1e-10))      # entropy
    c   = 1.0 - H / np.log(K)
    return float(np.clip(c, 0.0, 1.0))


# ---------------------------------------------------------------------------
# PhysiologicalPredictor
# ---------------------------------------------------------------------------

class PhysiologicalPredictor:
    """
    Wraps a trained PhysiologicalNet checkpoint for single-window inference.

    Reconstructs the model AND the fitted preprocessors from the checkpoint,
    so no separate preprocessor files are needed.

    Parameters
    ----------
    checkpoint_path : path to a .pt file saved by train.py
    device          : "cpu" | "cuda" | "auto"
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Support two checkpoint formats:
        #   1. Structured dict: keys = config, model_state_dict, eeg_mean, eeg_std, ...
        #      (saved by train.py via torch.save({...}, path))
        #   2. Raw state dict: keys are layer names directly
        #      (saved via torch.save(model.state_dict(), path))
        if "config" in ckpt:
            cfg          = ckpt["config"]
            model_weights = ckpt["model_state_dict"]
            self.subject_id          = ckpt.get("subject_id", "unknown")
            self.best_epoch          = ckpt.get("best_epoch", -1)
            self.checkpoint_macro_f1 = ckpt.get("fold_metrics", {}).get("macro_f1", None)
            has_prep_stats = "eeg_mean" in ckpt
        else:
            # Raw state dict — use default training config
            cfg           = {"d_model": 128, "n_heads": 4, "n_layers": 2, "dropout": 0.3}
            model_weights  = ckpt
            self.subject_id          = "random_split"
            self.best_epoch          = -1
            self.checkpoint_macro_f1 = 0.4171
            has_prep_stats = False

        # Reconstruct model
        self.model = PhysiologicalNet(
            d_model  = cfg["d_model"],
            n_heads  = cfg["n_heads"],
            n_layers = cfg["n_layers"],
            dropout  = cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(model_weights)
        self.model.eval()

        # Reconstruct preprocessors
        self.eeg_prep = EEGPreprocessor()
        self.gsr_prep = GSRPreprocessor()

        if has_prep_stats:
            self.eeg_prep.load_stats(mean=ckpt["eeg_mean"], std=ckpt["eeg_std"])
            self.gsr_prep.load_stats(mean=float(ckpt["gsr_mean"]), std=float(ckpt["gsr_std"]))
        else:
            # No saved stats — apply only within-window baseline subtraction (z-score skipped)
            self.eeg_prep.load_stats(
                mean=np.zeros(32, dtype=np.float32),
                std=np.ones(32, dtype=np.float32),
            )
            self.gsr_prep.load_stats(mean=0.0, std=1.0)

        # Store metadata for logging
        self.checkpoint_path = checkpoint_path

    # ------------------------------------------------------------------
    # Main inference method
    # ------------------------------------------------------------------

    def predict(
        self,
        eeg: np.ndarray,
        gsr: np.ndarray,
    ) -> dict:
        """
        Predict the emotion class for a single 4-second window.

        Parameters
        ----------
        eeg : np.ndarray, shape (32, 512)
              Raw (or QMUL-preprocessed) EEG window. Units: uV.
        gsr : np.ndarray, shape (512,)
              Raw GSR window. Units: raw ADC (as in DEAP .dat files).

        Returns
        -------
        physiological_prediction_dict — validated against the M1->M2 contract:
            {
                "predicted_emotion":   str,
                "confidence":          float,
                "class_probabilities": {"stress":f, "calm":f, "happy":f,
                                        "sad":f, "angry":f},
                "signal_quality":      {"eeg": str, "gsr": str}
            }
        """
        # --- Validate input shapes ---
        if eeg.shape != (32, 512):
            raise ValueError(f"EEG must be shape (32, 512), got {eeg.shape}")
        if gsr.shape != (512,):
            raise ValueError(f"GSR must be shape (512,), got {gsr.shape}")

        eeg = eeg.astype(np.float32)
        gsr = gsr.astype(np.float32)

        # --- Signal quality assessment (on raw signal before preprocessing) ---
        eeg_quality = assess_eeg_quality(eeg, raw=True)
        gsr_quality = assess_gsr_quality(gsr, use_raw_units=True)

        # --- Preprocess (z-score using training statistics from checkpoint) ---
        eeg_norm = self.eeg_prep.transform(eeg)   # (32, 512)
        gsr_norm = self.gsr_prep.transform(gsr)   # (512,)

        # --- Convert to tensors ---
        eeg_t = torch.from_numpy(eeg_norm).float().unsqueeze(0).to(self.device)  # (1,32,512)
        gsr_t = torch.from_numpy(gsr_norm).float().unsqueeze(0).to(self.device)  # (1,512)

        # --- Forward pass ---
        with torch.no_grad():
            logits = self.model(eeg_t, gsr_t)          # (1, 5)
            probs  = F.softmax(logits, dim=-1)          # (1, 5)

        probs_np  = probs.squeeze(0).cpu().numpy()      # (5,)
        pred_idx  = int(probs_np.argmax())
        pred_name = IDX_TO_EMOTION[pred_idx]

        # --- Entropy-based confidence ---
        confidence = entropy_confidence(probs_np)

        # --- Build class_probabilities dict ---
        class_probs = {
            IDX_TO_EMOTION[i]: float(probs_np[i])
            for i in range(len(EMOTION_CLASSES))
        }

        # --- Build and validate output via shared contract ---
        result = make_physiological_prediction(
            predicted_emotion   = pred_name,
            confidence          = confidence,
            class_probabilities = class_probs,
            eeg_quality         = eeg_quality,
            gsr_quality         = gsr_quality,
        )

        return result

    # ------------------------------------------------------------------
    # Batch inference (multiple windows at once — faster for evaluation)
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        eeg_batch: np.ndarray,
        gsr_batch: np.ndarray,
    ) -> List[dict]:
        """
        Predict emotions for a batch of windows.

        Parameters
        ----------
        eeg_batch : np.ndarray, shape (N, 32, 512)
        gsr_batch : np.ndarray, shape (N, 512)

        Returns
        -------
        List of N physiological_prediction_dicts
        """
        N = eeg_batch.shape[0]
        assert gsr_batch.shape[0] == N

        eeg_batch = eeg_batch.astype(np.float32)
        gsr_batch = gsr_batch.astype(np.float32)

        # Quality: assessed per window
        qualities = [
            (
                assess_eeg_quality(eeg_batch[i], raw=True),
                assess_gsr_quality(gsr_batch[i], use_raw_units=True),
            )
            for i in range(N)
        ]

        # Preprocess entire batch
        eeg_norm = self.eeg_prep.transform(eeg_batch)   # (N, 32, 512)
        gsr_norm = self.gsr_prep.transform(gsr_batch)   # (N, 512)

        eeg_t = torch.from_numpy(eeg_norm).float().to(self.device)   # (N,32,512)
        gsr_t = torch.from_numpy(gsr_norm).float().to(self.device)   # (N,512)

        with torch.no_grad():
            logits = self.model(eeg_t, gsr_t)          # (N, 5)
            probs  = F.softmax(logits, dim=-1)          # (N, 5)

        probs_np = probs.cpu().numpy()                  # (N, 5)

        results = []
        for i in range(N):
            pred_idx  = int(probs_np[i].argmax())
            pred_name = IDX_TO_EMOTION[pred_idx]
            class_probs = {IDX_TO_EMOTION[j]: float(probs_np[i, j]) for j in range(5)}
            results.append(
                make_physiological_prediction(
                    predicted_emotion   = pred_name,
                    confidence          = entropy_confidence(probs_np[i]),
                    class_probabilities = class_probs,
                    eeg_quality         = qualities[i][0],
                    gsr_quality         = qualities[i][1],
                )
            )
        return results

    def __repr__(self) -> str:
        f1_str = f"{self.checkpoint_macro_f1:.4f}" if self.checkpoint_macro_f1 else "unknown"
        return (
            f"PhysiologicalPredictor("
            f"subject={self.subject_id}, "
            f"epoch={self.best_epoch}, "
            f"macro_f1={f1_str}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Scan a checkpoints directory and return the path of the checkpoint
    with the highest fold macro-F1.

    Useful when you want to use the single best model across all folds
    for a quick demo (instead of the full LOSO ensemble).
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    pt_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ]

    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint files found in: {checkpoint_dir}")

    best_path = None
    best_f1   = -1.0

    for fpath in pt_files:
        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
        f1   = ckpt.get("fold_metrics", {}).get("macro_f1", -1.0)
        if f1 > best_f1:
            best_f1   = f1
            best_path = fpath

    print(f"Best checkpoint: {os.path.basename(best_path)}  (macro-F1={best_f1:.4f})")
    return best_path


def list_checkpoints(checkpoint_dir: str) -> None:
    """Print a summary table of all saved checkpoints."""
    pt_files = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
    ])

    if not pt_files:
        print("No checkpoints found.")
        return

    print(f"\n{'File':<35}  {'Subject':<8}  {'Epoch':<6}  {'Macro-F1'}")
    print("-" * 65)
    for fname in pt_files:
        ckpt = torch.load(os.path.join(checkpoint_dir, fname),
                          map_location="cpu", weights_only=False)
        subj  = ckpt.get("subject_id", "?")
        epoch = ckpt.get("best_epoch", "?")
        f1    = ckpt.get("fold_metrics", {}).get("macro_f1", -1.0)
        print(f"{fname:<35}  {subj:<8}  {str(epoch):<6}  {f1:.4f}")
    print()


# ---------------------------------------------------------------------------
# Demo helper — extract a window from a raw DEAP .dat file
# ---------------------------------------------------------------------------

def extract_window_from_dat(
    dat_path  : str,
    trial_idx : int = 0,
    window_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a DEAP .dat file and extract one 4-second EEG + GSR window.

    Parameters
    ----------
    dat_path   : path to a DEAP .dat file (e.g. data/DEAP/s01.dat)
    trial_idx  : which of the 40 trials (0-indexed)
    window_idx : which of the 15 windows within the trial (0-indexed)

    Returns
    -------
    eeg    : np.ndarray (32, 512)
    gsr    : np.ndarray (512,)
    meta   : dict with trial/window indices and raw V/A/D/L label values
    """
    with open(dat_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    data   = raw["data"]     # (40, 40, 8064)
    labels = raw["labels"]   # (40, 4)

    if trial_idx >= 40:
        raise ValueError(f"trial_idx must be 0-39, got {trial_idx}")
    if window_idx >= _WINDOWS_PER_TRIAL:
        raise ValueError(f"window_idx must be 0-14, got {window_idx}")

    # Trim 3-second baseline
    trial_data = data[trial_idx, :, _BASELINE_SAMPLES:]   # (40, 7680)

    # Extract window
    start = window_idx * _WINDOW_SAMPLES
    end   = start + _WINDOW_SAMPLES

    eeg = trial_data[:32, start:end].astype(np.float32)   # (32, 512)
    gsr = trial_data[36,  start:end].astype(np.float32)   # (512,)

    meta = {
        "trial_idx"  : trial_idx,
        "window_idx" : window_idx,
        "valence"    : float(labels[trial_idx, 0]),
        "arousal"    : float(labels[trial_idx, 1]),
        "dominance"  : float(labels[trial_idx, 2]),
        "liking"     : float(labels[trial_idx, 3]),
    }

    return eeg, gsr, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhysiologicalNet inference on a DEAP window"
    )

    # Checkpoint — either a specific file or auto-find best in a directory
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a specific .pt checkpoint file",
    )
    ckpt_group.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory of checkpoints — auto-selects the one with best macro-F1",
    )

    parser.add_argument(
        "--dat_file",
        type=str,
        required=True,
        help="Path to a DEAP .dat file (e.g. data/DEAP/s01.dat)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial index within the .dat file (0-39). Default: 0",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Window index within the trial (0-14). "
             "Omit to predict all 15 windows of the trial.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' | 'cuda' | 'cpu'. Default: auto",
    )
    parser.add_argument(
        "--list_checkpoints",
        action="store_true",
        help="List all checkpoints in --checkpoint_dir and exit.",
    )
    return parser.parse_args()


def _print_result(result: dict, meta: dict, window_idx: int) -> None:
    """Pretty-print a single prediction result."""
    sep = "-" * 55
    print(sep)
    print(f"Window {window_idx:2d} | Trial {meta['trial_idx']}  "
          f"(V={meta['valence']:.1f} A={meta['arousal']:.1f} "
          f"D={meta['dominance']:.1f})")
    print(sep)
    print(f"  Predicted emotion : {result['predicted_emotion'].upper()}")
    print(f"  Confidence        : {result['confidence']:.4f}  "
          f"({'HIGH' if result['confidence'] > 0.6 else 'MEDIUM' if result['confidence'] > 0.35 else 'LOW'})")
    print(f"  EEG quality       : {result['signal_quality']['eeg']}")
    print(f"  GSR quality       : {result['signal_quality']['gsr']}")
    print()
    print("  Class probabilities:")
    probs = result["class_probabilities"]
    for emotion, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        marker = " <--" if emotion == result["predicted_emotion"] else ""
        print(f"    {emotion:<8}: {prob:.4f}  {bar}{marker}")
    print()


def main() -> None:
    args = parse_args()

    # --- List checkpoints mode ---
    if args.list_checkpoints:
        if args.checkpoint_dir is None:
            print("ERROR: --checkpoint_dir required for --list_checkpoints")
            return
        list_checkpoints(args.checkpoint_dir)
        return

    # --- Resolve checkpoint path ---
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_best_checkpoint(args.checkpoint_dir)

    # --- Build predictor ---
    print(f"\nLoading checkpoint: {ckpt_path}")
    predictor = PhysiologicalPredictor(ckpt_path, device=args.device)
    print(predictor)
    print()

    # --- Determine windows to predict ---
    if args.window is not None:
        window_indices = [args.window]
    else:
        window_indices = list(range(_WINDOWS_PER_TRIAL))   # all 15

    # --- Extract and predict ---
    all_results = []
    for w_idx in window_indices:
        eeg, gsr, meta = extract_window_from_dat(args.dat_file, args.trial, w_idx)
        result = predictor.predict(eeg, gsr)
        all_results.append(result)
        _print_result(result, meta, w_idx)

    # --- Summary if multiple windows ---
    if len(all_results) > 1:
        from collections import Counter
        votes = Counter(r["predicted_emotion"] for r in all_results)
        majority = votes.most_common(1)[0][0]
        avg_conf = float(np.mean([r["confidence"] for r in all_results]))
        print("=" * 55)
        print(f"TRIAL SUMMARY (15 windows majority vote)")
        print(f"  Dominant emotion : {majority.upper()}")
        print(f"  Avg confidence   : {avg_conf:.4f}")
        print(f"  Window votes     : {dict(votes)}")
        print("=" * 55)

    # --- Print the M1->M2 contract dict for the last predicted window ---
    print("\nM1 -> M2 interface contract output (last window):")
    print(json.dumps(all_results[-1], indent=2))


if __name__ == "__main__":
    main()

"""
Commands:

Single window demo:
    python -m member1_physiological.predict ^
        --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt ^
        --dat_file data/DEAP/s01.dat --trial 0 --window 0

All 15 windows of trial 0:
    python -m member1_physiological.predict ^
        --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt ^
        --dat_file data/DEAP/s01.dat --trial 0

Auto-find best checkpoint:
    python -m member1_physiological.predict ^
        --checkpoint_dir member1_physiological/checkpoints ^
        --dat_file data/DEAP/s01.dat --trial 0 --window 0

List all saved checkpoints:
    python -m member1_physiological.predict ^
        --checkpoint_dir member1_physiological/checkpoints ^
        --list_checkpoints
"""
