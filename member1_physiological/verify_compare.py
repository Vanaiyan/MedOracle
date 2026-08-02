"""
Shows for each window:
  - True label
  - BiCrossModal prediction
  - SVM prediction
  - RF prediction
  - MLP prediction

Usage:
    python -m member1_physiological.verify_compare \
        --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt \
        --data_dir   data/DEAP \
        --test_subject s01 \
        --n_samples  20
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.preprocessing.deap_loader import DEAPLoader
from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor
from member1_physiological.models.physiological_net import PhysiologicalNet
from member1_physiological.train import load_checkpoint, TrainConfig

EMOTION_CLASSES = ["stress", "calm", "happy", "sad", "angry"]
FS    = 128
BANDS = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30), "gamma":(30,50)}


# ---------------------------------------------------------------------------
# Feature extraction for baseline models
# ---------------------------------------------------------------------------

def extract_features(eeg: np.ndarray, gsr: np.ndarray) -> np.ndarray:
    """165-dim feature vector: 32ch x 5 bands + 5 GSR stats."""
    features = []
    for ch in range(eeg.shape[0]):
        freqs, psd = welch(eeg[ch], fs=FS, nperseg=256)
        for lo, hi in BANDS.values():
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            features.append(float(np.mean(psd[idx])) if len(idx) > 0 else 0.0)
    features.append(float(np.mean(gsr)))
    features.append(float(np.std(gsr)))
    features.append(float(np.max(gsr)))
    features.append(float(np.min(gsr)))
    t = np.arange(len(gsr))
    features.append(float(np.polyfit(t, gsr, 1)[0]))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_compare(checkpoint_path, data_dir, test_subject, n_samples, device):
    # ── Load all subjects ─────────────────────────────────────────────────────
    print(f"\nLoading all DEAP subjects from: {data_dir}")
    loader       = DEAPLoader(data_dir=data_dir, verbose=False)
    subject_data = loader.load_all()
    subject_ids  = sorted(subject_data.keys())

    if test_subject not in subject_ids:
        raise ValueError(f"{test_subject} not found. Available: {subject_ids}")

    train_ids    = [s for s in subject_ids if s != test_subject]
    test_windows = subject_data[test_subject].windows
    print(f"Test subject  : {test_subject} ({len(test_windows)} windows)")
    print(f"Train subjects: {len(train_ids)} subjects")

    # ── Extract features for baseline models ──────────────────────────────────
    print("\nExtracting features for baseline models...")
    X_train, y_train = [], []
    for sid in train_ids:
        for w in subject_data[sid].windows:
            X_train.append(extract_features(w.eeg, w.gsr))
            y_train.append(w.label_int)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array([extract_features(w.eeg, w.gsr) for w in test_windows])
    y_test = np.array([w.label_int for w in test_windows])

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train baseline models ─────────────────────────────────────────────────
    print("Training SVM  ...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)

    print("Training RF   ...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("Training MLP  ...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)

    svm_preds = svm.predict(X_test)
    rf_preds  = rf.predict(X_test)
    mlp_preds = mlp.predict(X_test)

    # ── Load BiCrossModal model ───────────────────────────────────────────────
    print(f"Loading BiCrossModal checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path, device)
    cfg  = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                          if k in TrainConfig.__dataclass_fields__})

    model = PhysiologicalNet(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eeg_prep = EEGPreprocessor()
    eeg_prep._mean   = ckpt["eeg_mean"]
    eeg_prep._std    = ckpt["eeg_std"]
    eeg_prep._fitted = True

    gsr_prep = GSRPreprocessor()
    gsr_prep._mean   = ckpt["gsr_mean"]
    gsr_prep._std    = ckpt["gsr_std"]
    gsr_prep._fitted = True

    deep_preds = []
    with torch.no_grad():
        for w in test_windows:
            eeg_t = torch.from_numpy(eeg_prep.transform(w.eeg)).float().unsqueeze(0).to(device)
            gsr_t = torch.from_numpy(gsr_prep.transform(w.gsr)).float().unsqueeze(0).to(device)
            logits = model(eeg_t, gsr_t)
            deep_preds.append(logits.argmax(dim=-1).item())
    deep_preds = np.array(deep_preds)

    # ── Print comparison table ────────────────────────────────────────────────
    n = min(n_samples, len(test_windows))

    print(f"\n{'─'*90}")
    print(f"{'#':<4} {'TRUE':<10} {'BiCrossModal':<14} {'SVM':<10} {'RF':<10} {'MLP':<10}")
    print(f"{'─'*90}")

    deep_c = svm_c = rf_c = mlp_c = 0

    for i in range(n):
        true  = EMOTION_CLASSES[y_test[i]]
        deep  = EMOTION_CLASSES[deep_preds[i]]
        svm_p = EMOTION_CLASSES[svm_preds[i]]
        rf_p  = EMOTION_CLASSES[rf_preds[i]]
        mlp_p = EMOTION_CLASSES[mlp_preds[i]]

        def mark(pred): return f"{pred}✓" if pred == true else f"{pred}✗"

        if deep  == true: deep_c  += 1
        if svm_p == true: svm_c   += 1
        if rf_p  == true: rf_c    += 1
        if mlp_p == true: mlp_c   += 1

        print(f"{i+1:<4} {true:<10} {mark(deep):<14} {mark(svm_p):<10} {mark(rf_p):<10} {mark(mlp_p):<10}")

    print(f"{'─'*90}")
    print(f"\n{'Model':<16} {'Correct':>8} {'Accuracy':>10}")
    print(f"{'─'*38}")
    print(f"{'BiCrossModal':<16} {deep_c:>8}/{n} {deep_c/n:>9.1%}")
    print(f"{'SVM':<16} {svm_c:>8}/{n} {svm_c/n:>9.1%}")
    print(f"{'RF':<16} {rf_c:>8}/{n} {rf_c/n:>9.1%}")
    print(f"{'MLP':<16} {mlp_c:>8}/{n} {mlp_c/n:>9.1%}")
    print(f"{'─'*38}")
    print(f"\nAll predictions on held-out subject: {test_subject} (LOSO — never seen in training)")


def main():
    parser = argparse.ArgumentParser(description="Compare BiCrossModal vs baselines on same windows")
    parser.add_argument("--checkpoint",    required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_dir",      required=True, help="Path to DEAP folder")
    parser.add_argument("--test_subject",  default="s01", help="Subject to test on (default: s01)")
    parser.add_argument("--n_samples",     type=int, default=20, help="Windows to compare")
    parser.add_argument("--device",        default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    run_compare(
        checkpoint_path = args.checkpoint,
        data_dir        = args.data_dir,
        test_subject    = args.test_subject,
        n_samples       = args.n_samples,
        device          = torch.device(args.device),
    )


if __name__ == "__main__":
    main()
