from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.signal import welch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.preprocessing.deap_loader import DEAPLoader

# Constants ---------------------------------------------------------------------------

FS = 128
BANDS = {
    "delta": (0.5,  4),
    "theta": (4,    8),
    "alpha": (8,   13),
    "beta":  (13,  30),
    "gamma": (30,  50),
}
CLASS_NAMES = ["stress", "calm", "happy", "sad", "angry"]

# Feature extraction---------------------------------------------------------------------------

def extract_features(eeg: np.ndarray, gsr: np.ndarray) -> np.ndarray:
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

# LOSO with SVM ---------------------------------------------------------------------------

def run_svm_loso(subject_data: dict) -> None:
    subject_ids = sorted(subject_data.keys())

    print("Extracting features from all subjects...")
    subj_X, subj_y = {}, {}
    for subj_id in subject_ids:
        windows = subject_data[subj_id].windows
        subj_X[subj_id] = np.array([extract_features(w.eeg, w.gsr) for w in windows])
        subj_y[subj_id] = np.array([w.label_int for w in windows], dtype=np.int64)
    print(f"Done. Feature vector: {subj_X[subject_ids[0]].shape[1]} features per window\n")

    print("── SVM Baseline — 5-class LOSO ──────────────────────────────────")
    print(f"{'Fold':<6} {'Test Subject':<14} {'Accuracy':>9} {'Macro-F1':>10} {'Windows':>8}")
    print("-" * 52)

    fold_accs, fold_f1s = [], []

    for fold_idx, test_subj in enumerate(subject_ids):
        train_ids = [s for s in subject_ids if s != test_subj]

        X_train = np.concatenate([subj_X[s] for s in train_ids], axis=0)
        y_train = np.concatenate([subj_y[s] for s in train_ids], axis=0)
        X_test  = subj_X[test_subj]
        y_test  = subj_y[test_subj]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)

        print(f"{fold_idx+1:<6} {test_subj:<14} {acc:>9.4f} {f1:>10.4f} {len(y_test):>8}")

    print("-" * 52)
    mean_acc = float(np.mean(fold_accs))
    mean_f1  = float(np.mean(fold_f1s))
    print(f"{'MEAN':<20} {mean_acc:>9.4f} {mean_f1:>10.4f}")
    print(f"{'STD':<20} {np.std(fold_accs):>9.4f} {np.std(fold_f1s):>10.4f}")

    print(f"\n{'═'*52}")
    print(f"SVM Baseline   — Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"SVM Baseline   — Mean Macro-F1 : {mean_f1:.4f}")
    print(f"PhysiologicalNet (EEG+GSR+BDCMA) — LOSO Accuracy : ~32%")
    print(f"PhysiologicalNet (EEG+GSR+BDCMA) — LOSO Macro-F1 : ~0.17")
    print(f"{'═'*52}")


# CLI ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SVM baseline — 5-class LOSO on DEAP")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to DEAP .dat files")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    loader = DEAPLoader(data_dir=args.data_dir, verbose=True)
    subject_data = loader.load_all()
    run_svm_loso(subject_data)


if __name__ == "__main__":
    main()
