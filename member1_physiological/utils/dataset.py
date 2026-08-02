"""
PyTorch Dataset for DEAP EEG + GSR windows

How the dataset is used in LOSO training:
  - Training set:  all windows from 31 subjects
  - Test set:      all windows from the 1 held-out subject
  - Normalisation: fit EEGPreprocessor + GSRPreprocessor on TRAINING set only,
                   then apply to both train and test sets
                   (never fit on the test subject — that would be data leakage)

Two dataset classes:
  1. DEAPWindowDataset — wraps pre-loaded WindowSample lists with preprocessing
  2. build_loso_datasets() — helper that builds train/test pairs for one fold
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# local imports
import os, sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from member1_physiological.preprocessing.deap_loader import WindowSample, SubjectData
from member1_physiological.preprocessing.eeg_preprocessor import EEGPreprocessor
from member1_physiological.preprocessing.gsr_preprocessor import GSRPreprocessor


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DEAPWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of WindowSample objects.

    Each __getitem__ returns:
      eeg   : torch.FloatTensor, shape (32, 512)
      gsr   : torch.FloatTensor, shape (512,)
      label : torch.LongTensor, scalar — integer class index 0–4

    Parameters
    ----------
    windows      : list of WindowSample (from DEAPLoader)
    eeg_prep     : fitted EEGPreprocessor (must be fitted on training data)
    gsr_prep     : fitted GSRPreprocessor (must be fitted on training data)
    augment      : if True, apply light data augmentation (training only)
    """

    def __init__(
        self,
        windows  : List[WindowSample],
        eeg_prep : EEGPreprocessor,
        gsr_prep : GSRPreprocessor,
        augment  : bool = False,
    ):
        self.windows  = windows
        self.eeg_prep = eeg_prep
        self.gsr_prep = gsr_prep
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.windows[idx]

        # Normalise (applies baseline subtraction + z-score)
        eeg = self.eeg_prep.transform(w.eeg)   # (32, 512) np.float32
        gsr = self.gsr_prep.transform(w.gsr)   # (512,)    np.float32

        # Optional training augmentation
        if self.augment:
            eeg, gsr = self._augment(eeg, gsr)

        eeg_t   = torch.from_numpy(eeg).float()                # (32, 512)
        gsr_t   = torch.from_numpy(gsr).float()                # (512,)
        label_t = torch.tensor(w.label_int, dtype=torch.long)  # scalar

        return eeg_t, gsr_t, label_t

    # ------------------------------------------------------------------
    # Data augmentation (light, preserves physiological plausibility)
    # ------------------------------------------------------------------

    def _augment(
        self, eeg: np.ndarray, gsr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply light stochastic augmentations to a single window.

        Augmentations:
        1. Gaussian noise (σ = 0.05 × signal std) — simulates electrode noise
        2. Random temporal shift (±16 samples / ±125 ms) — slight misalignment
        3. Random amplitude scaling (×0.9–1.1) — simulates impedance drift

        We do NOT:
        - Flip channels (EEG spatial layout is meaningful)
        - Time-reverse (direction matters for SCR onset)
        - Mix labels (mixup) — class boundaries are sharp in DEAP
        """
        rng = np.random.default_rng()

        # 1. Gaussian noise
        eeg = eeg + rng.normal(0, 0.05 * eeg.std(), eeg.shape).astype(np.float32)
        gsr = gsr + rng.normal(0, 0.05 * gsr.std(), gsr.shape).astype(np.float32)

        # 2. Random temporal shift (circular shift preserves length)
        shift = rng.integers(-16, 17)    # ±16 samples
        eeg   = np.roll(eeg, shift, axis=-1).astype(np.float32)
        gsr   = np.roll(gsr, shift)     .astype(np.float32)

        # 3. Amplitude scaling
        scale = rng.uniform(0.9, 1.1)
        eeg   = (eeg * scale).astype(np.float32)
        gsr   = (gsr * scale).astype(np.float32)

        return eeg, gsr

    @property
    def labels(self) -> np.ndarray:
        """Return all integer labels as a numpy array (used for class weight computation)."""
        return np.array([w.label_int for w in self.windows])

    @property
    def class_counts(self) -> Dict[int, int]:
        """Return dict of {class_int: count}."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# LOSO dataset builder
# ---------------------------------------------------------------------------

def build_loso_datasets(
    subject_data : Dict[str, SubjectData],
    test_subject : str,
    augment_train: bool = True,
) -> Tuple[DEAPWindowDataset, DEAPWindowDataset, EEGPreprocessor, GSRPreprocessor]:
    """
    Build training and test datasets for one LOSO fold.

    CRITICAL: normalisation statistics are computed on the TRAINING subjects only.
    The test subject's data is normalised using TRAINING statistics — this is
    the only correct way to avoid data leakage.

    Parameters
    ----------
    subject_data  : dict from DEAPLoader.load_all()
    test_subject  : subject ID of the held-out test subject (e.g. "s01")
    augment_train : apply augmentation to training set

    Returns
    -------
    train_dataset, test_dataset, fitted_eeg_prep, fitted_gsr_prep
    """
    # Split subjects
    train_ids = [sid for sid in subject_data if sid != test_subject]
    test_ids  = [test_subject]

    # Collect windows
    train_windows = []
    for sid in train_ids:
        train_windows.extend(subject_data[sid].windows)

    test_windows = list(subject_data[test_subject].windows)

    # Fit normalisation on training data ONLY
    train_eeg = np.stack([w.eeg for w in train_windows], axis=0)  # (N_train, 32, 512)
    train_gsr = np.stack([w.gsr for w in train_windows], axis=0)  # (N_train, 512)

    eeg_prep = EEGPreprocessor().fit(train_eeg)
    gsr_prep = GSRPreprocessor().fit(train_gsr)

    # Build datasets
    train_ds = DEAPWindowDataset(train_windows, eeg_prep, gsr_prep, augment=augment_train)
    test_ds  = DEAPWindowDataset(test_windows,  eeg_prep, gsr_prep, augment=False)

    return train_ds, test_ds, eeg_prep, gsr_prep


# ---------------------------------------------------------------------------
# K-Fold dataset builder (shared by 5-fold group and 10-fold stratified)
# ---------------------------------------------------------------------------

def build_kfold_datasets(
    train_windows : List[WindowSample],
    test_windows  : List[WindowSample],
    augment_train : bool = True,
) -> Tuple[DEAPWindowDataset, DEAPWindowDataset, EEGPreprocessor, GSRPreprocessor]:
    """
    Build train/test datasets for one k-fold split.

    Preprocessors are fitted on training windows ONLY — never on test windows.
    Used by both 5-fold GroupKFold and 10-fold StratifiedKFold.

    Parameters
    ----------
    train_windows : list of WindowSample for training
    test_windows  : list of WindowSample for testing
    augment_train : apply augmentation to training set

    Returns
    -------
    train_dataset, test_dataset, fitted_eeg_prep, fitted_gsr_prep
    """
    train_eeg = np.stack([w.eeg for w in train_windows], axis=0)  # (N_train, 32, 512)
    train_gsr = np.stack([w.gsr for w in train_windows], axis=0)  # (N_train, 512)

    eeg_prep = EEGPreprocessor().fit(train_eeg)
    gsr_prep = GSRPreprocessor().fit(train_gsr)

    train_ds = DEAPWindowDataset(train_windows, eeg_prep, gsr_prep, augment=augment_train)
    test_ds  = DEAPWindowDataset(test_windows,  eeg_prep, gsr_prep, augment=False)

    return train_ds, test_ds, eeg_prep, gsr_prep


# ---------------------------------------------------------------------------
# Self-test (requires mock data — no actual DEAP needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from member1_physiological.preprocessing.deap_loader import WindowSample
    from shared.label_harmonization import EMOTION_CLASSES

    # Create mock windows
    rng = np.random.default_rng(42)

    def make_mock_window(subject_id, trial_idx, window_idx, label_int):
        return WindowSample(
            eeg        = rng.uniform(-80, 80, (32, 512)).astype(np.float32),
            gsr        = rng.uniform(1, 10, (512,)).astype(np.float32),
            label_str  = list(EMOTION_CLASSES.keys())[label_int],
            label_int  = label_int,
            subject_id = subject_id,
            trial_idx  = trial_idx,
            window_idx = window_idx,
            valence    = 5.0,
            arousal    = 5.0,
            dominance  = 5.0,
        )

    # Mock subject data
    all_subj = {}
    for s in range(1, 6):
        sid = f"s{s:02d}"
        windows = [make_mock_window(sid, t, w, (t+w) % 5)
                   for t in range(10) for w in range(15)]
        subj = SubjectData(subject_id=sid, windows=windows)
        all_subj[sid] = subj

    # Build LOSO fold: test on s01
    train_ds, test_ds, eeg_prep, gsr_prep = build_loso_datasets(
        all_subj, test_subject="s01", augment_train=True
    )

    print(f"Train size: {len(train_ds)} windows")
    print(f"Test size:  {len(test_ds)} windows")

    # Get one batch
    eeg_t, gsr_t, label_t = train_ds[0]
    print(f"EEG tensor: {eeg_t.shape}  dtype: {eeg_t.dtype}")
    print(f"GSR tensor: {gsr_t.shape}  dtype: {gsr_t.dtype}")
    print(f"Label:      {label_t}      dtype: {label_t.dtype}")

    assert eeg_t.shape == (32, 512)
    assert gsr_t.shape == (512,)
    assert label_t.dtype == torch.long

    print("✓ DEAPWindowDataset self-test passed")
