import os
import pickle
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys

# Allow running from any directory
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.label_harmonization import map_deap_to_class, EMOTION_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE      = 128          # Hz — all channels in preprocessed DEAP
BASELINE_SAMPLES = 3 * SAMPLE_RATE   # 384 samples = 3 seconds to trim
TRIAL_SAMPLES    = 60 * SAMPLE_RATE  # 7680 samples = 60 seconds remaining
WINDOW_SAMPLES   = 4 * SAMPLE_RATE   # 512 samples = 4 seconds per window
WINDOWS_PER_TRIAL = TRIAL_SAMPLES // WINDOW_SAMPLES  # = 15

EEG_CHANNELS = list(range(32))  # indices 0–31
GSR_CHANNEL  = 36               # GSR/EDA channel index in preprocessed file

EEG_CHANNEL_NAMES = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7",
    "PO3","O1","Oz","Pz","P4","P8","PO4","O2","T8","CP6","CP2","C4",
    "FC6","FC2","F8","F4","AF4","Fp2","Fz","Cz"
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WindowSample:
    eeg        : np.ndarray      # (32, 512)
    gsr        : np.ndarray      # (512,)
    label_str  : str
    label_int  : int
    subject_id : str
    trial_idx  : int
    window_idx : int
    valence    : float
    arousal    : float
    dominance  : float


@dataclass
class SubjectData:
    subject_id    : str
    windows       : List[WindowSample] = field(default_factory=list)
    skipped_trials: List[int]          = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts = {k: 0 for k in EMOTION_CLASSES}
        for w in self.windows:
            counts[w.label_str] += 1
        return counts

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self.windows:
            return np.empty((0, 32, 512)), np.empty((0, 512)), np.empty((0,), dtype=int)
        eeg    = np.stack([w.eeg for w in self.windows], axis=0)
        gsr    = np.stack([w.gsr for w in self.windows], axis=0)
        labels = np.array([w.label_int for w in self.windows], dtype=np.int64)
        return eeg, gsr, labels


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DEAPLoader:

    def __init__(
        self,
        data_dir: str = "data/DEAP/data_preprocessed_python",
        verbose : bool = True,
    ):
        self.data_dir = data_dir
        self.verbose  = verbose
        self._subject_data: Dict[str, SubjectData] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> Dict[str, SubjectData]:
        dat_files = sorted(glob.glob(os.path.join(self.data_dir, "s*.dat")))
        if not dat_files:
            raise FileNotFoundError(
                f"No .dat files found in '{self.data_dir}'.\n"
                f"Download the DEAP preprocessed data from Kaggle or QMUL and extract to:\n"
                f"  data/DEAP/data_preprocessed_python/"
            )

        self._log(f"Found {len(dat_files)} subject files in '{self.data_dir}'")

        for fpath in dat_files:
            subj_id = os.path.splitext(os.path.basename(fpath))[0]  # e.g. "s01"
            subj_data = self._load_subject(fpath, subj_id)
            self._subject_data[subj_id] = subj_data
            self._log(
                f"  {subj_id}: {subj_data.n_windows} windows | "
                f"skipped {len(subj_data.skipped_trials)} trials | "
                f"class dist: {subj_data.class_counts}"
            )

        total_windows = sum(s.n_windows for s in self._subject_data.values())
        self._log(f"\nTotal windows loaded: {total_windows} across {len(self._subject_data)} subjects")
        return self._subject_data

    def get_flat_windows(self) -> List[WindowSample]:
        if not self._subject_data:
            raise RuntimeError("Call load_all() before get_flat_windows()")
        all_windows = []
        for subj_data in self._subject_data.values():
            all_windows.extend(subj_data.windows)
        return all_windows

    def get_subject_ids(self) -> List[str]:
        """Return sorted list of loaded subject IDs."""
        return sorted(self._subject_data.keys())

    def get_subject(self, subject_id: str) -> SubjectData:
        """Return SubjectData for a single subject (e.g. 's01')."""
        if subject_id not in self._subject_data:
            raise KeyError(f"Subject '{subject_id}' not loaded. Call load_all() first.")
        return self._subject_data[subject_id]

    def verify_installation(self) -> bool:
        s01_path = os.path.join(self.data_dir, "s01.dat")
        if not os.path.exists(s01_path):
            raise FileNotFoundError(
                f"Cannot find {s01_path}.\n"
                f"Place the DEAP preprocessed files at:\n"
                f"  data/DEAP/data_preprocessed_python/s01.dat … s32.dat"
            )
        raw = self._pickle_load(s01_path)
        data_shape   = raw["data"].shape
        labels_shape = raw["labels"].shape
        assert data_shape   == (40, 40, 8064), \
            f"Expected data shape (40,40,8064), got {data_shape}"
        assert labels_shape == (40, 4), \
            f"Expected labels shape (40,4), got {labels_shape}"
        self._log("✓ DEAP data structure verified (s01.dat)")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_subject(self, fpath: str, subj_id: str) -> SubjectData:

        raw    = self._pickle_load(fpath)
        data   = raw["data"]    # (40, 40, 8064)
        labels = raw["labels"]  # (40, 4)

        # Trim 3-second baseline from all channels
        # Resulting shape: (40, 40, 7680)  — 60 seconds at 128 Hz
        data_trimmed = data[:, :, BASELINE_SAMPLES:]

        subj_data = SubjectData(subject_id=subj_id)

        for trial_idx in range(data_trimmed.shape[0]):
            v = float(labels[trial_idx, 0])   # Valence
            a = float(labels[trial_idx, 1])   # Arousal
            d = float(labels[trial_idx, 2])   # Dominance

            # --- Label harmonization ---
            emotion_str = map_deap_to_class(v, a, d)
            if emotion_str is None:
                # Unclassifiable V/A/D combination — skip this trial
                subj_data.skipped_trials.append(trial_idx)
                continue

            emotion_int = EMOTION_CLASSES[emotion_str]

            # --- Extract EEG + GSR for this trial ---
            eeg_trial = data_trimmed[trial_idx, :32, :]    # (32, 7680)
            gsr_trial = data_trimmed[trial_idx, 36, :]     # (7680,)

            # --- Slide 4-second windows ---
            for w_idx in range(WINDOWS_PER_TRIAL):
                start = w_idx * WINDOW_SAMPLES
                end   = start + WINDOW_SAMPLES

                eeg_window = eeg_trial[:, start:end].copy()  # (32, 512)
                gsr_window = gsr_trial[start:end].copy()      # (512,)

                # Shape validation (catches any indexing bugs early)
                assert eeg_window.shape == (32, WINDOW_SAMPLES), \
                    f"EEG window shape mismatch: {eeg_window.shape}"
                assert gsr_window.shape == (WINDOW_SAMPLES,), \
                    f"GSR window shape mismatch: {gsr_window.shape}"

                window = WindowSample(
                    eeg        = eeg_window,
                    gsr        = gsr_window,
                    label_str  = emotion_str,
                    label_int  = emotion_int,
                    subject_id = subj_id,
                    trial_idx  = trial_idx,
                    window_idx = w_idx,
                    valence    = v,
                    arousal    = a,
                    dominance  = d,
                )
                subj_data.windows.append(window)

        return subj_data

    @staticmethod
    def _pickle_load(fpath: str) -> dict:
        """Load a DEAP .dat pickle file. Handles both Python 2 and 3 pickles."""
        with open(fpath, "rb") as f:
            return pickle.load(f, encoding="latin1")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Quick self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DEAPLoader")
    parser.add_argument("--data_dir", default="data/DEAP/data_preprocessed_python",
                        help="Path to folder containing s*.dat files")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only check s01.dat structure, do not load all subjects")
    args = parser.parse_args()

    loader = DEAPLoader(data_dir=args.data_dir, verbose=True)

    if args.verify_only:
        loader.verify_installation()
    else:
        dataset = loader.load_all()
        windows = loader.get_flat_windows()

        # Spot-check first window
        w0 = windows[0]
        print(f"\n--- Spot-check first window ---")
        print(f"  Subject: {w0.subject_id}, Trial: {w0.trial_idx}, Window: {w0.window_idx}")
        print(f"  Label: {w0.label_str} ({w0.label_int})")
        print(f"  EEG shape: {w0.eeg.shape}  expected (32, 512)")
        print(f"  GSR shape: {w0.gsr.shape}  expected (512,)")
        print(f"  V={w0.valence:.1f}  A={w0.arousal:.1f}  D={w0.dominance:.1f}")
        print(f"\n✓ DEAPLoader working correctly")
