"""
member1_physiological/samples/extract_deap_samples.py
=====================================================
Extract one REAL DEAP window per emotion for the multimodal web demo.

For every (subject, trial) in DEAP it maps the ground-truth valence/arousal/
dominance to a 5-class emotion (shared.label_harmonization.map_deap_to_class),
then for each emotion picks the trial whose V/A/D are the CLEAREST example
(largest margin from the 5.0 thresholds), and saves a single 4-second window:
    {emotion}.npz  with keys  eeg (32,512), gsr (512,)  + provenance metadata.

Channel layout / windowing exactly match training (deap_loader.py):
    EEG   = channels 0–31        GSR = channel 36
    trim 3 s (384 samples) baseline, 4 s (512-sample) windows, 15 per trial.

Run (needs data/DEAP/s01.dat … s32.dat):
    python -m member1_physiological.samples.extract_deap_samples
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.label_harmonization import map_deap_to_class

DEAP_DIR = _ROOT / "data" / "DEAP"
OUT_DIR  = Path(__file__).parent

EMOTIONS         = ["stress", "calm", "happy", "sad", "angry"]
BASELINE_SAMPLES = 384          # 3 s @ 128 Hz — trimmed (matches deap_loader)
WINDOW_SAMPLES   = 512          # 4 s window
GSR_CHANNEL      = 36
WINDOW_IDX       = 7            # mid-trial window (~28–32 s in) — emotion well established


def main() -> None:
    if not DEAP_DIR.exists():
        raise FileNotFoundError(
            f"DEAP data not found at {DEAP_DIR}. Place s01.dat … s32.dat there first."
        )

    # emotion -> (margin, subject, trial, v, a, d, eeg, gsr) keeping the clearest example
    best: dict[str, tuple] = {}

    dat_files = sorted(DEAP_DIR.glob("s*.dat"))
    print(f"Scanning {len(dat_files)} DEAP subjects for the clearest window per emotion…\n")

    for dat in dat_files:
        with open(dat, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        data, labels = raw["data"], raw["labels"]        # (40,40,8064), (40,4)

        for t in range(data.shape[0]):
            v, a, d = (float(labels[t, 0]), float(labels[t, 1]), float(labels[t, 2]))
            # This DEAP dump ranges 0–9; the mapper validates [1,9]. Clip for the
            # mapping (0.9 and 1.0 sit on the same side of the 5.0 threshold).
            vc, ac, dc = (min(9.0, max(1.0, v)), min(9.0, max(1.0, a)), min(9.0, max(1.0, d)))
            emotion = map_deap_to_class(vc, ac, dc)
            if emotion is None:
                continue
            margin = min(abs(vc - 5.0), abs(ac - 5.0), abs(dc - 5.0))   # clarity of the example
            if emotion not in best or margin > best[emotion][0]:
                trial = data[t, :, BASELINE_SAMPLES:]                # (40, 7680)
                s = WINDOW_IDX * WINDOW_SAMPLES
                eeg = trial[:32, s:s + WINDOW_SAMPLES].astype(np.float32)   # (32, 512)
                gsr = trial[GSR_CHANNEL, s:s + WINDOW_SAMPLES].astype(np.float32)  # (512,)
                best[emotion] = (margin, dat.stem, t, v, a, d, eeg, gsr)

    for emotion in EMOTIONS:
        if emotion not in best:
            print(f"  ⚠ no DEAP trial mapped to '{emotion}' — skipped")
            continue
        margin, subj, t, v, a, d, eeg, gsr = best[emotion]
        np.savez(
            OUT_DIR / f"{emotion}.npz",
            eeg=eeg, gsr=gsr,
            subject=subj, trial=t, valence=v, arousal=a, dominance=d,
        )
        print(f"  {emotion:7s} <- {subj} trial {t:2d}  "
              f"V={v:.1f} A={a:.1f} D={d:.1f}  (margin {margin:.1f})  "
              f"eeg{eeg.shape} gsr{gsr.shape}")

    print(f"\n✓ Real DEAP sample windows written to {OUT_DIR}")


if __name__ == "__main__":
    main()
