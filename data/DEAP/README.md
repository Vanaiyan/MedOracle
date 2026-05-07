# DEAP Dataset — Download Instructions

**Used by**: Member 1 (Suhira) — Physiological Module  
**DO NOT commit raw data files to Git** (they are in `.gitignore`)

---

## About DEAP

- 32 subjects, 40 trials each × 60 seconds
- EEG: 32 channels @ 128 Hz
- GSR (EDA): 1 channel @ 4 Hz (resampled to 128 Hz in preprocessing)
- Labels: continuous Valence, Arousal, Dominance (1–9 scale)

Reference: Koelstra et al. (2012), IEEE Trans. Affective Computing

---

## How to Download

1. Go to https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Register with your university email and agree to the EULA
3. Download **data_preprocessed_python.zip** (Python-preprocessed version)
4. Extract here: `data/DEAP/`

---

## Expected File Structure After Extraction

```
data/DEAP/
├── README.md               ← this file
├── data_preprocessed_python/
│   ├── s01.dat             ← subject 01 (pickle format)
│   ├── s02.dat
│   ├── ...
│   └── s32.dat             ← subject 32
```

---

## Loading a Subject File

```python
import pickle
import numpy as np

with open("data/DEAP/data_preprocessed_python/s01.dat", "rb") as f:
    subject = pickle.load(f, encoding="latin1")

# subject["data"]   → shape (40, 40, 8064)
#   40 trials × 40 channels (32 EEG + 8 peripheral) × 8064 samples
#   At 128 Hz, 8064 samples = 63 seconds (3 s pre-trial baseline included)
#   Baseline is first 3 s → trim to last 60 s → 7680 samples

# subject["labels"] → shape (40, 4)
#   40 trials × [valence, arousal, dominance, liking]

eeg_data = subject["data"][:, :32, 3*128:]    # 32 EEG channels, skip 3s baseline
gsr_data = subject["data"][:, 36, 3*128:]     # GSR channel index 36
labels   = subject["labels"]                  # valence col 0, arousal col 1, dominance col 2
```

---

## Label Harmonization

Use the shared module:

```python
from shared.label_harmonization import map_deap_to_class

for trial_idx in range(40):
    v, a, d = labels[trial_idx, 0], labels[trial_idx, 1], labels[trial_idx, 2]
    emotion = map_deap_to_class(v, a, d)
    # emotion is "stress"|"calm"|"happy"|"sad"|"angry"|None
    # None → skip this trial (unclassifiable combination)
```

---

## Verification Checklist

After downloading, run this quick check:

```python
import os, glob
files = glob.glob("data/DEAP/data_preprocessed_python/s*.dat")
assert len(files) == 32, f"Expected 32 subject files, found {len(files)}"
print(f"✓ {len(files)} subject files found")
```
