# MedOracle — Multimodal Emotional Recognition System with Explainability

**Faculty of Information Technology, University of Moratuwa — Level 4 FYP (2026)**  
**Supervisor:** Dr. Firdhous M.F.M.

---

## Team

| Member | ID | Module |
|---|---|---|
| Suhira Balarajan | 214206G | Member 1 — Physiological (EEG + GSR) |
| Vanaiyan Kirupagaran | 214215H | Member 2 — Video + Fusion |
| Adshaya Balarajah | 214024V | Member 3 — Explainability + Web App |

---

## System Overview

MedOracle fuses EEG, GSR, and facial video to classify five discrete emotions — **stress, calm, happy, sad, angry** — and explains its predictions through SHAP and an LLM chatbot.

```
EEG + GSR  ──► [M1: BiAttn Network] ──► physiological_prediction_dict
                                                    │
Facial Video ──────────────────────────────────────►│
                                                    ▼
                                        [M2: Video + Gated Fusion]
                                                    │
                                                    ▼
                                        [M3: SHAP + FastAPI + React]
```

---

## Repository Structure

```
FYP/
├── shared/                          # Shared code — imported by all members
│   ├── __init__.py
│   ├── label_harmonization.py       # DEAP + CREMA-D → 5-class mapping
│   └── data_contracts.py            # Interface dataclasses M1→M2→M3
│
├── member1_physiological/           # Suhira — EEG + GSR module
│   ├── preprocessing/
│   ├── models/
│   ├── utils/
│   ├── baseline/
│   │   ├── baselineSVM.py           # SVM baseline (LOSO, 165 features)
│   │   ├── baselineRF.py            # Random Forest baseline (LOSO, 165 features)
│   │   └── baselineMLP.py           # MLP baseline (LOSO, 165 features)
│   ├── checkpoints/
│   │   ├── random_split_best.pt     # Random split model (F1: 0.4171)
│   │   ├── model_soup_best.pt       # Model soup — 32-fold average (F1: 0.1791)
│   │   └── loso/                    # All 32 LOSO fold checkpoints
│   │       ├── fold_00_s01_best.pt
│   │       └── ...fold_31_s32_best.pt
│   ├── predict.py                   # Inference script — loads any checkpoint + DEAP .dat
│   ├── model_soup.py                # Averages all 32 LOSO fold checkpoints into one model
│   └── fix_checkpoint.py            # One-time: adds preprocessor stats to raw checkpoint
│
├── member2_video_fusion/            # Vanaiyan — Video + fusion module
│   ├── preprocessing/
│   ├── models/
│   ├── fusion/
│   └── utils/
│
├── member3_explainability/          # Adshaya — SHAP + web app
│   ├── shap/
│   ├── backend/
│   └── frontend/
│
├── data/
│   ├── DEAP/                        # Raw DEAP files go here (not committed)
│   │   └── README.md                # Download instructions for DEAP
│   └── CREMA-D/                     # Raw CREMA-D files go here (not committed)
│       └── README.md                # Download instructions for CREMA-D
│
├── tests/
│   └── test_label_harmonization.py  # 59 unit tests — all must pass
│
├── requirements_shared.txt          # Base deps (all members)
├── requirements_m1.txt              # Member 1 deps
├── requirements_m2.txt              # Member 2 deps
├── requirements_m3.txt              # Member 3 deps
├── .gitignore
└── README.md                        # This file
```

---

## Phase 1 Setup — Every Member Must Complete This

### Step 1 — Clone the repository

```bash
git clone <your-repo-url> FYP
cd FYP
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install your dependencies

Install the shared base first, then your own module's requirements.

**Member 1 (Suhira):**
```bash
pip install -r requirements_shared.txt -r requirements_m1.txt
```

**Member 2 (Vanaiyan):**
```bash
pip install -r requirements_shared.txt -r requirements_m2.txt
```

**Member 3 (Adshaya):**
```bash
pip install -r requirements_shared.txt -r requirements_m3.txt
```

### Step 4 — Verify the shared module works

Run the test suite from the project root:

```bash
python3 -m unittest tests.test_label_harmonization -v
```

Expected output:
```
Ran 59 tests in 0.002s
OK
```

All 59 tests must pass before you proceed. If any fail, do not continue — raise it with the team.

### Step 5 — Quick smoke test

```python
# Run from the FYP/ directory
from shared.label_harmonization import map_deap_to_class, map_cremad_to_class

print(map_deap_to_class(3.0, 7.0, 2.0))   # → stress
print(map_deap_to_class(7.0, 7.0, 7.0))   # → happy
print(map_cremad_to_class("FEA"))          # → stress
print(map_cremad_to_class("DIS"))          # → None  (dropped)
```

---

## Dataset Setup

Raw dataset files are **never committed to Git**. Each member downloads their own copy locally and places it in the `data/` folder.

---

### DEAP Dataset (Member 1 — Suhira)

**What it is:** 32 subjects × 40 trials. EEG (32 ch @ 128 Hz) + GSR (4 Hz). Valence/Arousal/Dominance labels (1–9 scale).

**How to download:**

1. Go to https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Register with your university email and accept the EULA
3. Download `data_preprocessed_python.zip`
4. Extract into `data/DEAP/` so the structure looks like:

```
data/DEAP/
└── data_preprocessed_python/
    ├── s01.dat
    ├── s02.dat
    ├── ...
    └── s32.dat
```

**Verify the download:**

```python
import glob
files = glob.glob("data/DEAP/data_preprocessed_python/s*.dat")
assert len(files) == 32, f"Expected 32 files, found {len(files)}"
print(f"✓ {len(files)} subject files found")
```

**Loading a subject file:**

```python
import pickle
import numpy as np

with open("data/DEAP/data_preprocessed_python/s01.dat", "rb") as f:
    subject = pickle.load(f, encoding="latin1")

# EEG: 32 channels, skip 3-second baseline (first 384 samples at 128 Hz)
eeg = subject["data"][:, :32, 3*128:]    # shape (40, 32, 7680)

# GSR: channel index 36
gsr = subject["data"][:, 36, 3*128:]    # shape (40, 7680)

# Labels: [valence, arousal, dominance, liking]
labels = subject["labels"]               # shape (40, 4)
```

**Applying label harmonization:**

```python
from shared.label_harmonization import map_deap_to_class

for trial_idx in range(40):
    v = labels[trial_idx, 0]
    a = labels[trial_idx, 1]
    d = labels[trial_idx, 2]
    emotion = map_deap_to_class(v, a, d)
    if emotion is None:
        continue   # unclassifiable V/A/D combination — skip this trial
    print(f"Trial {trial_idx}: {emotion}")
```

---

### CREMA-D Dataset (Member 2 — Vanaiyan)

**What it is:** 91 actors, ~7,442 video clips at ~30 fps. Six emotion labels: ANG, DIS, FEA, HAP, NEU, SAD.

**How to download:**

```bash
# From the FYP/ root directory (~4 GB total)
git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/CREMA-D/raw/
```

Alternatively, download only the `VideoMP4/` folder manually from:  
https://github.com/CheyneyComputerScience/CREMA-D

After download the structure should look like:

```
data/CREMA-D/
└── raw/
    ├── VideoMP4/
    │   ├── 1001_DFA_ANG_XX.mp4
    │   ├── 1001_DFA_DIS_XX.mp4
    │   ├── ...
    │   └── 1091_WSI_SAD_HI.mp4
    └── processedResults/
        └── summaryTable.csv
```

**Verify the download:**

```python
from pathlib import Path

video_dir = Path("data/CREMA-D/raw/VideoMP4")
mp4_files  = list(video_dir.glob("*.mp4"))
actor_ids  = set(int(f.stem.split("_")[0]) for f in mp4_files)

assert len(mp4_files) >= 7000, f"Expected ~7,442 clips, found {len(mp4_files)}"
assert len(actor_ids) == 91,   f"Expected 91 actors, found {len(actor_ids)}"
print(f"✓ {len(mp4_files)} clips, {len(actor_ids)} actors")
```

**Filename convention:**

```
{ActorID}_{SentenceID}_{EmotionLabel}_{Intensity}.mp4

Example: 1045_IEO_FEA_HI.mp4
  → Actor 1045 | Sentence IEO | FEAR | High intensity
```

**Applying label harmonization:**

```python
from pathlib import Path
from shared.label_harmonization import map_cremad_to_class

video_dir = Path("data/CREMA-D/raw/VideoMP4")
records   = []

for mp4_path in sorted(video_dir.glob("*.mp4")):
    parts        = mp4_path.stem.split("_")
    actor_id     = int(parts[0])
    emotion_code = parts[2]                        # e.g. "ANG"

    emotion_label = map_cremad_to_class(emotion_code)
    if emotion_label is None:
        continue                                   # Drop DIS samples

    records.append({
        "path":     str(mp4_path),
        "actor_id": actor_id,
        "emotion":  emotion_label,
    })

print(f"✓ {len(records)} usable clips after dropping DIS")
# Expect ~6,250–6,500 clips
```

---

## Label Mapping Reference

### DEAP → 5-class (Russell 1980 / Mehrabian 1996)

| Class | Valence | Arousal | Dominance | Model |
|-------|---------|---------|-----------|-------|
| stress | < 5 | ≥ 5 | < 5 | Mehrabian PAD |
| calm | ≥ 5 | < 5 | ≥ 5 | Russell Circumplex |
| happy | ≥ 5 | ≥ 5 | ≥ 5 | Mehrabian PAD |
| happy | ≥ 5 | ≥ 5 | < 5 | Russell Circumplex (extended) |
| sad | < 5 | < 5 | < 5 | Russell Circumplex |
| angry | < 5 | ≥ 5 | ≥ 5 | Mehrabian PAD |

Boundary rule: values exactly equal to 5 are treated as ≥ 5.  
Only one combination remains unclassifiable: V≥5, A<5, D<5 → returns `None`.

**Happy class extension — academic justification:**  
Russell's (1980) Circumplex model defines happy purely as high valence + high arousal,
without requiring high dominance. Dominance is an addition from Mehrabian's (1996) PAD model.
The combination V≥5, A≥5, D<5 (high valence, high arousal, low dominance) maps to happy
under Russell's definition — for example, feeling excited and joyful but not in control.
This extension recovers previously unclassified samples and increases happy class
representation without fabricating data. It is consistent with how many DEAP-based papers
apply the Circumplex model (using V and A only for the primary emotion axis).

### CREMA-D → 5-class

| CREMA-D | MedOracle | Justification |
|---------|-----------|---------------|
| ANG | angry | Direct match |
| HAP | happy | Direct match |
| SAD | sad | Direct match |
| NEU | calm | Neutral ≈ calm low arousal |
| FEA | stress | Fear activates sympathetic system (Kreibig 2010) |
| DIS | **dropped** | No clean mapping — exclude from training |

---

## Interface Contracts

All members must produce and consume data in exactly these formats.

### M1 → M2: `physiological_prediction_dict`

```python
{
    "predicted_emotion":   "stress",          # one of the 5 class labels
    "confidence":          0.82,              # entropy-based, float in [0, 1]
    "class_probabilities": {
        "stress": 0.82, "calm": 0.05,
        "happy": 0.04, "sad": 0.06, "angry": 0.03
    },
    "signal_quality": {
        "eeg": "good",                        # "good" | "degraded" | "poor"
        "gsr": "degraded"
    }
}
```

### M2 → M3: `prediction_output`

```python
{
    "predicted_emotion":   "stress",
    "confidence":          0.78,
    "class_probabilities": {
        "stress": 0.78, "calm": 0.06,
        "happy": 0.05, "sad": 0.07, "angry": 0.04
    },
    "modality_weights":    {"physio": 0.65, "video": 0.35},
    "signal_quality":      {"eeg": "good", "gsr": "degraded", "video": "good"},
    "per_modality_predictions": {
        "physio": {"predicted_emotion": "stress", "confidence": 0.82,
                   "class_probabilities": {...}},
        "video":  {"predicted_emotion": "stress", "confidence": 0.71,
                   "class_probabilities": {...}}
    }
}
```

Use the helpers in `shared/data_contracts.py` to build and validate these dicts:

```python
from shared.data_contracts import make_physiological_prediction, make_prediction_output
```

---

## Running Tests

```bash
# From the FYP/ root — no external dependencies needed
python3 -m unittest tests.test_label_harmonization -v

# If pytest is installed
pytest tests/ -v
```

---

## Git Workflow

```bash
# Datasets are never committed — data/DEAP/ and data/CREMA-D/ are excluded in .gitignore

# Member 1 trained model checkpoints ARE committed via .gitignore exceptions:
#   member1_physiological/checkpoints/random_split_best.pt
#   member1_physiological/checkpoints/model_soup_best.pt
#   member1_physiological/checkpoints/loso/*.pt
# All other *.pt files (outside member1_physiological/checkpoints/) are still excluded.

# Suggested branching
git checkout -b member1/deap-preprocessing   # Suhira
git checkout -b member2/crema-preprocessing  # Vanaiyan
git checkout -b member3/shap-layer           # Adshaya

# Push and open a PR when a module is ready for integration
```

---

## Member 1 — Training Results (July 2026)

### Model: PhysiologicalNet (BiCrossModal Attention)
- Architecture: EEGEncoder + GSREncoder + BidirectionalCrossModalAttention + ClassificationHead
- Input: EEG (32 channels × 512 samples) + GSR (512 samples after resampling to 128 Hz)
- Output: 5-class softmax (stress / calm / happy / sad / angry)
- Dataset: DEAP (32 subjects × 40 trials × 15 windows = 19,200 windows)

---

### Evaluation 1 — Random Split (80/20 stratified)

| Metric | Result |
|--------|--------|
| Macro-F1 | **0.4171** |
| Accuracy | **~42%** |
| Checkpoint | `checkpoints/random_split_best.pt` |

**When to use:** Demo scenarios where the model has already seen data from the test subject
(simulates a real-world calibration session). Gives higher accuracy and more varied predictions.

---

### Evaluation 2 — LOSO (Leave-One-Subject-Out, 32 folds)

| Metric | Result |
|--------|--------|
| Mean Macro-F1 | **0.1791 ± 0.0411** |
| Folds | 32 (one per DEAP subject) |
| Checkpoints | `checkpoints/loso/fold_00_s01_best.pt` … `fold_31_s32_best.pt` |

**Why LOSO is hard:** The model is tested on a completely unseen person each fold. EEG signals
are highly subject-specific (amplitude, baseline, noise differ per person), so cross-subject
generalisation is inherently difficult. This is a known limitation of the DEAP benchmark.

---

### Evaluation 3 — Model Soup (Wortsman et al. 2022)

| Metric | Result |
|--------|--------|
| Method | F1-weighted average of all 32 LOSO fold checkpoints |
| Mean Macro-F1 | **0.1791** |
| Checkpoint | `checkpoints/model_soup_best.pt` |

The model soup combines knowledge from all 32 subjects into a single deployable model.
Better folds contribute more weight; the averaged preprocessor stats represent all subjects.

**Run:**
```bash
python -m member1_physiological.model_soup \
    --checkpoint_dir member1_physiological/checkpoints/loso \
    --output member1_physiological/checkpoints/model_soup_best.pt
```

---

### Baseline Comparison (all under LOSO protocol, 165 features)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| SVM (RBF kernel) | 16.48% | 0.0975 |
| Random Forest (100 trees) | 24.13% | 0.1171 |
| MLP (2-layer) | pending | pending |
| **PhysiologicalNet (LOSO)** | ~32% | **0.1791** |
| PhysiologicalNet (random split) | ~42% | **0.4171** |

Features for baselines: 160 EEG features (32 channels × 5 frequency bands: delta/theta/alpha/beta/gamma) + 5 GSR statistical features = 165 total.

**Run baselines:**
```bash
python -m member1_physiological.baseline.baselineSVM
python -m member1_physiological.baseline.baselineRF
python -m member1_physiological.baseline.baselineMLP
```

---

### Running Inference with predict.py

```bash
# Using model soup (recommended for demo)
python -m member1_physiological.predict \
    --checkpoint member1_physiological/checkpoints/model_soup_best.pt \
    --dat_file data/DEAP/data_preprocessed_python/s01.dat \
    --trial 0 --window 0

# Using random split model
python -m member1_physiological.predict \
    --checkpoint member1_physiological/checkpoints/random_split_best.pt \
    --dat_file data/DEAP/data_preprocessed_python/s01.dat \
    --trial 0 --window 0
```

Output format (`physiological_prediction_dict`):
```python
{
    "predicted_emotion": "stress",
    "confidence": 0.74,
    "class_probabilities": {
        "stress": 0.74, "calm": 0.08, "happy": 0.06, "sad": 0.07, "angry": 0.05
    },
    "signal_quality": {"eeg": "good", "gsr": "good"}
}
```

---

### Phase 3 — Integration and Demo Preparation (August 1–3)
- End-to-end integration with Member 2 (fusion) and Member 3 (SHAP + web app)
- Demo input: DEAP `.dat` files for EEG + GSR; video from DEAP face video (s01–s22) or CREMA-D matched by emotion label
- Share `model_soup_best.pt` with Vanaiyan for fusion integration

---

### Interface Contract — Unchanged Throughout All Phases
```python
physiological_prediction_dict = {
    "predicted_emotion":   str,    # "stress"|"calm"|"happy"|"sad"|"angry"
    "confidence":          float,  # entropy-based, 0–1
    "class_probabilities": {       # always 5 keys, sum to 1.0
        "stress": float, "calm": float, "happy": float,
        "sad": float, "angry": float
    },
    "signal_quality": {
        "eeg": str,   # "good"|"degraded"|"poor"
        "gsr": str
    }
}
```

---

## Key References

1. Russell (1980) — Circumplex model of affect — *JPSP* 39(6)
2. Mehrabian (1996) — PAD emotional state model — *J. Nonverbal Behavior*
3. Kreibig (2010) — Fear → sympathetic activation — *Biological Psychology*
4. Koelstra et al. (2012) — DEAP dataset — *IEEE Trans. Affective Computing*
5. Cao et al. (2014) — CREMA-D dataset — *IEEE Trans. Affective Computing*
6. Guo et al. (2017) — Neural network calibration — *ICML*
7. He et al. (2016) — ResNet — *CVPR*
8. Poria et al. (2017) — Late vs early fusion — *Information Fusion*
