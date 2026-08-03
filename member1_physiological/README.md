# Member 1 — Physiological Module (EEG + GSR)

**Author:** Suhira Balarajan (214206G)  
**Dataset:** DEAP — 32 subjects, 32-channel EEG (128 Hz) + GSR (4 Hz), 40 trials × 60 s each  
**Model:** Bidirectional Cross-Modal Attention Network (BiCrossModal)  
**Output:** `predict_physiological(eeg, gsr)` → `physiological_prediction_dict`

---

## Quick Setup

```bash
# From repo root (MedOracle/)
pip install -r requirements.txt

# Place DEAP files at:
# data/DEAP/s01.dat ... s32.dat
```

---

## Training Modes

Four evaluation protocols are available via `--eval_mode`. Run all commands from
the **repo root** (`MedOracle/`).

### Mode 1 — LOSO (Primary / Most Rigorous)

Leave-One-Subject-Out: train on 31 subjects, test on 1 unseen subject. Repeated
for all 32 subjects. Tests true cross-subject generalisation.

```bash
# Full 32-fold LOSO (run overnight — ~8–12 hours on CPU, ~2–3 hours on GPU):
python -m member1_physiological.train --data_dir data/DEAP --eval_mode loso

# Single fold for quick debug (s01 only):
python -m member1_physiological.train --data_dir data/DEAP --eval_mode loso --test_subject s01

# With GPU:
python -m member1_physiological.train --data_dir data/DEAP --eval_mode loso --device cuda
```

**Output files:**
- `member1_physiological/checkpoints/fold_XX_sXX_best.pt` — 32 checkpoints
- `member1_physiological/logs/loso_training.log` — epoch-by-epoch log
- `member1_physiological/logs/loso_summary.json` — aggregated results

**Expected result:** Mean macro-F1 ≈ 0.1791 ± 0.0411 (32 subjects, 5-class, cross-subject)

---

### Mode 2 — 5-Fold Subject GroupKFold (Cross-Subject, Faster)

Groups subjects by ID — no subject appears in both train and test within the same
fold. 5 folds × ~6–7 test subjects each. Cross-subject evaluation, faster than LOSO.

```bash
python -m member1_physiological.train --data_dir data/DEAP --eval_mode 5fold

# With GPU:
python -m member1_physiological.train --data_dir data/DEAP --eval_mode 5fold --device cuda
```

**Output files:**
- `member1_physiological/checkpoints/5fold_group_fold01_best.pt` … `fold05_best.pt`
- `member1_physiological/logs/5fold_group_training.log`
- `member1_physiological/logs/5fold_group_summary.json`

**Expected result:** Mean macro-F1 ≈ 0.30–0.45 (higher than LOSO — more training data per fold)

---

### Mode 3 — 10-Fold Stratified KFold (Within-Subject / AlgoRidge-style)

Pools all windows across all subjects and splits randomly into 10 folds, stratified
by class. The same subject can appear in both train and test (within-subject leakage).
Use only for protocol comparison — not the primary metric.

```bash
python -m member1_physiological.train --data_dir data/DEAP --eval_mode 10fold

# With GPU:
python -m member1_physiological.train --data_dir data/DEAP --eval_mode 10fold --device cuda
```

**Output files:**
- `member1_physiological/checkpoints/10fold_strat_fold01_best.pt` … `fold10_best.pt`
- `member1_physiological/logs/10fold_stratified_training.log`
- `member1_physiological/logs/10fold_stratified_summary.json`

**Expected result:** Mean macro-F1 ≈ 0.55–0.75 (inflated — within-subject leakage)

> **Warning:** This protocol mirrors AlgoRidge's setup. Results are not comparable
> to LOSO or 5-fold group without explicitly noting the protocol difference.

---

### Mode 4 — Random Split 80/20 (Demo / Baseline)

Randomly splits all windows 80% train / 20% test across all subjects. Single training
run — no folds. Subject leakage present.

```bash
python -m member1_physiological.train --data_dir data/DEAP --eval_mode random_split
```

**Output files:**
- `member1_physiological/checkpoints/random_split_best.pt`
- `member1_physiological/logs/random_split_training.log`

**Expected result:** Macro-F1 ≈ 0.4171

---

## Baseline Models

Pre-built classical ML baselines for comparison:

```bash
# SVM baseline (LOSO):
python -m member1_physiological.baseline.baselineSVM --data_dir data/DEAP

# Random Forest baseline (LOSO):
python -m member1_physiological.baseline.baselineRF --data_dir data/DEAP

# MLP baseline (LOSO):
python -m member1_physiological.baseline.baselineMLP --data_dir data/DEAP
```

---

## Evaluation Comparison Table

| Protocol | Eval mode flag | Folds | Subject leakage? | Expected macro-F1 |
|---|---|---|---|---|
| LOSO (primary) | `loso` | 32 | No | ~0.18 ± 0.04 |
| 5-fold GroupKFold | `5fold` | 5 | No | ~0.30–0.45 |
| 10-fold Stratified | `10fold` | 10 | **Yes** | ~0.55–0.75 |
| Random split | `random_split` | 1 | **Yes** | ~0.42 |
| Baseline SVM (LOSO) | — | 32 | No | ~0.15–0.20 |

**Primary metric for the report: LOSO macro-F1**

---

## Running Inference (Testing a Checkpoint)

After training, use `predict.py` to run a single prediction:

```bash
# Test with a saved checkpoint (LOSO fold s01):
python -m member1_physiological.predict \
    --checkpoint member1_physiological/checkpoints/fold_00_s01_best.pt \
    --eeg_file   path/to/eeg_array.npy \
    --gsr_file   path/to/gsr_array.npy

# Test with random split checkpoint:
python -m member1_physiological.predict \
    --checkpoint member1_physiological/checkpoints/random_split_best.pt \
    --eeg_file   path/to/eeg_array.npy \
    --gsr_file   path/to/gsr_array.npy
```

**Input array shapes:**
- EEG: `numpy.ndarray` shape `(32, 512)` — 32 channels × 512 samples (4 s at 128 Hz)
- GSR: `numpy.ndarray` shape `(512,)` — resampled to 128 Hz

**Output** (`physiological_prediction_dict`):
```python
{
    "predicted_emotion":   "stress",          # one of: stress, calm, happy, sad, angry
    "confidence":          0.73,              # entropy-based confidence [0–1]
    "class_probabilities": {
        "stress": 0.73, "calm": 0.08, "happy": 0.06, "sad": 0.07, "angry": 0.06
    },
    "signal_quality": {
        "eeg": "good",   # good | degraded | poor
        "gsr": "good"
    }
}
```

---

## Model Soup (Optional — Ensemble of LOSO Checkpoints)

Averages the 32 LOSO fold checkpoints into a single model for deployment:

```bash
python -m member1_physiological.model_soup \
    --checkpoint_dir member1_physiological/checkpoints \
    --output         member1_physiological/checkpoints/model_soup_best.pt
```

Use `model_soup_best.pt` as the checkpoint for the M2 interface integration.

---

## Interface Contract with M2 (Fusion Module)

The `predict_physiological()` function in `predict.py` returns this exact dict:

```python
physiological_prediction_dict = {
    "predicted_emotion":   str,   # "stress" | "calm" | "happy" | "sad" | "angry"
    "confidence":          float, # entropy-based, 0.0–1.0
    "class_probabilities": {      # sums to 1.0
        "stress": float,
        "calm":   float,
        "happy":  float,
        "sad":    float,
        "angry":  float,
    },
    "signal_quality": {
        "eeg": str,               # "good" | "degraded" | "poor"
        "gsr": str,               # "good" | "degraded" | "poor"
    }
}
```

This dict is passed directly to M2's `run_full_pipeline()` as the `physio_prediction` argument.

---

## Common CLI Options

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `data/DEAP` | Path to folder with `s01.dat` … `s32.dat` |
| `--eval_mode` | `loso` | `loso` / `5fold` / `10fold` / `random_split` |
| `--test_subject` | None | LOSO only: run single fold (e.g. `s01`) |
| `--epochs` | 80 | Max epochs per fold |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--patience` | 15 | Early stopping patience (epochs) |
| `--dropout` | 0.3 | Dropout rate |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--no_augment` | False | Disable training augmentation |
| `--checkpoint_dir` | `member1_physiological/checkpoints` | Where to save `.pt` files |
| `--log_dir` | `member1_physiological/logs` | Where to save logs and JSON summaries |
