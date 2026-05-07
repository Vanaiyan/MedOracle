# MedOracle — Project Memory File
> Auto-loaded by Claude at the start of every session in this folder.
> Last updated: May 2026 | Session: Full architecture review + issue resolution

---

## Project Identity

| Field | Detail |
|---|---|
| Title | Multimodal Emotional Recognition System with Explainability |
| Team | MedOracle |
| Institution | Faculty of Information Technology, University of Moratuwa |
| Level | Level 4 Final Year Project (2026) |
| Supervisor | Dr. Firdhous M.F.M. |

### Team Members & Responsibilities

| Member | ID | Module |
|---|---|---|
| Suhira Balarajan | 214206G | Member 1 — Physiological (EEG + GSR) |
| Adshaya Balarajah | 214024V | Member 3 — Explainability + Web App |
| Vanaiyan Kirupagaran | 214215H | Member 2 — Video + Fusion (primary contact) |

---

## System Pipeline (End-to-End)

```
EEG + GSR  ──► [M1: BiAttn Network] ──► physiological_prediction_dict
                                                    │
Facial Video ──────────────────────────────────────►│
                                                    ▼
                                        [M2: Video + Gated Fusion]
                                                    │
                                                    ▼
                                        prediction_output dict
                                                    │
                                                    ▼
                                        [M3: SHAP + FastAPI + React]
                                                    │
                                             shap_output dict
                                          FastAPI (9 endpoints)
                                          React Dashboard + LLM chatbot
```

---

## Shared Emotion Classes (LOCKED — all members must use this)

```python
EMOTION_CLASSES = {
    "stress": 0,
    "calm":   1,
    "happy":  2,
    "sad":    3,
    "angry":  4
}
```
- 5 discrete classes only. No 6th class.
- All model outputs, probability dicts, and SHAP values use these exact keys.

---

## Datasets

### DEAP (Member 1)
- 32 subjects, EEG (128 Hz, 32 channels) + GSR (4 Hz)
- Labels: continuous Valence (1–9), Arousal (1–9), Dominance (1–9)
- Trials: 40 per subject × 60 seconds each

### CREMA-D (Member 2)
- 91 actors, ~7,000 video clips (~30 fps)
- Original 6 labels: ANG, DIS, FEA, HAP, NEU, SAD

---

## RESOLVED ISSUE 1: Label Harmonization

### DEAP → 5-class mapping (V/A/D thresholds)
| Class | Valence | Arousal | Dominance | Citation |
|---|---|---|---|---|
| stress | <5 | ≥5 | <5 | Russell (1980), Mehrabian (1996) |
| calm | ≥5 | <5 | ≥5 | Russell (1980) |
| happy | ≥5 | ≥5 | ≥5 | Russell (1980) |
| sad | <5 | <5 | <5 | Russell (1980) |
| angry | <5 | ≥5 | ≥5 | Russell (1980) |

Threshold: midpoint of 1–9 scale = 5.0. Boundary cases (==5) assigned to positive side.

### CREMA-D → 5-class mapping
| CREMA-D label | → MedOracle class | Justification |
|---|---|---|
| ANG | angry | Direct match |
| HAP | happy | Direct match |
| SAD | sad | Direct match |
| NEU | calm | Neutral ≈ calm low arousal |
| FEA | stress | Fear activates sympathetic system (Kreibig, 2010) |
| DIS | **DROP** | No clean mapping; remove from training |

### Shared enforcement module
- File: `label_harmonization.py` (shared across M1, M2, M3)
- Functions: `map_deap_to_class(v, a, d)`, `map_cremad_to_class(label_str)`
- Both return an integer from EMOTION_CLASSES

---

## RESOLVED ISSUE 2: Signal Quality Flags

Three-tier system: `"good"` / `"degraded"` / `"poor"` — used in `signal_quality` field of `prediction_output`.

### EEG thresholds
| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Amplitude range | ±100 µV | ±100–150 µV | >±150 µV |
| Flat-line channels | 0 | 1–2 | >2 |
| NaN ratio | <1% | 1–5% | >5% |

### GSR thresholds
| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Signal range | 0.5–30 µS | 0.1–0.5 or 30–50 µS | <0.1 or >50 µS |
| SCR peaks (60s window) | ≥2 | 1 | 0 |
| Drift (slope) | <0.1 µS/s | 0.1–0.5 µS/s | >0.5 µS/s |

### Video thresholds
| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Face detection rate | ≥80% frames | 50–80% | <50% |
| Laplacian variance (sharpness) | ≥100 | 50–100 | <50 |
| Face bounding box | >10% frame area | 5–10% | <5% |

**Rule**: worst-performing metric determines the grade (e.g. if any video metric is "poor", video = "poor").
**Citation**: Pech-Pacheco et al. (2000) for Laplacian variance.

---

## RESOLVED ISSUE 3: Video Pipeline (Member 2)

### Architecture
```
Input video → Sample T=16 frames (uniform, temporal stride) →
YOLO face detection → Crop + resize to 224×224 →
ResNet50 (partially fine-tuned) → Frame feature vectors →
BiLSTM → Temporal aggregation → Softmax → P_video[5]
```

### ResNet50 fine-tuning strategy
- Freeze: `layer1`, `layer2` (keep ImageNet low-level features)
- Fine-tune: `layer3`, `layer4`, final FC
- Replace final FC: Linear(2048 → 512) → ReLU → Dropout(0.4) → Linear(512 → 5)

### BiLSTM config
- Hidden size: 256 per direction (512 effective)
- Layers: 2
- Dropout between layers: 0.3
- Input: 2048-dim ResNet50 feature per frame
- Output: hidden state of last timestep → 512-dim → FC → 5 logits

### Dataset split (CREMA-D)
- Actor-independent 5-fold GroupKFold (group = actor ID)
- Ensures no actor appears in both train and test

---

## RESOLVED ISSUE 4: Gated Fusion (Member 2)

### Confidence score (entropy-based — preferred over max-softmax)
```
c_m = 1 - H(P_m) / log(K)
where H(P_m) = -Σ p_k * log(p_k),  K=5
```
Citation: Guo et al. (2017) — "On Calibration of Modern Neural Networks"

### Quality penalty factor (α)
| Signal quality | α value |
|---|---|
| good | 1.0 |
| degraded | 0.5 |
| poor | 0.1 |

### Gate score
```
g_m = c_m × α_m      (for each modality m ∈ {physio, video})
```

### Modality weights (L1 normalisation — NOT softmax)
```
w_m = g_m / Σ_m g_m
```
Rationale: softmax compresses extreme confidence gaps; L1 preserves them.

### Fused prediction
```
P_fused = Σ_m w_m × P_m
predicted_emotion = argmax(P_fused)
```

### Graceful degradation rules
| Condition | Behaviour |
|---|---|
| Both modalities available | Full gated fusion |
| Video poor/missing | w_video → 0, use physio only (uniform if physio also poor) |
| Physio poor/missing | w_physio → 0, use video only |
| Both missing/poor | Uniform distribution, flag as unreliable |

---

## RESOLVED ISSUE 5: Late Fusion Justification

Three-part defence:

1. **Cross-dataset constraint (primary)**: DEAP and CREMA-D have zero overlapping subjects. Joint training is impossible without a shared subject pool. Late fusion is architecturally forced, not a design weakness.

2. **Hybrid framing**: M1 uses *intermediate/attention fusion* within the physiological branch (EEG attends to GSR and vice versa). The system is therefore hybrid — intermediate within physiology, late across modalities.

3. **Literature support**: Late fusion outperforms early fusion when modalities have different temporal resolutions, different noise characteristics, and different training sets (Poria et al., 2017; Baltrusaitis et al., 2019).

---

## RESOLVED ISSUE 6: Temporal Synchronization

### Window definition
- **Trial window**: 4 seconds (justified by EEG alpha-band dynamics, literature-standard)
- **EEG**: 128 Hz → 512 samples per 4s window
- **GSR**: 4 Hz → 16 samples per 4s window → resampled to 128 Hz (512 samples) using `scipy.signal.resample_poly` (polyphase, anti-aliasing)
- **Video**: T=16 frames sampled uniformly across the 4s window

### SynchronizedInput dataclass
```python
@dataclass
class SynchronizedInput:
    eeg:   np.ndarray   # shape (32, 512)
    gsr:   np.ndarray   # shape (512,) after resampling
    video: np.ndarray   # shape (16, 224, 224, 3)
    trial_id: str
    subject_id: str

    def validate(self):
        assert self.eeg.shape == (32, 512)
        assert self.gsr.shape == (512,)
        assert self.video.shape == (16, 224, 224, 3)
```

---

## RESOLVED ISSUE 7: Dataset Size & Cross-Validation

### DEAP (Member 1)
- **Strategy**: LOSO — Leave-One-Subject-Out (32 folds)
- **Class imbalance**: Use class weights (sklearn `compute_class_weight`) — no SMOTE on EEG
- **Report**: Mean ± Std macro-F1 across 32 folds

### CREMA-D (Member 2)
- **Strategy**: Actor-independent 5-fold GroupKFold
- **Class imbalance**: Weighted cross-entropy loss
- **Report**: Mean ± Std macro-F1 across 5 folds

---

## RESOLVED ISSUE 8: Evaluation Plan & Ablation Study

### Primary metric: Macro-averaged F1 (not accuracy)
Rationale: class imbalance present in both datasets.

### 5-condition ablation table
| Condition | Physio | Video | Fusion |
|---|---|---|---|
| C1 | ✓ (M1 only) | ✗ | — |
| C2 | ✗ | ✓ (M2 only) | — |
| C3 | ✓ | ✓ | Equal weights (0.5/0.5) |
| C4 | ✓ | ✓ | Confidence-weighted (no quality penalty) |
| **C5** | ✓ | ✓ | **Full method (confidence + quality gating)** |

### 3 robustness conditions
| Condition | Scenario |
|---|---|
| C6 | Video quality = poor (α=0.1) |
| C7 | Video completely missing (uniform weights) |
| C8 | Physio quality = poor (α=0.1) |

### Statistical significance
- Wilcoxon signed-rank test, one-tailed, α=0.05
- Compare C5 vs C1, C5 vs C2, C5 vs C3

### Literature baselines to beat
- Siddharth et al. (2019) — EEG + peripheral physio — ~73% accuracy
- Zhang et al. (2020) — video-only — ~65% accuracy on similar classes

---

## Interface Contracts (Between Members)

### M1 → M2: `physiological_prediction_dict`
```python
{
    "predicted_emotion":    str,          # e.g. "stress"
    "confidence":           float,        # entropy-based, 0–1
    "class_probabilities":  dict,         # {"stress":0.6, "calm":0.1, ...}
    "signal_quality": {
        "eeg": str,                       # "good"|"degraded"|"poor"
        "gsr": str
    }
}
```

### M2 → M3: `prediction_output`
```python
{
    "predicted_emotion":       str,
    "confidence":              float,
    "class_probabilities":     {"stress":float, "calm":float, "happy":float,
                                "sad":float, "angry":float},
    "modality_weights":        {"physio": float, "video": float},
    "signal_quality":          {"eeg":str, "gsr":str, "video":str},
    "per_modality_predictions":{
        "physio": {"predicted_emotion":str, "confidence":float,
                   "class_probabilities": dict},
        "video":  {"predicted_emotion":str, "confidence":float,
                   "class_probabilities": dict}
    }
}
```

### M3 → API: `shap_output` (enriched)
Adds `shap_values`, `feature_names`, `faithfulness_score`, `reliability_flags` to `prediction_output`.

---

## Member 1 — Suhira (Physiological Module)

### Architecture: Bidirectional Cross-Modal Attention Network
- Input: EEG (32ch × 512 samples) + GSR (512 samples after resampling)
- EEG attends to GSR, GSR attends to EEG (bidirectional cross-attention)
- Joint classification head → 5-class softmax
- Export function: `predict_physiological(eeg_data, gsr_data) -> dict`

### Key steps
1. DEAP dataset loading + label harmonization (V/A/D thresholds)
2. EEG preprocessing: bandpass filter (0.5–45 Hz), ICA artifact removal, epoch into 4s trials
3. GSR preprocessing: lowpass filter, polyphase resample to 128 Hz
4. Feature extraction (time-domain + frequency-domain)
5. Bidirectional cross-modal attention implementation
6. LOSO cross-validation (32 folds) with class weighting
7. Export `predict_physiological()` with signal quality flags
8. Validate interface contract with M2

---

## Member 2 — Vanaiyan (Video + Fusion Module)

### Architecture
- YOLO face detection → ResNet50 (partial fine-tune) → BiLSTM → video prediction
- Gated fusion: entropy confidence × quality penalty → L1-normalised weights → weighted sum
- Graceful degradation for missing/poor modalities
- Export: `run_full_pipeline(eeg, gsr, video) -> prediction_output`

### Key steps
1. CREMA-D loading + label harmonization (FEA→stress, NEU→calm, DIS→drop)
2. Face detection pipeline (YOLO)
3. ResNet50 fine-tuning (freeze layer1+2, train layer3+4+FC)
4. BiLSTM temporal modelling
5. Actor-independent 5-fold GroupKFold evaluation
6. Implement gated fusion module
7. Integrate M1 output via interface contract
8. Signal quality assessment for video modality
9. Graceful degradation logic
10. Export `run_full_pipeline()` + validate full `prediction_output` dict
11. Ablation study (C1–C5) + robustness tests (C6–C8)

---

## Member 3 — Adshaya (Explainability + Web App)

### SHAP Layer
- Kernel SHAP on full `prediction_output`
- Faithfulness metric: perturb features → measure prediction change
- Signal reliability qualification from `signal_quality` flags

### Web Application
- **Backend**: FastAPI, 9 endpoints, JWT auth, PostgreSQL/SQLite
- **Database**: 4 tables (users, sessions, predictions, chat_history)
- **Frontend**: React, Recharts (emotion trend LineChart, SHAP BarChart),
  session history list, chatbot panel
- **LLM chatbot**: SHAP-to-prompt bridging → Claude/GPT-4 → plain-English explanations

### 9 FastAPI endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | /auth/register | User registration |
| POST | /auth/login | JWT token issue |
| POST | /predict | Run full pipeline |
| GET | /explain/{session_id} | SHAP explanation |
| GET | /sessions | List user sessions |
| GET | /sessions/{id} | Session detail |
| POST | /chat | LLM chatbot message |
| GET | /chat/history/{session_id} | Chat history |
| GET | /dashboard/summary | Aggregated stats |

---

## Technology Stack

| Layer | Technology |
|---|---|
| EEG/GSR processing | Python, MNE, scipy, numpy |
| Deep learning | PyTorch |
| Face detection | YOLOv8 (ultralytics) |
| SHAP | shap (Kernel SHAP) |
| Backend | FastAPI, Uvicorn |
| Auth | JWT (python-jose) |
| Database | PostgreSQL (prod), SQLite (dev) |
| Frontend | React, Recharts, Axios |
| Validation | Pydantic |

---

## Key Academic References

1. Russell (1980) — Circumplex model of affect — *JPSP* 39(6)
2. Mehrabian (1996) — PAD emotional state model — *J. Nonverbal Behavior*
3. Kreibig (2010) — Fear → sympathetic activation — *Biological Psychology*
4. Koelstra et al. (2012) — DEAP dataset — *IEEE Trans. Affective Computing*
5. Cao et al. (2014) — CREMA-D dataset — *IEEE Trans. Affective Computing*
6. Guo et al. (2017) — Neural network calibration — *ICML*
7. Pech-Pacheco et al. (2000) — Laplacian variance sharpness — *ICPR*
8. Poria et al. (2017) — Late vs early fusion survey — *Information Fusion*
9. Baltrusaitis et al. (2019) — Multimodal ML survey — *IEEE TPAMI*
10. Siddharth et al. (2019) — EEG + peripheral physio emotion — *Frontiers in Neuroscience*
11. Hochreiter & Schmidhuber (1997) — LSTM — *Neural Computation*
12. He et al. (2016) — ResNet — *CVPR*

---

## Project Report

Full comprehensive technical report generated: `MedOracle_Project_Report.pdf`
Location: `/Users/vanaiyan/Documents/Claude/Projects/FYP/MedOracle_Project_Report.pdf`
Content: Architecture, math formulations, implementation plans, 20-week timeline,
step-by-step guides per member, risk register, evaluation plan.

---

## Status Tracking

| Phase | Owner | Status |
|---|---|---|
| Architecture design & issue resolution | All | ✅ Complete |
| Label harmonization spec | All | ✅ Decided |
| Signal quality thresholds | All | ✅ Decided |
| Interface contracts (M1→M2, M2→M3) | All | ✅ Decided |
| Fusion math (gated, L1, entropy conf) | M2 | ✅ Decided |
| Evaluation plan + ablation study | All | ✅ Decided |
| Full project report PDF | All | ✅ Generated |
| M1 implementation (physio module) | Suhira | 🔲 Not started |
| M2 implementation (video + fusion) | Vanaiyan | 🔲 Not started |
| M3 implementation (SHAP + web app) | Adshaya | 🔲 Not started |
| Integration testing | All | 🔲 Not started |
| Ablation study experiments | All | 🔲 Not started |
| Final report writing | All | 🔲 Not started |
