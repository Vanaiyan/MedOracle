<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 6 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout, EXCEPT the code listings noted below.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 6" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 6" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "6.2 Module 1 ..."): 12pt, bold.
- Subsection headings (e.g. "6.2.1 Data Loading..."): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- CODE LISTINGS: every fenced code block (```...```) below is a "Listing"
  — format it in a monospaced font (Consolas or Courier New), 10pt, SINGLE
  line spacing (not 1.5), left-aligned, ideally with light grey shading or
  a thin border to visually distinguish it from body text. Put the caption
  "Listing 6.x: <short description>" in bold directly ABOVE each code
  block (the description is given in the text immediately preceding the
  block in this document — use that sentence to write the caption). These
  listings do not need a separate "List of Listings" front-matter page.
- The five figures and four tables below must be inserted exactly as in
  earlier chapters: figures as actual images with bold captions below
  ("Figure 6.x: ..."), tables as real Word tables with bold header rows
  and bold captions above ("Table 6.x: ...").
- Figure image files (insert at the paths given, in order of appearance):
    Figure 6.1: member1_physiological/visualization/viz_deap_loader_output.png
    Figure 6.2: member1_physiological/visualization/viz_signal_quality_output.png
    Figure 6.3: member1_physiological/baseline/Outputs of baselines/SVM output.png
    Figure 6.4: member1_physiological/baseline/Outputs of baselines/Random Forest output.png
    Figure 6.5: member1_physiological/baseline/Outputs of baselines/MLP output.png
    Figure 6.6: docs/report_assets/fig6_6_discriminative_lr.png
- If a Chapter 6 already exists in the document (an earlier, shorter
  draft), REPLACE its content entirely with this enhanced version.
- Do not alter the numbering, content, or formatting of any other chapter.
- This chapter uses no citation markers ([n]) — implementation detail is
  not literature-cited — so do not add any.

=========================================================================
-->

# Chapter 6

## Implementation

### 6.1 Introduction

This chapter describes the implementation of each module, following the design presented in Chapter 5. Source code is organised into three top-level packages (member1_physiological, member2_video_fusion, member3_explainability) plus a shared package (shared/) holding the taxonomy, label harmonisation and interface-contract code common to all three. Where a design decision is best illustrated by the actual code rather than by prose alone, a short representative listing is included.

### 6.2 Module 1 — Physiological Emotion Recognition Implementation

#### 6.2.1 Data Loading and Preprocessing

The DEAP loader (preprocessing/deap_loader.py) reads each subject's pickled .dat file, trims the first 3 seconds (384 samples) of baseline recording from each 60-second trial, and slides 15 non-overlapping 4-second (512-sample) windows across the remainder. EEG channels 0-31 and the GSR channel (index 36) are extracted per window, and the trial's valence/arousal/dominance rating is converted to one of the five shared classes via the label-harmonisation mapping described in Section 5.3. Figure 6.1 shows the resulting per-subject window counts and class distribution produced by this loader.

**[INSERT IMAGE HERE: `member1_physiological/visualization/viz_deap_loader_output.png`]**

**Figure 6.1: DEAP dataset structure and per-subject class distribution (preprocessing visualisation)**

As Table 7.1 (Section 7.2) will show in more detail, this harmonisation produces a severely imbalanced class distribution — most notably only 19 "happy" trials across all 32 subjects — which later proved central to explaining the physiological module's evaluation results.

#### 6.2.2 Signal Quality Assessment

A dedicated signal_quality module grades each EEG and GSR window against the thresholds in Table 5.1. Figure 6.2 shows example output of this grading process across a sample of windows.

**[INSERT IMAGE HERE: `member1_physiological/visualization/viz_signal_quality_output.png`]**

**Figure 6.2: EEG / GSR / Video signal-quality grading visualisation**

#### 6.2.3 Network Architecture

The EEG encoder applies a depthwise temporal convolution (kernel size 25, approximately 200ms at 128Hz, capturing alpha/beta oscillations) followed by a pointwise convolution mixing across channels, then two further downsampling convolutional blocks reducing the 512-sample sequence to 32 timesteps at a 128-dimensional embedding, with sinusoidal positional encoding added before the attention layer. Listing 6.1 shows the depthwise/pointwise convolution pair at the heart of this encoder.

```python
# member1_physiological/models/encoders.py — EEGEncoder (excerpt)
self.depthwise = nn.Conv1d(
    in_channels=n_channels, out_channels=n_channels,
    kernel_size=25, padding=12, groups=n_channels, bias=False,
)
self.pointwise = nn.Conv1d(
    in_channels=n_channels, out_channels=32, kernel_size=1, bias=False,
)
```

**Listing 6.1: Depthwise + pointwise convolution pair in the EEG encoder**

The GSR encoder follows an analogous, single-channel convolutional structure. A bidirectional cross-modal attention layer then allows the EEG sequence to attend over the GSR sequence and vice versa, before the attended representations are pooled and passed to a shared five-class classification head.

#### 6.2.4 Baseline Comparisons

Three classical baselines — SVM, Random Forest and a Multi-Layer Perceptron, each using a 165-dimensional hand-crafted feature vector (32 channels × 5 band-power features + 5 GSR statistics) — were implemented and evaluated under the same 32-fold Leave-One-Subject-Out protocol as the neural network, in order to establish whether the neural architecture itself, rather than the underlying data, was limiting performance. Figures 6.3-6.5 show the resulting outputs.

**[INSERT IMAGE HERE: `member1_physiological/baseline/Outputs of baselines/SVM output.png`]**

**Figure 6.3: SVM baseline — subject-independent (LOSO) results**

**[INSERT IMAGE HERE: `member1_physiological/baseline/Outputs of baselines/Random Forest output.png`]**

**Figure 6.4: Random Forest baseline — subject-independent (LOSO) results**

**[INSERT IMAGE HERE: `member1_physiological/baseline/Outputs of baselines/MLP output.png`]**

**Figure 6.5: MLP baseline — subject-independent (LOSO) results**

All three classical baselines converge to a similar macro-F1 range as the neural network under LOSO, corroborating the diagnosis presented in Section 7.2: the limitation is primarily in the data (class imbalance and cross-subject variability) rather than in any single model architecture. As part of this same diagnostic effort, a Differential Entropy feature extractor was implemented as an alternative representation to the raw time-domain signal, shown in Listing 6.2.

```python
# member1_physiological/preprocessing/de_features.py (excerpt)
def eeg_de(eeg: np.ndarray, fs: int = FS) -> np.ndarray:
    nperseg = min(256, eeg.shape[-1])
    freqs, psd = welch(eeg, fs=fs, nperseg=nperseg, axis=-1)
    out = np.empty((eeg.shape[0], N_BANDS), dtype=np.float32)
    for i, (_, lo, hi) in enumerate(BANDS):
        idx = (freqs >= lo) & (freqs < hi)
        band_power = psd[:, idx].mean(axis=1) if idx.any() else 0.0
        out[:, i] = 0.5 * np.log(2.0 * np.pi * np.e * (band_power + _EPS))
    return out
```

**Listing 6.2: Differential Entropy feature extraction (log band power per channel per frequency band)**

#### 6.2.5 Interface Export

**Table 6.1: M1 → M2 interface contract (physiological_prediction_dict)**

| Field | Type | Description |
|---|---|---|
| predicted_emotion | str | One of the five shared emotion labels |
| confidence | float in [0,1] | Entropy-based confidence, 1 − H(P)/log(K) |
| class_probabilities | dict[str,float] | Full 5-class probability distribution |
| signal_quality.eeg | str | "good" \| "degraded" \| "poor" |
| signal_quality.gsr | str | "good" \| "degraded" \| "poor" |

The predict.py module exposes PhysiologicalPredictor.predict(eeg, gsr), returning a dictionary matching Table 6.1, validated by make_physiological_prediction() in the shared contracts module.

### 6.3 Module 2 — Video-Based Emotion Recognition Implementation

#### 6.3.1 Preprocessing and Training

Sixteen frames per clip are uniformly sampled and resized directly to 224×224 without cropping to a detected face region. This choice followed a controlled comparison during development: an earlier implementation cropped each frame to its YOLO-detected face region (matching common practice), but a held-out evaluation on CREMA-D clips showed this configuration achieving only 0.56 macro-F1, versus 0.70 for the same model applied to full, uncropped frames — likely because cropping discards contextual cues (head pose, shoulder movement) that the temporal model otherwise exploits. YOLOv8 face detection is retained in the pipeline, but is now used exclusively for signal-quality grading (Section 5.4) rather than for cropping the recognition input.

The ResNet50 backbone is partially fine-tuned: the first two residual blocks (layer1, layer2) are frozen to retain general low-level ImageNet features, while the later blocks (layer3, layer4) and a replaced classification head (Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→5)) are fine-tuned. Table 6.4 summarises the full set of training hyperparameters, adopted after the discriminative-learning-rate fix described in Section 4.3.2 resolved an earlier overfitting/underfitting instability.

**Table 6.4: Video model training hyperparameters**

| Hyperparameter | Value |
|---|---|
| Backbone learning rate (layer3, layer4) | 1e-5 |
| Head + BiLSTM learning rate | 5e-4 |
| Optimizer | Adam |
| LR schedule | Cosine annealing with warmup |
| Batch normalisation | Frozen (pretrained statistics retained) |
| Augmentation | Cutout |
| Cross-validation | Actor-independent StratifiedGroupKFold, 5 folds |
| BiLSTM configuration | 2 layers, 256 hidden units per direction, dropout 0.3 |
| Frames sampled per clip | 16 (uniform sampling) |
| Input resolution | 224×224, full frame (no face crop) |

Training was carried out on Kaggle across multiple 12-hour sessions with automatic checkpoint upload/resume between sessions, as motivated in Section 3.2. Figure 6.6 visualises the train/validation gap that motivated the discriminative-learning-rate fix in the first place.

**[INSERT IMAGE HERE: `docs/report_assets/fig6_6_discriminative_lr.png`]**

**Figure 6.6: Effect of fine-tuning strategy on the train/validation macro-F1 gap**

#### 6.3.2 Gated Fusion Implementation

The canonical fusion implementation (member2_video_fusion/fusion.py) implements exactly the formulas given in Section 5.6. Listing 6.3 shows the entropy-confidence and quality-penalty functions at its core.

```python
# member2_video_fusion/fusion.py (excerpt)
QUALITY_ALPHA = {"good": 1.0, "degraded": 0.5, "poor": 0.1}

def entropy_confidence(probs: dict) -> float:
    K = len(probs)
    H = -sum(p * math.log(p + _EPS) for p in probs.values())
    return max(0.0, min(1.0, 1 - H / math.log(K)))

def quality_alpha(quality: str) -> float:
    return QUALITY_ALPHA[quality]
```

**Listing 6.3: Entropy-based confidence and signal-quality penalty functions**

This implementation is validated against Member 3's earlier, independently-implemented reference fusion function (conflict/gate.py) with a parametrised unit test covering all combinations of signal-quality grade for both modalities, asserting that both implementations produce identical modality weights and fused probabilities to within 1e-9 absolute tolerance. This allows gate.py and other duplicate fusion logic elsewhere in the codebase to eventually be collapsed to import the single canonical implementation.

#### 6.3.3 Full Pipeline

member2_video_fusion/pipeline.py exposes run_full_pipeline(eeg, gsr, video_path), shown in Listing 6.4, which lazily loads Member 1's PhysiologicalPredictor, runs Member 2's video model, and combines both outputs via gated_fusion().

```python
# member2_video_fusion/pipeline.py (signature)
def run_full_pipeline(
    eeg: Optional[np.ndarray] = None,
    gsr: Optional[np.ndarray] = None,
    video_path: Optional[Union[str, Path]] = None,
    video_frames: Optional[np.ndarray] = None,
    video_quality: str = "good",
) -> dict:
    ...
```

**Listing 6.4: Full-pipeline entry point — any single modality may be supplied alone**

Any single modality may be omitted: if only eeg/gsr are supplied the function returns a physio-only prediction; if only a video path is supplied it returns a video-only prediction (still routed through the same gated-fusion function so the output schema and graceful-degradation behaviour are identical regardless of which modality is present); and a SynchronizedInput convenience wrapper is provided for the case where all three raw signals are available together.

**Table 6.2: M2 → M3 interface contract (prediction_output)**

| Field | Type | Description |
|---|---|---|
| predicted_emotion | str | Fused emotion label |
| confidence | float | Fused entropy-based confidence |
| class_probabilities | dict[str,float] | Fused 5-class probability distribution |
| modality_weights | dict | {"physio": w1, "video": w2}, sums to 1.0 |
| signal_quality | dict | {eeg, gsr, video} quality grades |
| per_modality_predictions | dict | Raw physio and video predictions before fusion |

#### 6.3.4 Cross-Dataset Evaluation Set Construction

Because DEAP and CREMA-D share no common subjects, a paired evaluation set was constructed by pairing, for each of the five emotion classes, a real DEAP EEG/GSR window with a real CREMA-D video clip carrying the same emotion label — the standard methodology for evaluating cross-dataset late fusion when no single dataset provides every modality (Section 2.4). build_synced_dataset.py constructs ten such "virtual subjects" (two per emotion) for the web-application demonstration, and the ablation study (Section 6.3.5) uses a larger, randomly re-paired version of the same construction (20 samples per emotion, 50 random pairings) to obtain statistically meaningful macro-F1 estimates with a measurable variance.

#### 6.3.5 Ablation Study Implementation

ablation_study.py precomputes each modality's prediction once per held-out sample (20 DEAP windows and 20 held-out CREMA-D clips per emotion, drawn from actors never used in video training), then evaluates eight fusion conditions over 50 random re-pairings of the precomputed predictions, reporting mean and standard deviation of macro-F1 per condition and a one-tailed Wilcoxon signed-rank test (reusing Member 1's wilcoxon_test utility) comparing the full method (C5) against each single-modality and naive-fusion baseline. Full results are presented in Section 7.4.

### 6.4 Module 3 — Explainability and Web Application Implementation

#### 6.4.1 SHAP Explainability

shap_output_builder.py wraps the prediction_output in a Kernel SHAP explanation, computing per-feature attribution values, a faithfulness score (obtained by perturbing individual features and measuring the resulting change in the fused prediction), and signal-reliability flags derived from the signal_quality fields.

#### 6.4.2 Backend

**Table 6.3: FastAPI endpoint summary**

| Method | Path | Purpose |
|---|---|---|
| POST | /auth/register | User registration |
| POST | /auth/login | JWT token issue |
| POST | /predict | Run full pipeline (generic entry point) |
| POST | /predict/video | Video-only prediction |
| POST | /predict/multimodal | Video + EEG + GSR file upload → fused prediction |
| GET | /explain/{session_id} | SHAP explanation for a session |
| GET | /sessions | List user sessions |
| POST | /chat | LLM chatbot message |
| GET | /dashboard/summary | Aggregated dashboard statistics |

The /predict/multimodal endpoint, integrated during this project's system-integration phase, accepts a required video file together with optional EEG and GSR files, as shown in Listing 6.5.

```python
# member3_explainability/backend/routers/predict_router.py (signature)
@router.post("/predict/multimodal", response_model=PredictResponse)
async def predict_multimodal(
    file: UploadFile = File(..., description="Video clip (required)"),
    eeg:  Optional[UploadFile] = File(None, description="EEG window, shape (32,512)"),
    gsr:  Optional[UploadFile] = File(None, description="GSR window, shape (512,)"),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    ...
```

**Listing 6.5: The /predict/multimodal endpoint signature**

EEG/GSR files (.npy or .csv) are validated against the required (32, 512) and (512,) shapes respectively before being passed to run_full_pipeline; if EEG/GSR are omitted the endpoint falls back to video-only prediction via the same underlying pipeline, so a single code path serves both the video-only and full-multimodal use cases.

#### 6.4.3 Frontend

The React dashboard was extended with EEG and GSR file inputs alongside the existing video drop-zone, and two prediction actions — "Video only" and "Run Multimodal Fusion" — the latter enabled only once all three files are selected. The existing SHAP bar chart, emotion-trend chart, and modality-conflict explanation panel render directly from whichever prediction_output the selected action produces, requiring no separate code path for the multimodal case.

### 6.5 Testing

The project maintains 82 automated tests, all passing, organised as follows: 59 tests over the shared label-harmonisation module (covering DEAP threshold boundaries, CREMA-D label mapping including case-insensitivity, and batch-processing helpers), 18 tests over the canonical fusion module (including 12 parametrised tests asserting numerical equivalence with the independent reference implementation across every signal-quality combination, plus degradation and both-modality-missing edge cases), and 5 tests over the full pipeline (physio-only, video-only, and full multimodal branches, plus input-validation tests that reject an incomplete eeg/gsr pair). A full in-process HTTP test additionally exercises the backend end-to-end: registration, login, a real multimodal prediction request (returning HTTP 200 with genuinely non-zero weight on both modalities and a populated SHAP explanation), a video-only request, and a malformed-EEG-shape request (correctly rejected with HTTP 422). The React frontend was separately verified to build without errors after the EEG/GSR upload controls were added.

### 6.6 Summary

This chapter described the implementation of each module and the integration work — the canonical fusion module, the full pipeline, the cross-dataset evaluation set, and the multimodal web endpoint — that ties them together, illustrated with representative code listings at each key design point. The next chapter presents the evaluation and discussion of the complete system, including an honest account of the physiological module's limitations.
