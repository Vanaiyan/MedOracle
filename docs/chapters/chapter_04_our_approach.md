<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 4 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 4" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 4" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "4.2 Member 1 ..."): 12pt, bold.
- Subsection headings (e.g. "4.2.1 Input"): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- There are no tables in this chapter. Three figures are referenced as
  image file paths in this document — insert each actual image at its
  path, centered, with the caption "Figure 4.x: <caption text>" centered
  directly below it in bold:
    Figure 4.1: docs/report_assets/fig4_1_m1_pipeline.png
    Figure 4.2: docs/report_assets/fig4_2_m2_pipeline.png
    Figure 4.3: docs/report_assets/fig4_3_m3_pipeline.png
- If a Chapter 4 already exists in the document (an earlier, shorter
  draft), REPLACE its content entirely with this enhanced version; do not
  merge or keep both.
- Do not alter the numbering, content, or formatting of any other chapter.
- Keep every citation marker exactly as written (e.g. "[6]") — these numbers
  are fixed by the master References list (alphabetical by first author)
  and must NOT be renumbered or have new sources inserted into the middle
  of the list, since Chapters 1 and 2 already depend on this exact
  numbering:
  [1] Baltrusaitis et al. (2019)   [2] Cao et al. (2014)
  [3] Guo et al. (2017)            [4] He et al. (2016)
  [5] Hochreiter & Schmidhuber (1997)  [6] Koelstra et al. (2012)
  [7] Kreibig (2010)               [8] Mehrabian (1996)
  [9] Pech-Pacheco et al. (2000)   [10] Poria et al. (2017)
  [11] Russell (1980)              [12] Siddharth et al. (2019)
  [13] Zhang et al. (2020)
  This chapter cites ONLY these 13 sources.

=========================================================================
-->

# Chapter 4

## Our Approach

### 4.1 Introduction

This chapter describes each member's individual approach in detail, structured consistently as Input, Process, Output and Evaluation for each of the three modules, so that the three approaches can be compared directly despite operating on entirely different signal types. All three modules are bound by the shared five-class emotion taxonomy — stress, calm, happy, sad, angry — and by the interface contracts defined in shared/data_contracts.py (detailed fully in Chapter 5), which is what allows the Input/Process/Output boundary of one member's module to be developed and tested independently of how any other member implements theirs.

### 4.2 Member 1 — Physiological Emotion Recognition

#### 4.2.1 Input

The module's raw input is a 4-second window of 32-channel EEG (128Hz, 512 samples per window) and single-channel GSR (originally recorded at 4Hz and resampled to 128Hz via polyphase resampling to match the EEG sampling rate), drawn from the DEAP dataset [6]. Each DEAP trial is 60 seconds long, with a 3-second baseline recording trimmed from the start before the remaining 57 seconds are slid into fifteen non-overlapping 4-second windows; every window inherits the label of its parent trial, which is itself derived from the subject's self-reported valence, arousal and dominance rating for that trial. Each window is additionally graded for signal quality — separately for EEG and GSR — against thresholds covering amplitude range, flat-line channel count and NaN ratio for EEG, and signal range, skin-conductance-response peak count and drift for GSR (Chapter 5), so that a downstream fusion decision can distinguish a genuinely poor prediction from a poor input.

#### 4.2.2 Process

Each DEAP trial's valence, arousal and dominance rating (originally on a continuous 1-9 scale) is first mapped onto the shared five-class taxonomy using thresholds at the scale midpoint, following Russell's circumplex model [11] and Mehrabian's PAD model [8]: for example, low valence with high arousal and low dominance maps to stress, a mapping consistent with Kreibig's finding that fear-related states activate the sympathetic nervous system in a manner comparable to stress [7], which is separately relevant to how Member 2's CREMA-D fear label is harmonised into the same taxonomy (Chapter 5).

The EEG window is passed through a dedicated encoder: a depthwise temporal convolution (kernel size 25, corresponding to roughly 200 milliseconds at 128Hz, chosen to capture alpha- and beta-band oscillations relevant to emotional arousal) applied independently per channel, followed by a pointwise convolution that mixes information across the 32 channels, and two further strided convolutional blocks that progressively downsample the 512-sample sequence to a 32-timestep, 128-dimensional embedding, with sinusoidal positional encoding added before the next stage. The GSR window is passed through an analogous, single-channel convolutional encoder producing a comparably-shaped embedding. The two embeddings are then passed into a bidirectional cross-modal attention layer, in which the EEG sequence attends over the GSR sequence and the GSR sequence attends over the EEG sequence, before the two attended representations are pooled and concatenated and passed to a shared classification head producing a five-class softmax distribution. Figure 4.1 shows this pipeline end to end.

**[INSERT IMAGE HERE: `docs/report_assets/fig4_1_m1_pipeline.png`]**

**Figure 4.1: Member 1 physiological emotion recognition pipeline**

Training and evaluation were carried out under two distinct protocols, deliberately kept separate throughout this project. An initial random-split protocol, in which 4-second windows are shuffled and split into training and test sets without regard to which subject or trial they originated from, was used only to obtain an early, working checkpoint so that Member 2's fusion pipeline could be developed and integration-tested before the more expensive, honest evaluation was complete. The full evaluation protocol is Leave-One-Subject-Out (LOSO) cross-validation across all 32 DEAP subjects: in each of the 32 folds, one subject is held out entirely and the model is trained from scratch on the remaining 31, so that no window from the held-out subject's data — and, because each subject's trials are unique to them, no near-duplicate window either — is ever seen during that fold's training.

#### 4.2.3 Output

The module exposes a single function, PhysiologicalPredictor.predict(eeg, gsr), returning a physiological_prediction_dict containing four fields: predicted_emotion (one of the five shared class labels), confidence (an entropy-based score in [0, 1], computed identically to the confidence term used in Member 2's fusion mechanism, so that both modalities' confidence is directly comparable), class_probabilities (the full five-class softmax distribution), and signal_quality (a nested dictionary with independent "good"/"degraded"/"poor" grades for the EEG and GSR inputs). This dictionary is validated against the shared interface contract before being passed to Member 2's fusion module (Chapter 5), so a malformed or out-of-range value is rejected at the module boundary rather than silently propagating downstream.

#### 4.2.4 Evaluation

Under the random-split protocol, the module achieves a macro-F1 of 0.4171 (accuracy approximately 42%). Under the honest, subject-independent LOSO protocol, the same architecture achieves a macro-F1 of 0.179 ± 0.041 across the 32 folds, and a "model soup" formed by averaging all 32 fold checkpoints together performs almost identically (macro-F1 0.179), indicating this is a stable ceiling rather than fold-specific noise. Three classical baselines — SVM, Random Forest and a Multi-Layer Perceptron, each trained on a 165-dimensional hand-crafted feature vector — were evaluated under the identical LOSO protocol specifically to determine whether this ceiling reflected a limitation of the neural architecture or of the underlying data; all three converged to a comparable macro-F1 range. The gap between the two protocols, and the diagnostic experiments undertaken to explain the LOSO result — comparing feature representations, auditing GSR data quality, and quantifying the effect of coarser label taxonomies — are presented in full in Section 7.2, together with this project's position on why the honest, lower figure is the one reported as the module's true performance.

### 4.3 Member 2 — Video-Based Emotion Recognition and Multimodal Fusion

#### 4.3.1 Input

The video module's raw input is a short video clip from the CREMA-D dataset [2] (with RAVDESS clips additionally used during model development to increase training diversity), each labelled with one of six original emotion categories, five of which — anger, happy, sad, fear (harmonised to stress), and neutral (harmonised to calm) — map onto the shared five-class taxonomy, with disgust dropped as having no clean correspondence (Chapter 5). Sixteen frames are sampled uniformly across each clip to represent its temporal span within a single forward pass of the recognition network.

#### 4.3.2 Process

Member 2's contribution has three parts: the video recognition model itself, the gated fusion mechanism that combines its output with Member 1's, and the full pipeline and evaluation methodology that ties the whole system together.

For video recognition, the sixteen sampled frames are resized directly to 224×224 pixels without cropping to a detected face region — a deliberate choice following a controlled comparison made during development, in which cropping to a YOLOv8-detected face region was found to reduce held-out macro-F1 from approximately 0.70 to 0.56, likely because cropping discards contextual cues (head pose, shoulder movement) that the temporal model otherwise exploits. YOLOv8 face detection is retained in the pipeline, but is used exclusively to grade video signal quality — face-detection rate, Laplacian-variance sharpness [9], and face bounding-box area — rather than to crop the recognition input. The resized frames are passed through a ResNet50 [4] backbone, partially fine-tuned: the first two residual blocks are frozen to retain general low-level features learned from large-scale image pretraining, while the later two blocks and a replaced classification head are fine-tuned using a discriminative learning rate (1e-5 for the fine-tuned backbone layers, 5e-4 for the new head), a configuration adopted after an initial fully-frozen-backbone configuration underfit (train and validation macro-F1 both approximately 0.30) and a fully-unfrozen configuration overfit (train macro-F1 0.88 against validation 0.63). The resulting per-frame feature sequence is passed to a two-layer bidirectional LSTM [5] (256 hidden units per direction, inter-layer dropout 0.3) before a final softmax layer produces a five-class prediction. Training used actor-independent StratifiedGroupKFold cross-validation (ensuring no actor's clips appear in both a training and validation fold), cosine-annealed learning-rate scheduling with warmup, cutout augmentation applied to input frames, and frozen batch-normalisation statistics, and was conducted across multiple Kaggle training sessions with automatic checkpoint persistence and resume between sessions.

Figure 4.2 shows this recognition pipeline together with where the fusion step (described next) sits relative to it.

**[INSERT IMAGE HERE: `docs/report_assets/fig4_2_m2_pipeline.png`]**

**Figure 4.2: Member 2 video recognition and gated fusion pipeline**

For fusion, each modality's prediction is combined using a quality-aware gated mechanism (the full mathematical formulation is presented in Chapter 5): an entropy-based confidence score is computed for each modality's output distribution, multiplied by a signal-quality penalty (1.0 for "good", 0.5 for "degraded", 0.1 for "poor"), and the resulting per-modality gate scores are L1-normalised into fusion weights, rather than passed through a softmax, which would compress large confidence gaps. The implementation (fusion.py) was validated against an independent reference fusion function that Member 3 had implemented earlier for a different purpose, using a parametrised unit test that asserts both implementations produce numerically identical modality weights and fused probabilities (within 1e-9 absolute tolerance) across every combination of signal-quality grade for both modalities.

For the full pipeline and evaluation, run_full_pipeline(eeg, gsr, video_path) invokes Member 1's PhysiologicalPredictor, the video recognition model, and the gated fusion function in sequence, accepting any single modality alone (in which case the missing modality's weight is driven to zero by the fusion formula rather than requiring a separate code path). Because DEAP and CREMA-D share no subjects, a cross-dataset evaluation set was constructed by pairing, for each emotion class, a real DEAP physiological window with a real CREMA-D video clip carrying the same label — the standard methodology for evaluating cross-dataset late fusion [1, 10] — and this construction underlies both the web-application demonstration dataset and the eight-condition ablation study (C1-C8) with Wilcoxon significance testing presented in Chapter 7.

#### 4.3.3 Output

The video model alone produces a video prediction_dict (predicted_emotion, confidence, class_probabilities), matching Member 1's output schema so the two can be compared and combined directly. Once fused, the pipeline produces a complete prediction_output dictionary containing the fused predicted_emotion and confidence, the fused class_probabilities, modality_weights (the physio/video split computed by the gating formula), signal_quality grades for all three raw signals (EEG, GSR, video), and per_modality_predictions (the raw, pre-fusion output from each modality), matching the M2→M3 interface contract detailed in Chapter 5.

#### 4.3.4 Evaluation

The video model achieves a 5-fold cross-validation macro-F1 of 0.6433 ± 0.0303, and a held-out test macro-F1 of 0.6486 ± 0.0154 across 788 clips from 11 actors entirely unseen during training or cross-validation (best single checkpoint: 0.6615), with the near-zero gap between validation and held-out test performance indicating genuine generalisation rather than overfitting. Per-class performance on the held-out test set ranges from 0.82 (happy, the strongest class) to 0.54 (sad, the weakest, most often confused with stress). The complete fused system was evaluated through the eight-condition ablation study: the full gated-fusion method (C5) significantly outperforms both the physio-only (C1, Wilcoxon p = 8.9×10⁻¹⁶) and video-only (C2, p = 5.5×10⁻¹⁰) baselines, and correctly degrades to the physio-only prediction (C7) when video is unavailable, though it does not significantly outperform naive equal-weight fusion (C3) for reasons diagnosed in Section 7.4 and connected directly to Member 1's LOSO evaluation result. Full results, the confusion-matrix analysis, and the honest interpretation of the C3 comparison are presented in Section 7.4.

### 4.4 Member 3 — Explainability and Web Application

#### 4.4.1 Input

The module's input is the fused prediction_output dictionary produced by Member 2's pipeline, together with user authentication credentials for the web application and, for the chat feature, free-text queries a user submits about a specific prediction.

#### 4.4.2 Process

A Kernel SHAP explainability layer is applied over the prediction_output to attribute the fused prediction to each contributing modality and feature. Because the fusion mechanism is itself a fixed, known function of each modality's prediction and weight rather than an opaque end-to-end network, SHAP can be applied directly to this function without needing access to Member 1's or Member 2's underlying model internals, which keeps the explainability layer decoupled from how either upstream model is implemented. A faithfulness score is computed by perturbing individual input features and measuring the resulting change in the fused prediction, giving a quantitative check on whether the SHAP attribution genuinely reflects what drives the prediction rather than only appearing plausible.

A FastAPI backend exposes this functionality, together with the rest of the system's functionality, through nine endpoints: user registration and login (issuing JWT access and refresh tokens), a generic /predict entry point, a video-only /predict/video endpoint, a /predict/multimodal endpoint (accepting a required video file together with optional EEG and GSR files, validated against their required (32, 512) and (512,) shapes respectively, and falling back to video-only prediction via the same underlying pipeline if EEG/GSR are omitted), SHAP explanation retrieval, session listing, chat, and dashboard summary statistics. These are backed by a four-table SQLAlchemy schema — users, sessions, a SHAP-log table, and chat history — over SQLite in development and PostgreSQL in production.

A React frontend renders an emotion-trend line chart, a SHAP bar chart, a session-history panel, and a chatbot panel that bridges the SHAP attribution into a plain-English explanation generated by a large language model. Member 3's novel contribution is a modality-conflict explanation panel, which specifically explains cases where the physiological and video predictions disagree, surfacing the same confidence-and-quality reasoning the gated fusion mechanism used internally to decide which modality to trust more in that instance — connecting the explainability layer directly back to the fusion design in Chapter 5 rather than treating the two as unrelated features. Figure 4.3 shows this architecture end to end.

**[INSERT IMAGE HERE: `docs/report_assets/fig4_3_m3_pipeline.png`]**

**Figure 4.3: Member 3 explainability and web application architecture**

#### 4.4.3 Output

The module's output is a shap_output dictionary — the prediction_output enriched with shap_values, feature_names, faithfulness_score and reliability_flags — persisted to the database and rendered in the dashboard, together with natural-language chat responses returned from the LLM-backed chat endpoint.

#### 4.4.4 Evaluation

The web application was evaluated through a full in-process HTTP test using FastAPI's test client against an isolated test database: user registration and login, a /predict/multimodal request with a real video file and real EEG/GSR windows (returning HTTP 200 with a genuinely fused prediction — both modality weights strictly positive — and a populated SHAP explanation), a video-only request (correctly falling back to video-only weighting), and a request with a malformed EEG array shape (correctly rejected with HTTP 422 and a descriptive error message). The React frontend was verified to build without errors and to expose the EEG/GSR upload controls and the "Run Multimodal Fusion" action introduced alongside the pre-existing video-only path. Full results are presented in Section 7.5.

### 4.5 Summary

This chapter described each member's individual contribution in full technical detail, following a consistent Input/Process/Output/Evaluation structure: Member 1's bidirectional cross-modal attention network over EEG and GSR, evaluated honestly under both a random-split and a subject-independent LOSO protocol; Member 2's ResNet50+BiLSTM video model and the gated fusion mechanism, pipeline and ablation methodology that combine it with Member 1's output; and Member 3's SHAP-based explainability layer and web application, including its novel modality-conflict explanation feature. The next chapter presents the system's analysis and design — the shared taxonomy, interface contracts, signal-quality thresholds and fusion formulation — that make these three independently-developed approaches compose into a single system.
