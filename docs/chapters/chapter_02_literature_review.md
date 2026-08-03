<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 2 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 2" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 2" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "2.2 Physiological..."): 12pt, bold.
- Subsection headings (e.g. "2.2.1 Datasets for..."): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- The two tables below must be inserted as real Word tables (not images),
  each with a bold header row, and the caption "Table 2.x: <caption text>"
  centered directly above it in bold.
- One figure is referenced as an image file path in this document — insert
  the actual image at that path, centered, with the caption
  "Figure 2.1: <caption text>" centered directly below it in bold:
    Figure 2.1: docs/report_assets/fig2_1_fusion_taxonomy.png
- If a Chapter 2 already exists in the document, REPLACE its content with
  this; otherwise insert it immediately after Chapter 1 and before
  Chapter 3 (if Chapter 3 already exists).
- Do not alter the numbering, content, or formatting of any other chapter.
- Keep every citation marker exactly as written (e.g. "[6]") — these numbers
  are fixed by the master References list (alphabetical by first author)
  and must NOT be renumbered or have new sources inserted into the middle
  of the list, since Chapter 1 already depends on this exact numbering:
  [1] Baltrusaitis et al. (2019)   [2] Cao et al. (2014)
  [3] Guo et al. (2017)            [4] He et al. (2016)
  [5] Hochreiter & Schmidhuber (1997)  [6] Koelstra et al. (2012)
  [7] Kreibig (2010)               [8] Mehrabian (1996)
  [9] Pech-Pacheco et al. (2000)   [10] Poria et al. (2017)
  [11] Russell (1980)              [12] Siddharth et al. (2019)
  [13] Zhang et al. (2020)
  This chapter intentionally cites ONLY these 13 sources (reused across
  multiple subsections where relevant) rather than introducing new ones,
  to keep the whole document's citation numbering stable while it is being
  assembled chapter by chapter.

=========================================================================
-->

# Chapter 2

## Literature Review

### 2.1 Introduction

This chapter reviews prior work relevant to MedOracle's three constituent problems: physiological (EEG/GSR) emotion recognition, facial video emotion recognition, and multimodal fusion with explainability. Each area is reviewed with attention to datasets, feature representations, classification approaches, and the challenges specific to that modality — particularly, for the physiological literature, the distinction between subject-dependent and subject-independent evaluation, which this project's own results (Chapter 7) show to be far from a minor methodological footnote.

### 2.2 Physiological (EEG/GSR) Emotion Recognition

#### 2.2.1 Datasets for Physiological Emotion Recognition

The DEAP dataset [6] is one of the most widely used benchmarks for physiological emotion recognition. It provides 32-channel EEG (sampled at 128Hz in its preprocessed release) together with peripheral physiological signals, including GSR, recorded from 32 subjects while each watched 40 one-minute music videos selected to elicit a range of emotional responses. After each video, subjects self-reported their valence, arousal, dominance and liking on a continuous 1-9 scale, rather than selecting from a fixed set of discrete emotion labels — a design choice that gives researchers flexibility in defining their own discrete taxonomy, but also means that any discrete mapping (including the one this project uses, described in Chapter 5) is itself a modelling decision with direct consequences for class balance, as this project's own results demonstrate. DEAP's continuous, dimensional (valence/arousal/dominance) annotation is representative of a broader pattern in the physiological emotion-recognition literature, where dimensional models of affect are often preferred over discrete categorical labels precisely because self-reported discrete emotion categories are harder for participants to annotate consistently than a continuous rating scale.

Table 2.1 summarises the two primary datasets used across this project, contrasting DEAP's physiological recording protocol against CREMA-D's video recording protocol (reviewed in Section 2.3.1); the two datasets share no common subjects, which is precisely the constraint (Section 2.4) that this project's cross-dataset fusion evaluation methodology is designed to work around.

#### 2.2.2 Feature Extraction and Signal Processing

Early work on physiological emotion recognition relied on hand-crafted features computed from the raw signal: time-domain statistics (mean, variance, zero-crossing rate), frequency-domain band power computed via Fourier or Welch power spectral density estimation across the standard EEG bands (delta, theta, alpha, beta, gamma), and simple GSR descriptors such as the number and amplitude of skin-conductance-response peaks. Differential Entropy (DE), a log-transform of band power, has become a de facto standard feature for EEG emotion recognition because the log transform makes the otherwise heavy-tailed band-power distribution closer to Gaussian, which in turn makes it considerably easier for a downstream classifier to separate — a property this project verified directly (Section 7.2) by comparing raw band power, DE, and the raw time-domain signal itself under an identical evaluation protocol.

More recent work has moved toward learned feature representations: convolutional architectures inspired by EEGNet apply small, channel-wise temporal filters directly to the raw signal, allowing the network to learn its own frequency-selective filters rather than relying on a fixed Fourier or wavelet decomposition chosen in advance. This is the approach this project's own physiological encoder (Chapter 6) takes for its EEG branch, using a depthwise-separable convolutional stack, before Member 1's bidirectional cross-modal attention layer combines the resulting EEG representation with an analogously encoded GSR representation.

#### 2.2.3 Classification Approaches

Classical classifiers — Support Vector Machines, Random Forests, and Multi-Layer Perceptrons trained on hand-crafted spectral features — remain common baselines in the physiological emotion-recognition literature and were implemented as such in this project (Section 6.2.4) specifically to establish whether a given evaluation result reflects a limitation of the neural architecture or of the underlying data. Deep learning approaches for EEG emotion recognition have layered convolutional feature extraction with recurrent (LSTM [5]) or attention-based temporal modelling to capture the evolution of the signal over a trial window, and, closer to this project's own architecture, some approaches use cross-modal attention specifically to let one physiological signal inform the interpretation of another rather than combining them only at a late, decision-level stage.

Siddharth et al. [12] combined EEG with peripheral physiological signals (including GSR) on the DEAP dataset and reported approximately 73% classification accuracy — a figure that, as discussed further in Section 2.2.4 and revisited in this project's own comparison with related work (Section 7.6), is understood in this report to reflect a within-subject or near-within-subject evaluation protocol common in this literature, rather than the harder subject-independent setting this project adopts for its own headline evaluation.

#### 2.2.4 Challenges: Cross-Subject Generalisation and Class Imbalance

Two challenges recur throughout the physiological emotion-recognition literature and were found, in this project's own experiments, to be decisive rather than incidental. The first is cross-subject generalisation: a model trained on a group of subjects frequently fails to transfer its performance to a new, previously unseen subject, because the electrophysiological signature of a given emotion varies considerably between individuals — in contrast to facial expression, which is comparatively more consistent across people (Section 2.3). This is most rigorously exposed by Leave-One-Subject-Out (LOSO) cross-validation, in which every fold tests on a subject entirely absent from training; LOSO evaluations of fine-grained (four- or five-class) discrete emotion recognition are reported far less often in the literature than within-subject or coarser-grained (binary valence/arousal) evaluations, and, where they are reported, generally show substantially lower performance — a pattern this project's own results in Section 7.2 corroborate directly.

The second challenge is class imbalance arising from the mapping of a continuous, dimensional rating scale onto a discrete taxonomy: because the choice of thresholds on valence, arousal and dominance is itself a modelling decision, a poorly-balanced choice of thresholds can leave some discrete emotion classes represented by only a handful of examples across an entire dataset, an effect this project measured directly and traced as a primary cause of its physiological module's evaluation result (Section 7.2).

### 2.3 Facial Video Emotion Recognition

#### 2.3.1 Datasets for Facial Video Emotion Recognition

The CREMA-D dataset [2] comprises over 7,000 audio-visual clips from 91 actors spanning a range of ages and ethnic backgrounds, each portraying one of six emotions (anger, disgust, fear, happiness, neutral, sadness) at varying levels of intensity, with each clip additionally rated by multiple independent human evaluators for the perceived emotion. This crowd-sourced multi-rater annotation, combined with its actor diversity, makes CREMA-D well suited to training models intended to generalise across previously unseen individuals, and is the dataset used for this project's video-based recognition module. RAVDESS, a comparable but smaller audio-visual dataset recorded from 24 professional actors under more controlled studio conditions, was used during this project's model-development phase alongside CREMA-D to increase training diversity, as described in Chapter 6.

**Table 2.1: Comparison of the DEAP and CREMA-D datasets**

| Property | DEAP | CREMA-D |
|---|---|---|
| Modality | EEG (32 ch, 128Hz) + GSR (peripheral) | Facial video (~30fps) |
| Subjects/Actors | 32 subjects | 91 actors |
| Stimuli | 40 one-minute music videos per subject | ~7,000 short emotionally-labelled clips |
| Labels | Continuous valence/arousal/dominance (1-9) | Discrete: anger, disgust, fear, happy, neutral, sad |
| Annotation | Self-reported by the subject | Crowd-sourced, multiple raters per clip |
| Overlap with the other dataset | None — no shared subjects | None — no shared subjects |

#### 2.3.2 Preprocessing and Face Detection

A face-detection front end is standard practice ahead of the recognition network itself, isolating the face region from background and reducing the input's variability due to camera framing. YOLO-family detectors are increasingly used for this purpose because of their speed and robustness across lighting conditions and pose. Detected faces are conventionally cropped and resized to a fixed input resolution before being passed to the recognition network; this project's own development process, however, found that cropping to the detected face region measurably reduced recognition accuracy relative to resizing the full, uncropped frame (Section 6.3.1), an empirical finding that runs against the more common assumption that cropping is strictly beneficial, and which led this project to repurpose its face-detection pass as a signal-quality assessment tool (Section 5.4) rather than a cropping step.

#### 2.3.3 Feature Extraction and Temporal Modelling

Convolutional architectures — ResNet variants [4] in particular — are the dominant choice for extracting per-frame spatial features in video-based facial emotion recognition, typically initialised from weights pretrained on a large image classification dataset and then fine-tuned, fully or partially, on the target emotion dataset. Because a facial expression evolves over the duration of a clip rather than being fully captured by any single frame, a temporal model — most commonly a Long Short-Term Memory network [5] or its bidirectional variant — is applied over the sequence of per-frame features to capture how the expression develops, rather than aggregating frame-level predictions independently. This convolutional-encoder-plus-recurrent-temporal-model combination is the architecture this project's own video module adopts (Chapter 6), using a partially fine-tuned ResNet50 encoder followed by a two-layer bidirectional LSTM.

Zhang et al. [13] reported approximately 65% accuracy for video-only emotion recognition on a comparable set of classes, a figure broadly consistent with this project's own video module's held-out performance (Section 7.3), and used in Section 7.6 as a point of comparison for this project's own results.

#### 2.3.4 Challenges in Video-Based Emotion Recognition

Video-based facial emotion recognition faces its own distinct challenges: variation in pose and lighting, partial occlusion of the face, and the risk of a temporal model overfitting to actor-specific visual characteristics — a person's facial structure or mannerisms — rather than learning genuinely emotion-relevant patterns, if training and evaluation are not conducted with an actor-independent split. This last risk was encountered directly during this project's own model development (Section 6.3.1), where an early, more heavily fine-tuned configuration of the video model showed a large gap between training and validation performance indicative of exactly this kind of overfitting, and was resolved through a combination of discriminative learning rates and actor-independent cross-validation.

### 2.4 Multimodal Fusion Approaches

#### 2.4.1 Fusion Taxonomy

Multimodal affective-computing surveys [1, 10] distinguish three broad fusion strategies, illustrated schematically in Figure 2.1. Early (feature-level) fusion concatenates raw or lightly-processed features from each modality before a single classifier is trained over the combined representation; this requires the modalities to be temporally aligned and is disrupted if one modality is missing at inference time. Intermediate (attention-level) fusion allows representations from different modalities to interact within the network — for example, through cross-modal attention — before a shared decision is reached; Member 1's bidirectional cross-modal attention between EEG and GSR (Section 2.2.2) is an example of this style of fusion applied within a single modality family. Late (decision-level) fusion, by contrast, trains each modality's model independently and combines only their final output distributions, typically via weighted averaging. Late fusion is generally preferred when modalities have different temporal resolutions, different noise characteristics, and — as is the case for this project — come from datasets with no shared subjects, since DEAP and CREMA-D share no overlapping individuals and joint end-to-end training is therefore not possible without a shared subject pool [1, 10]. This project's system is consequently hybrid: intermediate (attention-based) fusion within the physiological branch, and late fusion across the physiological and video branches, combined via the gated fusion mechanism described in Chapter 5.

**[INSERT IMAGE HERE: `docs/report_assets/fig2_1_fusion_taxonomy.png`]**

**Figure 2.1: Fusion taxonomy — early, intermediate and late fusion**

#### 2.4.2 Confidence- and Quality-Aware Fusion

Static or learned-but-fixed late-fusion weights are common in the literature, but adaptive, confidence-driven weighting — in which the contribution of each modality is adjusted at inference time according to how reliable that modality's prediction currently appears to be — is less common, despite direct motivation from the calibration literature. Guo et al. [3] showed that modern neural networks are frequently poorly calibrated when judged by their raw softmax output, but that an entropy-based confidence measure computed over the full output distribution remains a useful, model-agnostic signal of prediction reliability even when the raw softmax values themselves are not directly trustworthy as probabilities — the basis of this project's own confidence term (Chapter 5). Separately, signal-quality-aware weighting, in which a modality's fusion weight is penalised according to an independently measured quality metric (blur, missing data, artefact) rather than the model's own confidence, has precedent in the image-quality literature: the Laplacian-variance sharpness metric this project uses for video-quality grading (Chapter 5) originates with Pech-Pacheco et al. [9]. This project's gated fusion design combines both signals — entropy-based confidence and an independently measured quality grade — multiplicatively, on the basis that a modality can be quality-degraded without necessarily producing an obviously low-confidence output, and vice versa, so relying on either signal alone would leave a gap the other is needed to close.

**Table 2.2: Summary of related multimodal emotion-recognition approaches**

| Approach | Fusion level | Subject-independent evaluation | Adapts to per-input signal quality |
|---|---|---|---|
| Early/feature-level fusion (general survey pattern [1, 10]) | Early | Not typically emphasised | No |
| Static-weight late fusion (general survey pattern [1, 10]) | Late | Not typically emphasised | No |
| EEG + peripheral physiological signals [12] | Early/feature-level | Not the primary reported protocol | No |
| Video-only recognition [13] | Single-modality | Partially | N/A |
| MedOracle (this project) | Hybrid: intermediate (within physiology) + late (across modalities), confidence- and quality-gated | Yes (physiological module evaluated under LOSO; video module under actor-independent cross-validation) | Yes |

### 2.5 Explainable AI for Emotion Recognition

SHAP (SHapley Additive exPlanations) provides a model-agnostic, game-theoretically grounded attribution of a prediction to its input features, and has been widely adopted in sensitive application domains — including healthcare-adjacent settings — precisely because it produces a per-instance, per-feature attribution rather than only a global feature-importance ranking computed once over an entire dataset. In a multimodal fusion setting, applying SHAP over the fused prediction naturally extends to attributing the outcome to each modality's contribution rather than only to individual low-level features, which is the basis of this project's own explainability layer (Member 3, Chapter 4) and its associated faithfulness metric, obtained by perturbing input features and measuring the resulting change in the fused prediction, so that the explanation's own reliability can be quantified rather than simply assumed.

### 2.6 Summary

This chapter reviewed the datasets, feature representations, classification approaches and fusion strategies relevant to physiological, video-based and multimodal emotion recognition, and highlighted two recurring themes that proved directly consequential to this project's own results: the gap between within-subject and subject-independent evaluation of physiological emotion recognition, and the relative scarcity, in prior work, of fusion mechanisms that adapt to per-input signal quality rather than relying on fixed or purely confidence-based weighting. The next chapter describes the technologies adopted to implement the system that this project's design responds to these findings with.
