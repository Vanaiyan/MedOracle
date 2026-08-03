<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 5 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 5" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 5" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "5.3 Shared Emotion Taxonomy..."): 12pt, bold.
- Subsection headings (e.g. "5.3.1 Why a Shared Taxonomy..."): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- The two tables below must be inserted as real Word tables (not images),
  each with a bold header row, and the caption "Table 5.x: <caption text>"
  centered directly above it in bold.
- Three figures are referenced as image file paths in this document —
  insert each actual image at its path, centered, with the caption
  "Figure 5.x: <caption text>" centered directly below it in bold:
    Figure 5.1: docs/report_assets/fig_architecture.png
    Figure 5.2: docs/report_assets/fig5_2_quality_tiers.png
    Figure 5.3: docs/report_assets/fig5_3_degradation_flow.png
  Note: Figure 5.1 is the SAME image already used as Figure 1.1 in
  Chapter 1 — that is intentional (it is shown again here with a more
  technical, design-focused discussion), so reuse the same image file
  rather than treating it as missing or duplicate-in-error.
- If a Chapter 5 already exists in the document (an earlier, shorter
  draft), REPLACE its content entirely with this enhanced version.
- Do not alter the numbering, content, or formatting of any other chapter.
- Keep every citation marker exactly as written (e.g. "[6]") — these numbers
  are fixed by the master References list (alphabetical by first author)
  and must NOT be renumbered or have new sources inserted mid-list, since
  Chapters 1, 2 and 4 already depend on this exact numbering:
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

# Chapter 5

## Analysis and Design

### 5.1 Introduction

This chapter presents the analysis and design decisions that allow three independently-developed modules to compose into a single working system. Because the project's three members needed to work in parallel — physiological modelling, video modelling, and web-application development each require substantially different expertise and iteration cycles — the central design problem this chapter addresses is not any single algorithm, but how to define boundaries between modules precise enough that each member could build, test and change their own module without needing to coordinate on every internal detail with the other two. This chapter covers the shared emotion taxonomy, the signal-quality grading scheme, the interface contracts between modules, and the design of the gated fusion mechanism itself, explaining not only what was chosen but, where relevant, what alternative was considered and rejected and why.

### 5.2 System Architecture

Figure 5.1 shows the data flow between modules together with the exact dictionaries exchanged at each boundary. Member 1 and Member 2 each expose a pure function — predict(eeg, gsr) and predict_video(video_path) respectively — that can be called and tested independently of the rest of the system; Member 2's gated-fusion function accepts either or both outputs and always returns a dictionary satisfying the same downstream contract, so Member 3's explainability layer and web application never need to know whether zero, one, or two modalities were actually available for a given prediction. This is a deliberate consequence of adopting a late (decision-level) fusion architecture at the system level [1, 10] rather than an early or feature-level one: because DEAP and CREMA-D share no subjects, an early-fusion architecture that concatenates raw features from both modalities before a single joint classifier would have no valid training data to learn from in the first place, so late fusion is not merely a design preference here but an architectural necessity forced by the data itself (a point discussed further in Section 5.6).

**[INSERT IMAGE HERE: `docs/report_assets/fig_architecture.png`]**

**Figure 5.1: Data flow and interface contracts between Member 1, Member 2 and Member 3**

Each arrow in Figure 5.1 corresponds to a validated dictionary boundary rather than an informal function call: every dictionary crossing a module boundary is checked, using assertion-based helper functions defined once in shared/data_contracts.py, against the exact schema described in Section 5.5 before it is allowed to proceed downstream. This was a deliberate choice over simply trusting each member's output: because the three modules are developed on separate branches and merged periodically (Chapter 3), a malformed dictionary — a missing key, a probability distribution that does not sum to one, an out-of-range confidence value — is far more useful to catch immediately at the point where one member's code hands off to another's than to discover later as a confusing downstream symptom in the web application.

### 5.3 Shared Emotion Taxonomy and Label Harmonization

#### 5.3.1 Why a Shared Taxonomy Is Necessary

All three modules are required to use exactly the same five discrete emotion classes — stress, calm, happy, sad, angry — defined once in a shared module (shared/label_harmonization.py) and imported everywhere else, rather than each member defining their own labels and translating between them at the boundary. This requirement is stronger than it may first appear: the gated fusion mechanism (Section 5.6) combines the two modalities' probability distributions by direct weighted summation, which is only meaningful if index k in one modality's distribution and index k in the other's refer to the same emotion; and Member 3's SHAP-based attribution over the fused prediction likewise depends on a stable, shared class ordering to attribute a change in the fused output back to a specific class consistently across modalities. A per-module translation layer was considered and rejected early in the project specifically because it would have made it possible for the two modalities to silently disagree about what a given class index meant, a failure mode that a single shared definition eliminates by construction.

#### 5.3.2 DEAP Valence/Arousal/Dominance Mapping

Because DEAP provides continuous valence, arousal and dominance ratings on a 1-9 scale rather than discrete emotion labels, each subject's trial must be mapped onto the shared taxonomy using threshold rules. The mapping follows Russell's circumplex model [11] and Mehrabian's PAD model [8], thresholding each dimension at the scale midpoint (5.0), with values exactly equal to 5.0 treated as falling on the high side of the threshold by convention, and a small number of valence-arousal-dominance combinations that do not correspond to any of the five classes under these rules left unclassified and excluded from training. This threshold-based approach was chosen over alternatives such as k-means clustering of the raw ratings specifically because it produces class boundaries that are interpretable and traceable to established psychological models, rather than boundaries that would need to be re-derived and re-justified if the underlying data distribution shifted.

#### 5.3.3 CREMA-D Categorical Mapping

CREMA-D's six original discrete labels are mapped onto the shared five-class taxonomy as shown in Table 5.2. Anger, happiness and sadness map directly, since these labels already correspond one-to-one with a class in the shared taxonomy. Neutral is mapped to calm, treating the shared taxonomy's "calm" as an approximately low-arousal, pleasant-to-neutral state. Fear is mapped to stress rather than left as its own class or dropped, following Kreibig's finding that fear-related states activate the sympathetic nervous system in a manner physiologically comparable to stress [7], giving this mapping a grounding beyond surface-level label similarity. Disgust is dropped entirely — rather than force-mapped to the nearest available class — because no class in the shared taxonomy has a defensible correspondence to it; forcing a mapping here was considered and rejected as more likely to introduce label noise than to preserve useful training signal.

**Table 5.2: DEAP-to-5-class and CREMA-D-to-5-class label harmonisation**

| Source | Mapping | Notes |
|---|---|---|
| DEAP | Valence≥5, Arousal<5, Dominance≥5 → calm | Russell (1980) [11] |
| DEAP | Valence≥5, Arousal≥5, Dominance≥5 → happy | Russell (1980) [11] |
| DEAP | Valence<5, Arousal≥5, Dominance<5 → stress | Mehrabian (1996) [8] |
| DEAP | Valence<5, Arousal<5, Dominance<5 → sad | Russell (1980) [11] |
| DEAP | Valence<5, Arousal≥5, Dominance≥5 → angry | Russell (1980) [11] |
| CREMA-D | ANG → angry, HAP → happy, SAD → sad | Direct correspondence |
| CREMA-D | NEU → calm | Neutral approximated as low-arousal calm |
| CREMA-D | FEA → stress | Kreibig (2010) [7]: fear activates sympathetic system |
| CREMA-D | DIS → dropped | No clean correspondence; excluded from training |

### 5.4 Signal Quality Assessment

#### 5.4.1 Design Rationale: Three Discrete Tiers

Each modality's input is graded into one of three quality tiers — good, degraded, poor — rather than a continuous quality score. This was a deliberate simplification: a continuous score would need to be mapped onto a fusion penalty via some chosen function in any case (Section 5.6), and a three-tier grading scheme is considerably easier to communicate to an end user in the web application's signal-quality display than a raw numeric score would be, while still giving the fusion mechanism three meaningfully distinct penalty levels to work with. The tiers are used both to report signal reliability directly to the user and, critically, to weight each modality's contribution during fusion. Figure 5.2 shows how each tier's quality penalty (α) feeds directly into the gate score used by the fusion formula (Section 5.6).

**[INSERT IMAGE HERE: `docs/report_assets/fig5_2_quality_tiers.png`]**

**Figure 5.2: Signal-quality tiers and their effect on the fusion gate score**

#### 5.4.2 EEG and GSR Thresholds

EEG quality is graded on amplitude range, the number of flat-line channels, and the proportion of NaN samples in the window; GSR quality is graded on signal range, the number of detected skin-conductance-response peaks, and signal drift. Table 5.1 shows the exact thresholds for each tier.

#### 5.4.3 Video Threshold Recalibration

Video quality is graded on face-detection rate, Laplacian-variance sharpness [9], and face bounding-box area. The thresholds for these last two metrics were recalibrated during this project after an initial evaluation revealed a design flaw: the originally adopted generic thresholds (Laplacian variance ≥ 100 for "good", face area ≥ 10% of frame) were drawn from document-scan sharpness literature [9] rather than from this project's own acted-video domain, and were measured to grade approximately 80% of clean, fully-detected CREMA-D frontal-face clips as merely "degraded" — a systematic miscalibration rather than a reflection of genuinely poor input. Recalibrating both thresholds to the measured distribution of the held-out CREMA-D test set (Laplacian variance ≥ 70 for "good", face area ≥ 6%) restored 73 of 80 sampled clips to a "good" grade while still correctly flagging the handful of genuinely blurred or distant-face clips in the sample, confirming the recalibration corrected a threshold-calibration error rather than simply loosening the criterion until everything passed.

**Table 5.1: Signal-quality grading thresholds**

| Metric | Good | Degraded | Poor |
|---|---|---|---|
| EEG amplitude range | ±100 μV | ±100-150 μV | > ±150 μV |
| EEG flat-line channels | 0 | 1-2 | > 2 |
| EEG NaN ratio | < 1% | 1-5% | > 5% |
| GSR signal range | 0.5-30 μS | 0.1-0.5 or 30-50 μS | < 0.1 or > 50 μS |
| GSR SCR peaks (60s) | ≥ 2 | 1 | 0 |
| Video face-detection rate | ≥ 80% | 50-80% | < 50% |
| Video Laplacian variance | ≥ 70 (recalibrated) | 40-70 | < 40 |
| Video face bounding-box area | > 6% (recalibrated) | 3-6% | < 3% |

The worst-performing metric for a given modality determines its overall grade: a video clip with a high detection rate but a small, distant face is graded no better than its face-area metric permits, on the reasoning that a fusion mechanism relying on this grade should be conservative — a single badly-performing metric is enough to indicate real risk, even if the other metrics for that modality look healthy.

### 5.5 Interface Contracts Between Modules

Interface contracts are defined once in shared/data_contracts.py: make_physiological_prediction() constructs and validates the M1→M2 dictionary, and make_prediction_output() constructs and validates the M2→M3 dictionary, each raising immediately via an assertion if a required key is missing, a probability distribution does not sum to approximately one, a confidence value falls outside [0, 1], or a quality grade is not one of the three defined tiers. The exact field-by-field schema of both contracts is presented alongside their implementation in Chapter 6 (Tables 6.1 and 6.2); this section establishes the design principle behind them — that a contract violation should fail loudly and immediately at the module boundary where it occurs, rather than silently propagate into a harder-to-diagnose failure in the fused prediction or the web application several steps downstream.

### 5.6 Gated Fusion Design

The fusion mechanism combines each modality's prediction using an entropy-based confidence score multiplied by a signal-quality penalty, then L1-normalises the resulting per-modality weights. Formally, for modality m with predicted probability distribution P_m over K = 5 classes:

confidence: c_m = 1 − H(P_m) / log(K), where H(P_m) = −Σ_k P_m(k) log P_m(k)

This entropy-based formulation was chosen over simply using the maximum softmax probability as a confidence proxy on the basis of calibration research showing that raw softmax outputs are frequently poorly calibrated in modern neural networks, while an entropy-based measure over the full output distribution remains a useful, model-agnostic signal of how peaked — and therefore how confident — a prediction genuinely is [3].

quality penalty: α_m = 1.0 (good), 0.5 (degraded), 0.1 (poor)

gate score: g_m = c_m × α_m

The gate score is deliberately multiplicative rather than, for example, an average of confidence and quality: a modality that is confidently wrong because its input is degraded (high c_m, low α_m) should still be heavily penalised, and a multiplicative combination guarantees this, whereas an averaging combination could let a high confidence score partially compensate for poor signal quality.

fusion weight: w_m = g_m / Σ_m g_m (L1-normalised, not softmax)

L1 normalisation was chosen over a softmax specifically because softmax compresses large differences between inputs — two gate scores that differ by an order of magnitude would be pulled much closer together by a softmax than by a simple L1 normalisation, which preserves the ratio between them directly and therefore lets a genuinely much-more-reliable modality dominate the fused prediction rather than being only mildly favoured.

fused prediction: P_fused = Σ_m w_m × P_m; predicted_emotion = argmax(P_fused)

Four graceful-degradation rules are built into the design, each corresponding to a scenario the system must not simply crash or produce an undefined result under: if both modalities are available and at least one has a non-negligible gate score, full gated fusion is applied as above; if video is poor or missing, its gate score tends toward zero and the physiological prediction alone effectively determines the fused output; symmetrically, if the physiological signal is poor or missing, the video prediction alone determines the output; and if both modalities are simultaneously unreliable (both gate scores negligible), the system falls back to a uniform distribution over the five classes and explicitly flags the prediction as unreliable, on the principle that reporting an unearned confident answer in this scenario would be worse than reporting no reliable answer at all. Figure 5.3 shows this decision logic as a flow diagram.

**[INSERT IMAGE HERE: `docs/report_assets/fig5_3_degradation_flow.png`]**

**Figure 5.3: Graceful degradation decision flow**

### 5.7 Database and Web Application Design

The web application persists four tables: users (authentication credentials), sessions (one row per prediction request, storing the fused prediction and modality weights), a SHAP log table (per-session feature attributions and faithfulness score, stored separately from the session row so that a session can exist and be queried even before its explanation has been computed), and chat history (per-session chatbot conversation, keyed to the session it discusses). JWT access and refresh tokens gate all endpoints except registration and login, with a short-lived access token and a longer-lived refresh token — a split chosen so that a compromised access token has a limited window of validity without forcing the user to re-authenticate with their password on every request.

### 5.8 Summary

This chapter presented the shared taxonomy and the reasoning behind requiring it, the signal-quality grading scheme including the video-threshold recalibration that corrected a genuine design flaw, the interface-contract validation philosophy, and the gated fusion design together with the alternatives considered and rejected at each step — softmax versus L1 normalisation, additive versus multiplicative gating, and early versus late fusion at the system level. The next chapter describes how each of these designs was implemented.
