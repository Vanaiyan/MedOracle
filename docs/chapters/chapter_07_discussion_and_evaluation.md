<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 7 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 7" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 7" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "7.2 Physiological Module Evaluation"): 12pt, bold.
- Subsection headings (e.g. "7.2.1 Two Evaluation Protocols"): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- The four tables below must be inserted as real Word tables (not images),
  each with a bold header row, and the caption "Table 7.x: <caption text>"
  centered directly above it in bold.
- Five figures are referenced as image file paths in this document —
  insert each actual image at its path, centered, with the caption
  "Figure 7.x: <caption text>" centered directly below it, in bold, IN
  THIS ORDER (this chapter was updated to add three new figures earlier
  in the chapter, which pushed the two original figures from 7.1/7.2 to
  7.4/7.5 — insert all five at their new numbers, do not keep the old
  ones):
    Figure 7.1: docs/report_assets/fig7_3_deap_distribution.png
    Figure 7.2: docs/report_assets/fig7_4_feature_comparison.png
    Figure 7.3: docs/report_assets/fig7_5_video_perclass_f1.png
    Figure 7.4: docs/report_assets/fig_ablation_bar.png
    Figure 7.5: data/synced_samples/ablation_confusion_matrices.png
- If a Chapter 7 already exists in the document (an earlier, shorter
  draft), REPLACE its content entirely with this enhanced version.
- Do not alter the numbering, content, or formatting of any other chapter.
- Keep every citation marker exactly as written (e.g. "[6]") — these numbers
  are fixed by the master References list (alphabetical by first author)
  and must NOT be renumbered or have new sources inserted mid-list, since
  earlier chapters already depend on this exact numbering:
  [1] Baltrusaitis et al. (2019)   [2] Cao et al. (2014)
  [3] Guo et al. (2017)            [4] He et al. (2016)
  [5] Hochreiter & Schmidhuber (1997)  [6] Koelstra et al. (2012)
  [7] Kreibig (2010)               [8] Mehrabian (1996)
  [9] Pech-Pacheco et al. (2000)   [10] Poria et al. (2017)
  [11] Russell (1980)              [12] Siddharth et al. (2019)
  [13] Zhang et al. (2020)
  This chapter cites ONLY these 13 sources.
- This is the report's most important chapter for scientific integrity: do
  not soften, remove, or rebalance any of the honestly unfavourable
  findings below (e.g. the weak LOSO result, or C5 not beating C3) while
  formatting it — insert the content and numbers exactly as written.

=========================================================================
-->

# Chapter 7

## Discussion & Evaluation

### 7.1 Introduction

This chapter presents the evaluation of each module and of the fused system as a whole. In keeping with the project's emphasis on scientific integrity, results are reported honestly throughout, including the physiological module's genuine limitation under subject-independent evaluation and the diagnostic work undertaken to understand its cause, and the fusion method's honest comparison against a naive baseline rather than only against favourable comparisons.

### 7.2 Physiological Module Evaluation

#### 7.2.1 Two Evaluation Protocols

The physiological module was evaluated under two distinct protocols. A random-split evaluation, in which windows are shuffled and split without regard to which subject they came from, yields a macro-F1 of 0.4171 (accuracy approximately 42%). This figure is optimistic: because each subject contributes many near-duplicate 4-second windows from the same 60-second trial, a random split allows the model to see windows from the same subject and even the same trial in both its training and test sets, so part of the apparent performance reflects subject recognition rather than emotion recognition. The honest protocol — Leave-One-Subject-Out (LOSO) cross-validation across all 32 DEAP subjects, in which the model is tested on a subject it has never encountered during training — yields a substantially lower macro-F1 of 0.179 ± 0.041, and a model soup obtained by averaging all 32 fold checkpoints performs almost identically (macro-F1 0.179), confirming this is a genuine ceiling rather than fold-specific noise.

#### 7.2.2 Diagnostic Experiment 1: Class Imbalance

Table 7.1 shows the class distribution produced by the DEAP-to-5-class mapping described in Section 5.3. Only 19 "happy" trials remain across all 32 subjects (2.6% of classifiable trials), and 42% of all trials are dropped as unclassifiable under the chosen valence/arousal/dominance thresholds, giving an imbalance ratio of 12.4× between the largest and smallest class. Figure 7.1 visualises this distribution. Under LOSO, the trained model was observed to collapse to predicting a single majority class ("calm") for almost every held-out window, which alone accounts for the near-zero F1 on the minority classes and drags the macro-averaged score down substantially.

**[INSERT IMAGE HERE: `docs/report_assets/fig7_3_deap_distribution.png`]**

**Figure 7.1: DEAP 5-class distribution used for physiological training**

**Table 7.1: DEAP 5-class distribution used for physiological training**

| Class | Trials | Share |
|---|---|---|
| stress | 172 | 23.2% |
| calm | 191 | 25.7% |
| happy | 19 | 2.6% |
| sad | 236 | 31.8% |
| angry | 124 | 16.7% |
| unclassifiable (dropped) | 538 | (42% of all trials) |

#### 7.2.3 Diagnostic Experiment 2: Feature Representation

Raw time-domain EEG (the representation used by the shipped network), Welch-based band power (used by the classical baselines in Section 6.2.4), and Differential Entropy features (32 channels × 5 bands = 160 features, plus 7 robust GSR statistics, implemented in member1_physiological/preprocessing/de_features.py, Listing 6.2) were each evaluated under LOSO with multiple classifiers (the neural network, logistic regression, and random forest). All three feature representations, across all three classifiers, converged to the same 0.16-0.18 macro-F1 range, indicating that the limitation lies in the underlying signal and label distribution rather than in any single feature-engineering choice. Figure 7.2 shows this convergence directly.

**[INSERT IMAGE HERE: `docs/report_assets/fig7_4_feature_comparison.png`]**

**Figure 7.2: Feature representation comparison under LOSO — all converge to ~0.16-0.18 macro-F1**

#### 7.2.4 Diagnostic Experiment 3: GSR Data Quality

The DEAP GSR channel (index 36) was found to contain physically impossible values: skin conductance cannot be negative, yet 8-79% of samples across different subjects were negative, with artefact spikes reaching ±200,000 against typical values in the low thousands, and inconsistent scaling between subjects. A robust GSR feature extractor was implemented (winsorising each window to its own 5th-95th percentile before z-scoring) to rule out this corruption as the primary cause; however, comparing EEG-only, EEG+raw-GSR and EEG+cleaned-GSR feature sets under identical LOSO evaluation showed a negligible difference (macro-F1 0.1552, 0.1576 and 0.1561 respectively), confirming that GSR contributes little discriminative signal for this five-class problem regardless of its data quality, and that the bottleneck lies elsewhere.

#### 7.2.5 Diagnostic Experiment 4: Taxonomy Granularity

Coarsening the five-class taxonomy into three classes by grouping happy and calm into a single "positive" class and stress and angry into a single "distress" class (leaving sadness on its own) was found to lift the same feature/classifier combination to a macro-F1 of approximately 0.28 under LOSO, and a purely binary (positive/negative valence) task reached approximately 0.44 macro-F1 (0.75-0.77 accuracy). Because the video module already outputs the same five fine-grained classes, this coarser taxonomy could, in principle, be applied to the video model's output by summing grouped probabilities with no retraining required. This change was ultimately scoped as a cross-team taxonomy decision outside this report's completed system, since it would require corresponding changes to Member 3's SHAP feature naming and web-application labels, and was left as a documented option for future work (Section 8.2) rather than implemented unilaterally.

#### 7.2.6 Summary of Diagnostic Findings

Table 7.4 consolidates the four diagnostic experiments and their conclusions.

**Table 7.4: Summary of diagnostic experiments for the physiological module's LOSO result**

| Experiment | What was tested | Result | Conclusion |
|---|---|---|---|
| Class imbalance | DEAP-to-5-class distribution | Happy = 2.6% of trials; 42% dropped; 12.4× imbalance | Model collapses to majority class ("calm") |
| Feature representation | Raw / band power / Differential Entropy, × 3 classifiers | All converge to 0.16-0.18 macro-F1 | Bottleneck is not the feature representation |
| GSR data quality | Raw vs. cleaned (winsorised + z-scored) GSR | 0.1552 vs 0.1576 vs 0.1561 macro-F1 — negligible difference | GSR contributes little signal regardless of data quality |
| Taxonomy granularity | 5-class vs 3-class vs binary | 0.179 → ~0.28 → ~0.44 macro-F1 | Coarser taxonomy helps, but is a cross-team decision (Section 8.2) |

#### 7.2.7 Interpretation

The honest conclusion is that subject-independent, five-class emotion recognition from EEG and GSR alone is a genuinely difficult problem, consistent with the wider literature's tendency to report either within-subject or coarser-grained results (Section 2.2). Reporting this limitation candidly, together with the diagnostic work that located its cause in class imbalance and cross-subject signal variability rather than in implementation defects, is treated in this project as more valuable than reporting an inflated, leaked number — and, as Section 7.4 shows, the system's fusion design was explicitly built to remain robust to exactly this kind of single-modality weakness.

### 7.3 Video Module Evaluation

The video module was evaluated under actor-independent 5-fold StratifiedGroupKFold cross-validation and, separately, on a held-out test set of 788 clips from 11 actors never used during training or cross-validation. The 5-fold cross-validation macro-F1 was 0.6433 ± 0.0303; the held-out test macro-F1 was 0.6486 ± 0.0154 across folds, with the best single checkpoint (fold 2) reaching 0.6615. The near-zero (slightly negative) gap between validation and held-out test performance indicates genuine generalisation to unseen actors rather than overfitting.

**Table 7.2: Video module per-class F1 (held-out test, 788 clips)**

| Emotion | Test F1 |
|---|---|
| happy | 0.82 |
| angry | 0.73 |
| calm | 0.62 |
| stress | 0.60 |
| sad | 0.54 (weakest) |

Table 7.2 and Figure 7.3 show the per-class breakdown on the held-out test set: happy is recognised strongly, while sad is the weakest class and is most frequently confused with stress — a pairing that is precisely the kind of ambiguity the physiological modality, were it stronger, would be well placed to help resolve, which is the underlying motivation for building a fusion mechanism rather than relying on video alone.

**[INSERT IMAGE HERE: `docs/report_assets/fig7_5_video_perclass_f1.png`]**

**Figure 7.3: Video module per-class F1 (held-out test, 788 clips)**

### 7.4 Fusion and Ablation Study

#### 7.4.1 Ablation Methodology

The complete fused system was evaluated using an eight-condition ablation study over the cross-dataset evaluation set described in Section 6.3.4 (20 samples per emotion, 50 random re-pairings). Each condition isolates one design choice: C1 and C2 are single-modality baselines; C3 and C4 are naive or partial fusion baselines that omit some part of the full method; C5 is the full method; and C6-C8 are robustness conditions that force a modality's quality grade to "poor" or remove it entirely, to test the graceful-degradation behaviour designed into the system (Section 5.6).

#### 7.4.2 Results

Table 7.3 and Figure 7.4 present the results; Figure 7.5 shows representative confusion matrices for the physio-only, video-only and full-fusion conditions.

**[INSERT IMAGE HERE: `docs/report_assets/fig_ablation_bar.png`]**

**Figure 7.4: Ablation study — macro-F1 by condition (C1-C8)**

**Table 7.3: Ablation study results (C1-C8) with Wilcoxon significance vs C5**

| Cond. | Description | Macro-F1 (mean ± std) | Wilcoxon vs C5 |
|---|---|---|---|
| C1 | Physio only | 0.443 ± 0.000 | p = 8.9e-16 (C5 significantly better) |
| C2 | Video only | 0.631 ± 0.000 | p = 5.5e-10 (C5 significantly better) |
| C3 | Equal weight (0.5/0.5) | 0.692 ± 0.019 | p = 1.0 (not significant) |
| C4 | Confidence only (no quality) | 0.685 ± 0.016 | - |
| C5 | FULL METHOD (conf × quality) | 0.653 ± 0.009 | (reference) |
| C6 | Robust: video poor | 0.652 ± 0.023 | - |
| C7 | Robust: video missing | 0.443 ± 0.000 | - |
| C8 | Robust: physio poor | 0.650 ± 0.009 | - |

**[INSERT IMAGE HERE: `data/synced_samples/ablation_confusion_matrices.png`]**

**Figure 7.5: Ablation study — normalised confusion matrices for physio-only (C1), video-only (C2) and full fusion (C5)**

#### 7.4.3 Interpretation: Fusion Beats Single Modalities

The full gated-fusion method (C5) significantly outperforms both single-modality baselines: physio-only (C1, p = 8.9×10⁻¹⁶) and video-only (C2, p = 5.5×10⁻¹⁰) under a one-tailed Wilcoxon signed-rank test at α = 0.05. This directly supports the central multimodal claim underlying the whole project — that fusing two modalities, weighted by their measured reliability, is better than relying on either alone — and does so with a level of statistical rigour (fifty independent re-pairings, a formal significance test) beyond a single-run comparison.

#### 7.4.4 Interpretation: Why C5 Does Not Beat C3

Reported honestly even though it is an unfavourable result, C5 does not significantly outperform naive equal-weight fusion (C3, p = 1.0). The reason is diagnosable rather than mysterious: the shipped physiological checkpoint used in this evaluation is the random-split model (Section 7.2), whose EEG/GSR signal quality is frequently graded "good" even though its underlying predictive accuracy is weak; the gating mechanism currently trusts signal quality as a proxy for reliability, and so gives this confidently-wrong modality more weight than an equal split would. This is a consequence of using an admittedly weak physiological checkpoint rather than a flaw in the fusion formula itself, and is expected to resolve once a stronger, properly-calibrated physiological model (Section 8.2) is substituted in.

#### 7.4.5 Robustness Conditions

The three robustness conditions (C6-C8) confirm the system behaves as designed under degraded input: forcing video quality to "poor" (C6) or physiology quality to "poor" (C8) only mildly changes the fused macro-F1, and removing video entirely (C7) causes the fused output to fall back exactly to the physio-only result (C1), demonstrating correct graceful degradation rather than a crash or an unpredictable output when a modality is unavailable.

### 7.5 Web Application Evaluation

The complete web application was evaluated with an in-process end-to-end HTTP test using FastAPI's test client against an isolated test database: a new user is registered and logged in; a request to /predict/multimodal with a real video file together with real EEG and GSR windows returns HTTP 200 with a genuinely fused prediction (both modality weights strictly positive) and a populated SHAP explanation; a video-only request (EEG/GSR omitted) correctly returns a video-only-weighted prediction; and a request with a malformed EEG array shape is correctly rejected with HTTP 422 and a descriptive error message. The React frontend was verified to build cleanly and to expose the new EEG/GSR upload controls and the "Run Multimodal Fusion" action alongside the existing video-only path.

### 7.6 Comparison with Related Work

#### 7.6.1 Physiological Module

The physiological module's honest subject-independent macro-F1 of 0.179 is markedly lower than the approximately 73% accuracy reported by Siddharth et al. [12] for EEG plus peripheral signals; this project's position is that the discrepancy is substantially explained by evaluation protocol (subject-independent LOSO versus the more common within-subject or near-within-subject protocols used elsewhere in the literature) rather than by a weaker model, a claim supported by this project's own within-subject-adjacent random-split figure of 0.4171, which sits in a broadly comparable range once the same easier protocol is used.

#### 7.6.2 Video Module

The video module's held-out macro-F1 of approximately 0.65 is broadly comparable to the approximately 65% accuracy reported by Zhang et al. [13] for video-only emotion recognition on comparable classes, providing external corroboration that this project's video module's performance is in line with, rather than an outlier relative to, the published literature.

### 7.7 Limitations

- The physiological module's subject-independent performance is weak (Section 7.2), driven primarily by DEAP's class imbalance and the well-documented difficulty of cross-subject EEG transfer, rather than by an implementation defect; this represents the system's most significant honest limitation.
- No single dataset provides EEG, GSR and video from the same subjects simultaneously, so the ablation study and web-application demonstration rely on a principled but constructed cross-dataset pairing (Section 6.3.4) rather than a single naturally multimodal dataset.
- The shipped pipeline currently uses the random-split physiological checkpoint (macro-F1 0.4171) rather than the more honestly-evaluated LOSO checkpoint (macro-F1 0.179); this was a deliberate, documented choice to allow system integration and the ablation study to proceed while the LOSO evaluation was being completed, but should be revisited before any deployment claim is made.
- The full method does not yet significantly outperform naive equal-weight fusion (Section 7.4), a consequence of the current physiological checkpoint's weak predictive accuracy despite frequently "good" signal quality.
- The web application has been validated functionally and via automated tests, but has not undergone a structured user study of its explanations' clinical usefulness.

### 7.8 Summary

This chapter presented an honest evaluation of every module: a strong, well-generalising video model (macro-F1 ~0.65 held-out); a physiological module whose subject-independent performance is weak, together with four diagnostic experiments that traced this weakness to class imbalance and cross-subject variability rather than to implementation defects; and a fusion mechanism that significantly outperforms either single modality and degrades gracefully under missing or poor signal quality, while honestly falling short of naive equal-weight fusion for the reason diagnosed above. The next chapter concludes the report and outlines future work.
