# Chapter 7 — Changes Only

**Base document:** `docs/chapters/chapter_07_discussion_and_evaluation.md` (keep all unlisted sections unchanged).

No file names or repository paths in the replacement text below. **Do not alter any reported numbers or unfavourable findings.**

---

## INSERT — New subsection after Section 7.1

### 7.1.1 Evaluation Scope and Research Questions

Evaluation is organised around the four research questions stated in Section 1.4.3. **RQ3** (unimodal performance under honest protocols) is addressed in Sections 7.2 and 7.3. **RQ1** and **RQ2** (fusion benefit and graceful degradation) are addressed in Section 7.4. **RQ4** (interpretable delivery to the user) is addressed in Section 7.5.

The primary classification metric throughout is **macro-averaged F1**, chosen because both DEAP (after harmonisation) and CREMA-D exhibit class imbalance; accuracy alone would overweight frequent classes such as calm and sad. Fusion comparisons use **one-tailed Wilcoxon signed-rank tests** (α = 0.05) over 50 independent cross-dataset re-pairings, treating each re-pairing as a paired observation of macro-F1 under two fusion conditions. This protocol is stricter than reporting a single held-out split, because it exposes variance induced by which physiological window is paired with which video clip when no naturally paired dataset exists.

---

## REPLACE — Section 7.2.3 (entire subsection)

#### 7.2.3 Diagnostic Experiment 2: Feature Representation

Raw time-domain EEG (the representation used by the shipped network), Welch-based band power (used by the classical baselines in Section 6.2.4), and Differential Entropy features (32 channels × five frequency bands, plus robust GSR statistics — see Listing 6.2) were each evaluated under LOSO with multiple classifiers (the neural network, logistic regression, and random forest). All three feature representations, across all three classifiers, converged to the same 0.16–0.18 macro-F1 range, indicating that the limitation lies in the underlying signal and label distribution rather than in any single feature-engineering choice. Figure 7.2 shows this convergence directly.

---

## REPLACE — Section 7.5 (entire subsection)

### 7.5 Web Application and Explainability Evaluation

Functional evaluation covered the full stack from authenticated upload through fused prediction, explainability persistence, and chat retrieval — not only whether the emotion label is plausible in isolation.

**Multimodal API path.** An in-process HTTP test against an isolated database confirmed: user registration and login; multimodal prediction with real video, EEG, and GSR uploads returning HTTP 200 with strictly positive weights on both modalities; video-only prediction when physiological uploads are omitted; rejection of malformed EEG array shape with HTTP 422 and a descriptive error; and retrieval of chat history for a session that already has a stored prediction.

**Explainability payload.** Successful multimodal responses included populated Shapley attributions for EEG, GSR, and video, absolute feature-importance magnitudes, a faithfulness score, and reliability flags consistent with the graded signal-quality fields. This confirms the explainability contract defined in Section 5.5.2 is satisfied end-to-end, not only in offline unit tests.

**Modality-conflict path.** Conflict explanation was tested on synthetic fixtures where physiological and video branches predict different emotion classes. The trusted-modality field matched the fused argmax, gate weights reproduced the fusion arithmetic to numerical tolerance, and counterfactual quality-tier messages remained consistent with the gate trace. These tests verify RQ4 at the engineering level — the user-facing panel reflects the same trust logic as fusion — though not yet at the level of user comprehension (Section 7.7).

**Frontend.** The production frontend build completed without errors after multimodal upload controls and the dual action flow (video-only versus full fusion) were added; both paths render the same SHAP and conflict components from the returned payload.

---

## INSERT — New subsection after Section 7.4.5 (before Section 7.5)

#### 7.4.6 Mapping Ablation Outcomes to RQ1 and RQ2

| Research question | Evidence from ablation | Outcome |
|---|---|---|
| RQ1 — Fusion beats unimodal baselines | C5 vs C1 (p = 8.9×10⁻¹⁶), C5 vs C2 (p = 5.5×10⁻¹⁰) | Supported |
| RQ2 — Graceful degradation | C7 equals C1 exactly; C6/C8 change macro-F1 only marginally vs C5 | Supported |
| RQ1 — Full method beats naive fusion | C5 vs C3 (p = 1.0, not significant) | **Not supported** with current physiological checkpoint; diagnosed in Section 7.4.4 |

Reporting the third row explicitly is necessary for scientific integrity: the project achieves statistically significant multimodal gain over either single modality, but has not yet demonstrated significant gain over equal-weight fusion under the checkpoint used for integration testing.

---

## REPLACE — Section 7.7 Limitations (entire bullet list)

- The physiological module's subject-independent performance is weak (Section 7.2), driven primarily by DEAP's class imbalance and cross-subject EEG variability, not by an implementation defect; this is the system's most significant performance limitation.
- No single dataset provides EEG, GSR, and video from the same subjects simultaneously; ablation and demonstration therefore rely on principled cross-dataset pairing (Section 6.3.4), which measures fusion behaviour but not natural multimodal synchronisation.
- The integrated pipeline uses the random-split physiological checkpoint (macro-F1 0.4171) rather than the LOSO checkpoint (macro-F1 0.179) for fusion and ablation, enabling non-degenerate multimodal demos while the honest evaluation was finalised; deployment claims should swap checkpoints first (Section 8.2).
- Quality-aware gating currently penalises **signal cleanliness**, not **predictive reliability**; a weak but "good-quality" physiological model can receive excess weight relative to equal fusion (Section 7.4.4). Recalibrating quality tiers against predictive error — not only amplitude or artefact metrics — is outstanding work.
- Explainability outputs are **technically verified** (faithfulness correlation, gate-trace consistency) but have **not** been evaluated in a structured user study for clarity, trust, or clinical usefulness.
- The web application has not undergone independent security review of authentication, token storage, or file-upload handling beyond functional validation.

---

## REPLACE — Section 7.8 Summary (entire subsection)

### 7.8 Summary

This chapter evaluated every module against the research questions in Section 1.4.3. The video branch generalises to unseen actors (held-out macro-F1 ~0.65). The physiological branch performs poorly under honest LOSO (macro-F1 0.179), with four diagnostic experiments tracing that result to label imbalance and cross-subject variability rather than feature choice or GSR corruption alone. Fusion significantly outperforms either unimodal baseline and degrades gracefully when a modality is missing or forced poor-quality, satisfying RQ1 and RQ2 at the unimodal-comparison and robustness levels — but does not yet beat equal-weight fusion (RQ1 partial gap), for reasons diagnosed in Section 7.4.4. The web stack delivers fused predictions with Shapley attributions, faithfulness scoring, and conflict explanation under automated test (RQ4, engineering level). The next chapter concludes the project and maps objectives and research questions to their final outcomes.

---

## Notes (Word assembly — not chapter body)

1. Section **7.2.3 replacement** changes one sentence only (removes repository path); keep the existing Figure 7.2 block from the base draft immediately after it.
2. **Table 7.4**, **Tables 7.1–7.3**, and **Figures 7.1–7.5** in the base draft are unchanged — keep all numbers exactly as written.
3. Renumbering: if Section 7.5 expands, ensure former cross-references to "Section 7.5" for web-only tests still point to the new combined section.
