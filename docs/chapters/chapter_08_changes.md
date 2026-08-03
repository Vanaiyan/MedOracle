# Chapter 8 — Changes Only

**Base document:** `docs/chapters/chapter_08_conclusion.md` (keep all unlisted sections unchanged).

No file names, repository paths, or API route literals in the replacement text below.

---

## REPLACE — Fourth objective paragraph in Section 8.1.1

**Replace the paragraph beginning "The objective to implement a SHAP-based explainability layer…" with:**

The objective to implement an explainability layer and a FastAPI/React web application supporting real multimodal input, including explanation of modality disagreement, was met: the multimodal prediction route accepts real video, EEG, and GSR uploads and returns a fused prediction with closed-form Shapley attributions, faithfulness scoring, and persisted session records, verified through full in-process HTTP testing; the modality-conflict explanation panel surfaces the fusion gate trace and trusted branch to the end user when physiological and video predictions disagree.

---

## REPLACE — Second paragraph of Section 8.1.2 (Overall Reflection)

**Replace the paragraph beginning "Equally important to this project's conclusions…" with:**

Equally important is what was found not to work, and why. Physiological LOSO performance remains weak (macro-F1 0.179), and four controlled diagnostics isolated class imbalance and cross-subject variability as primary causes rather than implementation error. Fusion does not yet significantly outperform equal-weight combination under the integrated physiological checkpoint — an unfavourable but diagnosed result (Section 7.4.4), not omitted from this conclusion. Transparent reporting of both outcomes is treated as part of the project's contribution: a working multimodal, explainable prototype with documented limits is more useful than an inflated unimodal or leaked-split figure would have been.

---

## INSERT — New Section 8.1.3 (after Section 8.1.2, before Section 8.2)

#### 8.1.3 Research Questions — Final Assessment

| ID | Question (abbrev.) | Verdict | Primary evidence |
|---|---|---|---|
| RQ1 | Late fusion beats unimodal baselines | **Partially met** | C5 ≫ C1, C2 (significant); C5 vs C3 not significant |
| RQ2 | Graceful degradation under poor/missing input | **Met** | C6–C8 behaviour matches Section 5.6 design rules |
| RQ3 | Honest unimodal evaluation + diagnosis | **Met** | LOSO 0.179 with four diagnostics; video held-out ~0.65 |
| RQ4 | Interpretable, user-facing explanations | **Partially met** | Shapley + conflict + chat verified technically; no user study |

RQ1 and RQ4 remain open at the "full method beats naive fusion" and "human usefulness" levels respectively; Section 8.2 lists concrete steps to close those gaps without changing the video architecture.

---

## REPLACE — Section 8.2.1 (entire subsection)

#### 8.2.1 Physiological Module Improvements

Priority should be retraining under a coarser three-class or valence/arousal taxonomy — preliminary work lifted LOSO macro-F1 from 0.179 to approximately 0.28 (three-class) or 0.44 (binary valence) without retraining the video model, since five-class probabilities can be remapped by summation. Physiological **quality tiers should be recalibrated against predictive error**, not only raw signal metrics, mirroring the video-threshold recalibration in Section 5.4.3 but using held-out LOSO performance as the calibration target so gating down-weights weak predictors even when inputs appear artefact-free. Domain-adversarial or subject-invariant representation learning remains a longer-term direction for cross-subject EEG transfer.

---

## REPLACE — Section 8.2.2 (entire subsection)

#### 8.2.2 System-Level Improvements

After physiological improvement, the integrated checkpoint should switch from the random-split model to the honestly evaluated one, and the eight-condition ablation re-run — specifically to test whether full gated fusion then significantly outperforms equal-weight fusion (closing the gap in Table 7.3, Section 7.4.6). Cross-dataset pairing should be replaced or supplemented when genuinely synchronised multimodal corpora become available. Fusion gating may also incorporate **calibrated predictive confidence** (for example, temperature scaling or fold-wise calibration on LOSO outputs) so entropy-based scores better reflect subject-independent reliability.

---

## REPLACE — Section 8.2.3 (entire subsection)

#### 8.2.3 Application and Deployment

A structured user study should measure whether Shapley bar charts, conflict explanations, and chatbot narratives improve comprehension and trust for non-technical users — extending RQ4 from engineering verification to human factors. Deployment should move from the development database to production PostgreSQL, harden token and upload handling through independent security review, and define operational monitoring for explanation faithfulness scores and template-fallback rates when the language-model API is unavailable.

---

## Notes (Word assembly — not chapter body)

1. **Table in 8.1.3** is new; format as a real Word table with caption *Table 8.1: Research questions — final assessment* if your report numbering allows a single table in Chapter 8, or present as a structured list without a table number if Faculty guidelines restrict tables in the conclusion chapter.
2. Objectives 1–3 and 5 in Section 8.1.1 are unchanged in the base draft — do not rewrite unless you merge 8.1.3 content into them manually.
3. References and Appendix A must still follow Chapter 8 unchanged.
