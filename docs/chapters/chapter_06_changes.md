# Chapter 6 — Changes Only

**Base document:** `docs/chapters/chapter_06_implementation.md` (keep all unlisted sections unchanged).

No file names or repository paths appear in the replacement text below. Code listings omit source-path comments — use only the caption and code body in Word.

---

## REPLACE — Section 6.1 Introduction (entire subsection)

### 6.1 Introduction

This chapter describes the implementation of each module following the design in Chapter 5. The codebase is organised into three member packages (physiological, video-and-fusion, explainability-and-web) plus a shared package holding the emotion taxonomy, label harmonisation rules, and interface-contract validators. Where a design decision is best shown through implementation detail, a short representative listing is included; listings show logic only and are not tied to storage locations in the repository tree.

---

## REPLACE — Section 6.2.1 (first sentence only)

**Replace the opening sentence of Section 6.2.1:**

The DEAP loader reads each subject's pickled trial file, trims the first 3 seconds (384 samples) of baseline recording from each 60-second trial, and slides 15 non-overlapping 4-second (512-sample) windows across the remainder.

---

## REPLACE — Section 6.2.5 (final paragraph only)

**Replace the last paragraph of Section 6.2.5:**

The physiological module exposes a single prediction entry point accepting EEG and GSR windows and returning a dictionary matching Table 6.1, validated by the shared contract helper before hand-off to fusion.

---

## REPLACE — Section 6.3.2 (paragraph after Listing 6.3 — entire paragraph)

This implementation is validated against an independently developed reference fusion module (Member 3) using parametrized unit tests across every combination of physiological and video signal-quality grades, asserting identical modality weights and fused probabilities to within 1e-9 absolute tolerance. A single canonical fusion implementation is therefore shared by the pipeline, ablation study, and explainability layers, eliminating drift between numerically equivalent copies of the same formulas.

---

## REPLACE — Section 6.3.3 (opening sentence + Listing 6.4 caption context)

**Replace the first sentence of Section 6.3.3:**

The full-pipeline entry point lazily loads the physiological predictor and video model, then combines their outputs through gated fusion.

**Replace Listing 6.4** — remove path comment from code block; caption remains *Listing 6.4: Full-pipeline entry point — any single modality may be supplied alone*.

---

## REPLACE — Section 6.3.4 (first sentence only)

**Replace the opening sentence:**

Because DEAP and CREMA-D share no common subjects, a paired evaluation set was constructed by pairing, for each of the five emotion classes, a real DEAP physiological window with a real CREMA-D video clip carrying the same harmonised label.

*(Keep the remainder of 6.3.4 unchanged.)*

---

## REPLACE — Section 6.3.5 (first sentence only)

**Replace the opening sentence:**

The ablation study precomputes each modality's prediction once per held-out sample (20 DEAP windows and 20 held-out CREMA-D clips per emotion, from actors never used in video training), then evaluates eight fusion conditions over 50 random re-pairings of those predictions.

*(Keep the remainder of 6.3.5 unchanged.)*

---

## REPLACE — Section 6.4.1 (entire subsection)

#### 6.4.1 Explainability Implementation

Explainability is implemented as a pipeline over the fused prediction dictionary rather than as gradient access into upstream neural networks.

**Shapley attribution.** A custom two-modality explainer enumerates all four coalitions of physiological and video inputs (neither, physiological only, video only, both). For each coalition, class probabilities are replaced with a fixed background distribution when a modality is absent, and the gated fusion function is evaluated to obtain the winning-class probability. Because there are only two fusion players, Shapley values for physiological and video contributions are computed in closed form without Monte Carlo sampling. The physiological Shapley value is subdivided into EEG and GSR attributions using quality-weighted splitting, reflecting that both signals are processed jointly in the physiological branch.

**Faithfulness.** After attributions are computed, modalities are masked in descending order of absolute SHAP magnitude and the fused confidence for the predicted class is re-evaluated after each mask. A correlation score between attribution rank and confidence drop quantifies whether the explanation tracks the prediction; this score is stored alongside attributions in the SHAP log table.

**Output assembly.** The explainability builder merges prediction fields, SHAP values, feature importance magnitudes, faithfulness score, reliability flags, and coalition audit values into the enriched dictionary consumed by the API and dashboard.

---

## INSERT — New Section 6.4.2 (renumber existing 6.4.2 Backend → 6.4.3, 6.4.3 Frontend → 6.4.4)

#### 6.4.2 Modality-Conflict and Chat Implementation

When physiological and video argmax labels differ, a conflict explainer reconstructs the fusion gate trace from the stored prediction: per-modality confidence, quality penalty, gate score, normalised weights, and which branch the fused label agreed with. Counterfactual analysis varies a modality's quality tier (for example, from degraded to good) and re-runs the gate arithmetic to determine whether the fused emotion would change — making the fusion rule inspectable without re-loading upstream models.

The chatbot consumes the structured prediction and SHAP payload (and conflict trace when present). When an external language-model API is configured, prompts are constrained to the numeric fields and retrieved knowledge-base facts; generated text is scored against those numbers and regenerated or replaced with a template rationale if faithfulness falls below threshold. Without an API key, the template path ensures the demo remains runnable offline.

---

## REPLACE — Section 6.4.2 Backend (will become 6.4.3 after insert above)

*If you inserted 6.4.2 above, apply this block as **Section 6.4.3 Backend**.*

#### 6.4.3 Backend

**Table 6.3: FastAPI endpoint summary** *(replace entire table)*

| Method | Path | Purpose |
|---|---|---|
| POST | /auth/register | User registration |
| POST | /auth/login | JWT access and refresh token issue |
| POST | /predict/multimodal | Video + optional EEG/GSR upload → fused prediction and SHAP |
| POST | /predict/video | Video-only prediction |
| GET | /explain/{session_id} | SHAP explanation for a stored session |
| GET | /sessions | List user sessions |
| GET | /sessions/{session_id} | Session detail |
| POST | /chat | Chatbot message grounded in SHAP output |
| GET | /chat/history/{session_id} | Chat history for a session |
| GET | /dashboard/summary | Aggregated dashboard statistics |

The multimodal prediction route accepts a required video upload together with optional EEG and GSR arrays, validates EEG shape (32 × 512) and GSR shape (512,) before inference, and falls back to video-only fusion when physiological files are omitted — one pipeline implementation for both modes.

**Listing 6.5** — retain the endpoint signature listing but **delete the comment line** at the top of the code block that names a source file; caption only: *Listing 6.5: Multimodal prediction endpoint signature*.

---

## REPLACE — Section 6.5 Testing (entire subsection)

### 6.5 Testing

The project maintains 82 automated tests, all passing: 59 over label harmonisation (DEAP threshold boundaries, CREMA-D mappings, batch helpers), 18 over gated fusion (including 12 parametrisations asserting equivalence with the reference fusion implementation across all quality-grade combinations, plus graceful-degradation and both-modality-unreliable edge cases), and 5 over the full pipeline (physiological-only, video-only, multimodal, and validation that rejects an incomplete EEG/GSR pair).

An in-process HTTP test exercises registration, login, multimodal prediction with real uploads (HTTP 200, both modality weights positive, populated SHAP fields), video-only prediction, malformed EEG shape (HTTP 422), and chat history retrieval. Conflict explanation is tested on synthetic fixtures where branches intentionally disagree, verifying correct trusted-modality reporting and numerical agreement with fusion weights. The React frontend build was confirmed clean after multimodal upload controls were added.

---

## REPLACE — Section 6.6 Summary (final sentence only)

**Replace the last sentence:**

The next chapter presents the evaluation and discussion of the complete system, including an honest account of the physiological module's limitations and the diagnostic experiments that explain them.

---

## Notes (for Word assembly — not chapter body)

1. After inserting new **6.4.2**, renumber former 6.4.2 → 6.4.3 and 6.4.3 → 6.4.4.
2. **Table 5.3** is new in Chapter 5; insert when applying Chapter 5 changes.
3. Strip `# …` comment lines from **Listings 6.1, 6.2, 6.3, 6.5** in the base Chapter 6 draft for the final Word document (logic-only listings).
