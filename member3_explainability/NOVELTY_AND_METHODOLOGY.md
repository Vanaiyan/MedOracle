# Trust-Aware Modality-Conflict Explanation for Quality-Gated Multimodal Emotion Recognition

**Member 3 — Adshaya Balarajah (214024V)**
MedOracle · Multimodal Emotional Recognition System with Explainability
Faculty of Information Technology, University of Moratuwa · Level 4 FYP (2026)
Supervisor: Dr. Firdhous M.F.M.

---

## 1. One-sentence contribution (viva defence)

> *To the best of our knowledge, this is the first faithfulness-validated method for explaining and quantifying trust in the conflict resolution of a quality-gated fusion of physiological (EEG/GSR) and facial-video emotion signals.*

This narrow scoping is deliberate: it is defensible because the contribution is only possible on this exact architecture (Member 2's confidence × signal-quality gate over two modalities that can disagree), so it cannot be dismissed as generic SHAP, and no prior work occupies this intersection.

---

## 2. Problem and motivation

The base MedOracle system fuses a physiological prediction (Member 1) and a video prediction (Member 2) with a quality-gated late-fusion rule. When the two modalities **disagree** — physiology says *stress*, the face says *happy* — the system silently outputs a single fused label. In a clinical / affective-health setting a prediction that cannot be justified is unusable, and a *confident* prediction built on a *low-quality* signal is actively dangerous.

The explainability layer must therefore answer three questions the base system cannot:

1. **Why** did physiology and video disagree?
2. **How** did the quality gate resolve the disagreement?
3. **Which** modality should be trusted, and **how much can the explanation itself be trusted**?

## 3. Gap in the literature

A targeted review (2023–2025) shows every neighbouring line of work stops short of this intersection:

- **Modality-conflict work** (Benchmarking and Bridging Emotion Conflicts, arXiv:2508.01181, 2025; Dual-Path Conflict Resolution, 2025; EmoMM, 2025) *resolves/reduces* conflict inside the model — this is fusion work (Member 2's territory) — and operates on **text/audio/video**, never **physiology vs video**. None *explains* the conflict to a human.
- **Multimodal XAI reviews** explicitly name *"explaining the employed modality fusion scheme itself"* and *"quantifying each modality's contribution"* as open problems, and note that multimodal-explanation **benchmarks remain underdeveloped**.
- **LLM-explanation faithfulness** tooling (RAGAS, TruLens, DeepEval; CAuSE, 2025) measures faithfulness of generated text **to retrieved documents** — not to a model's own attribution/gate mathematics.

The unclaimed intersection — conflict *explanation* (not resolution) for physiology-vs-video, with faithfulness measured against the *gate maths*, plus a reliability/trust score — is this contribution.

## 4. Method

### 4.1 Modality-Conflict Explanation (headline capability)

Operates post-hoc on Member 2's `prediction_output` contract; no model retraining. `member3_explainability/conflict/`.

- **Conflict detection** — compares the two per-modality argmax labels.
- **Gate resolution** — re-derives the introspectable gate trace (entropy confidence `c = 1 − H/logK`, quality penalty `α ∈ {1.0, 0.5, 0.1}`, gate score `g = c·α`, L1-normalised weights `w`) so the system can state *why* one modality won.
- **Exact 2-modality Shapley** — attributes the fused decision to each modality. With M = 2 players the Shapley values have a closed form (no sampling); value function `v(S)` = probability mass on the fused class using only modalities in `S`. Efficiency (`φ_physio + φ_video = v(all) − v(∅)`) holds to machine precision.
- **Counterfactuals over the gate variables** — because fusion is arithmetic, a counterfactual ("if EEG quality were *good*, the decision flips to *stress*") is a re-run of the weighted sum with a flipped `α` — milliseconds, no retraining.
- **Resolution trust score** — `trust = |w_physio − w_video| × α_winner`, flagging resolutions that were marginal or that trusted a low-quality modality.

### 4.2 Attribution engine — the "why not SHAP?" answer

`member3_explainability/attribution/`. The interim-evaluation objection ("SHAP is used too often") is answered by **not defaulting** to it:

- **Integrated Gradients** (Sundararajan et al., 2017) is the **primary** feature-level method — axiomatic, faithful, and cheap for differentiable nets (the ResNet50 + BiLSTM + attention branches are all differentiable).
- **Kernel SHAP** (Lundberg & Lee, 2017) is retained as a **compared baseline**; exact 2-modality Shapley is used for modality-level attribution.
- The model's **intrinsic attention** weights corroborate the post-hoc attributions.
- The method is **selected by measured faithfulness (comprehensiveness)**, not by convention — so the choice of attribution primitive is a documented, data-backed decision.

### 4.3 Faithfulness + hallucination evaluation (safety layer)

`member3_explainability/evaluation/`. Three metrics, scored against the **attribution / gate maths** (not against retrieved documents — the distinction from RAGAS):

- **Faithfulness** — fraction of the explanation's checkable claims (fused emotion, trusted modality, counterfactual, mechanism) that match the maths.
- **Hallucination rate** — fraction of claims that are false or unsupported.
- **Clinical relevance** — rubric coverage (names both modalities, notes the disagreement, cites signal quality/confidence, uses correct terminology, grounds mechanisms).

**RAG grounding** (`knowledge_base.py`) restricts the LLM chatbot to a small curated set of the project's own citations (Russell 1980, Kreibig 2010, Guo 2017, Koelstra 2012, Cao 2014, Poria 2017); a mechanism term counts as *grounded* only if a retrieved fact supports it. A **hallucination guard** verifies every generated explanation against the maths and regenerates (or falls back to a guaranteed-faithful template) if it fails — so every explanation the system returns has either passed a numeric faithfulness check or been replaced.

## 5. Results (verified in code)

| Result | Value | Meaning |
|---|---|---|
| Exact-Shapley efficiency error | ≈ 2×10⁻¹⁶ | modality attribution is exact, not approximate |
| Gate re-derivation vs M2 contract | ≤ 5×10⁻⁵ (display rounding) | conflict layer reproduces the real fusion |
| Attribution faithfulness (planted model) | IG ≈ SHAP ≈ +0.033 > attention +0.019 | IG/SHAP recover true drivers; selection is data-backed |
| Faithful vs corrupted explanation | 0.99 / 0.01 vs 0.08 / 0.92 (faithfulness / hallucination) | metrics cleanly separate good from hallucinated |
| RAG on vs off (same faithful text) | hallucination 0.01 → 0.20 | grounding measurably reduces hallucination |
| Hallucination guard | rejects a lying generator, falls back after 2 attempts | no unverified explanation is ever returned |

(Reproduce: `python -m member3_explainability.evaluation.run_eval`, `python -m member3_explainability.conflict.conflict_explainer`, `python -m member3_explainability.attribution.attribution_engine`.)

## 6. How this answers the evaluators

- **"There is no novelty in your scope."** The novelty is a *capability* (conflict explanation) plus an *evaluation* (faithfulness of that explanation), targeting a named open problem, unclaimed for physiology-vs-video. It is uniquely enabled by the teammates' architecture, so it cannot be reduced to "you imported SHAP."
- **"Why did you use SHAP — it is used so many times?"** SHAP is now one *baseline* in a benchmarked set; the primary primitive is Integrated Gradients, selected by measured faithfulness. The contribution sits at the conflict/fusion level, above the attribution tool.
- **Equal weight to the other two members.** Member 1 contributes a novel *model* (bi-attention); Member 2 a novel *algorithm* (gated fusion); Member 3 contributes a novel *capability* + a novel *evaluation* + the deployed system — structurally equal or heavier.

## 7. System integration

Surfaced through the existing FastAPI + React application (`member3_explainability/backend`, `.../frontend`):

- `POST /explain/conflict` — returns the full conflict explanation + a hallucination-verified natural-language explanation (gate-normalised for internal consistency).
- `ConflictExplanationPanel.jsx` — dashboard panel visualising the disagreement, gate resolution, Shapley contributions, counterfactual, and verified explanation with faithfulness/hallucination badges.
- Runs on SQLite (dev) / PostgreSQL (prod) via the existing SQLAlchemy async layer.

## 8. Remaining work

- **Live LLM path** — wire the chatbot to the project's Gemini key (the hallucination guard already wraps it; template fallback works today).
- **Expert validation** — a small study (supervisor + peers, n ≈ 15–20) rating a sample of explanations; report correlation between the automated faithfulness metric and human judgement. This validates the metric and is itself evaluation novelty.
- **Real-model results table** — swap synthetic `prediction_output` for Member 1/Member 2's trained-model outputs and regenerate all tables.

## 9. Key references

Russell (1980); Mehrabian (1996); Kreibig (2010); Koelstra et al. (2012); Cao et al. (2014); Guo et al. (2017); Sundararajan et al. (2017), *Axiomatic Attribution for Deep Networks* (Integrated Gradients); Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* (SHAP); Poria et al. (2017); *Benchmarking and Bridging Emotion Conflicts for Multimodal Emotion Reasoning* (arXiv:2508.01181, 2025); multimodal-XAI reviews (2023–2025); RAGAS / CAuSE (2025).
