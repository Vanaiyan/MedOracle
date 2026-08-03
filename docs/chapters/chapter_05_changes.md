# Chapter 5 — Changes Only

**Base document:** `docs/chapters/chapter_05_analysis_and_design.md` (keep all unlisted sections unchanged).

No file names or repository paths appear in the replacement text below.

---

## REPLACE — Section 5.2 (paragraph 2 only)

**Replace the second paragraph of Section 5.2** (the one beginning “Each arrow in Figure 5.1…”):

Each arrow in Figure 5.1 corresponds to a validated dictionary boundary rather than an informal function call: every dictionary crossing a module boundary is checked, using shared assertion-based validation helpers, against the exact schema described in Section 5.5 before it is allowed to proceed downstream. This was a deliberate choice over simply trusting each member's output: because the three modules are developed on separate branches and merged periodically (Chapter 3), a malformed dictionary — a missing key, a probability distribution that does not sum to one, an out-of-range confidence value — is far more useful to catch immediately at the point where one member's code hands off to another's than to discover later as a confusing downstream symptom in the web application.

---

## REPLACE — Section 5.3.1 (first paragraph only)

**Replace the first paragraph of Section 5.3.1** (remove the parenthetical path reference):

All three modules are required to use exactly the same five discrete emotion classes — stress, calm, happy, sad, angry — defined once in a shared label-harmonisation module and imported everywhere else, rather than each member defining their own labels and translating between them at the boundary.

---

## REPLACE — Section 5.5 (entire subsection)

### 5.5 Interface Contracts Between Modules

Interface contracts are enforced through shared factory and validation helpers that construct the Member 1→Member 2 and Member 2→Member 3 dictionaries and raise immediately if any invariant is violated. The design principle is that a contract violation should fail loudly at the module boundary where it occurs, rather than propagate silently into a fused prediction or web response several steps downstream.

**Member 1 → Member 2 (`physiological_prediction_dict`).** Required fields: fused emotion label (one of the five shared classes); entropy-based confidence in [0, 1]; a five-class probability dictionary whose keys match the shared taxonomy exactly and whose values sum to approximately one; and nested signal-quality grades for EEG and GSR, each constrained to good, degraded, or poor.

**Member 2 → Member 3 (`prediction_output`).** Required fields: fused label, fused confidence, and fused class probabilities (same validation as above); modality weights for physiological and video branches that sum to 1.0; signal-quality grades for EEG, GSR, and video; and per-modality predictions preserving each branch's raw label, confidence, and class probabilities before fusion. This schema is intentionally rich enough for Member 3 to explain both the final decision and the pre-fusion disagreement between branches without re-invoking either upstream model.

**Raw multimodal input (`SynchronizedInput`).** When all three raw signals are supplied together, a dataclass validates fixed shapes before any model runs: EEG (32 channels × 512 samples), GSR (512 samples after resampling to the EEG rate), and video (16 frames at 224×224 RGB). This enforces the four-second trial window agreed across members (128 Hz × 4 s = 512 samples; 16 uniformly sampled frames over the same window) and prevents shape mismatches from reaching the neural networks.

The field-by-field tables for the first two contracts appear in Chapter 6 (Tables 6.1 and 6.2); the explainability enrichment contract is summarised in Section 5.5.2 below.

#### 5.5.1 Temporal Synchronization Design

Although DEAP and CREMA-D were never recorded from the same subjects, the system assumes a **four-second analysis window** as the common temporal unit at inference time. This duration aligns with standard EEG epoch lengths used in affective-computing experiments and is long enough for sixteen video frames to capture expression dynamics without averaging away temporal structure.

Within that window, sampling rates are harmonised as follows. EEG is natively 128 Hz (512 samples). GSR is originally recorded at a lower rate in DEAP and is resampled to 128 Hz using polyphase resampling so that each sample index aligns with the EEG timeline. Video is represented as sixteen frames sampled uniformly across the same four seconds, each resized to 224×224 pixels. The recognition model consumes full frames; face detection runs in parallel only to grade video quality (Section 5.4), not to define the recognition crop.

This design trades strict frame-level synchronisation across datasets (impossible without paired recordings) for **schema-level synchronisation**: at fusion time both modalities present a prediction over the same semantic window and the same five-class taxonomy, which is sufficient for late fusion even when the underlying clips originate from different corpora (Section 2.4).

#### 5.5.2 Explainability Output Contract (Member 3)

Member 3 enriches `prediction_output` into `shap_output` for persistence and display. Beyond the fused fields already present, the explainability contract adds: signed SHAP values and absolute feature importance for EEG, GSR, and video; a faithfulness score in [0, 1]; signal-reliability flags derived from the quality grades; optional coalition values recording the fused winning-class probability under each physio/video subset (empty, physio-only, video-only, both); and a copy of per-modality predictions for audit. Storing coalition values alongside attributions allows the faithfulness metric and dashboard to verify explanations against the same fusion arithmetic used at prediction time, rather than recomputing an approximate surrogate.

---

## REPLACE — Section 5.7 (entire subsection)

### 5.7 Database and Web Application Design

The web application uses a relational schema of four tables, chosen so that authentication, prediction history, explainability artefacts, and chat context remain normalised but queryable from a single session identifier.

**Users** store credentials (hashed password, email, display name) and anchor all per-user data through a foreign key. **Sessions** record one row per prediction request: timestamp, fused emotion label, fused confidence, modality weights, signal-quality JSON, and the fused class-probability vector. Separating the session row from explainability artefacts allows a prediction to be listed in session history even while SHAP computation completes, and keeps the core prediction payload small for dashboard aggregation.

**SHAP logs** hold a one-to-one extension of a session: attribution values, feature importance magnitudes, faithfulness score, reliability flags, coalition audit JSON, and a snapshot of per-modality predictions. **Chat history** stores user and assistant messages keyed to both user and session, so follow-up questions refer to the same prediction context.

**Table 5.3: Database entities and primary stored fields**

| Entity | Primary purpose | Key stored fields |
|---|---|---|
| User | Authentication | email, password hash, display name |
| Session | Prediction record | emotion label, confidence, class probabilities, modality weights, signal quality |
| SHAP log | Explainability artefact | SHAP values, feature importance, faithfulness score, coalition values |
| Chat message | Conversational context | role (user/assistant), message text, session reference |

JWT access and refresh tokens protect all routes except registration and login. Access tokens are short-lived to limit exposure if intercepted; refresh tokens allow renewal without repeating password entry. File uploads (video required; EEG and GSR optional on the multimodal route) are validated for expected array shapes before inference, returning a client error rather than a silent coercion when shapes are wrong.

---

## REPLACE — Section 5.8 Summary (entire subsection)

### 5.8 Summary

This chapter presented the shared taxonomy and the reasoning behind requiring it, the signal-quality grading scheme including video-threshold recalibration, interface-contract validation and temporal synchronisation across modalities, the explainability output contract, the gated fusion design with its rejected alternatives (softmax versus L1 normalisation, additive versus multiplicative gating, early versus late fusion), and the database layout supporting predictions, SHAP artefacts, and chat history. The next chapter describes how each of these designs was implemented.
