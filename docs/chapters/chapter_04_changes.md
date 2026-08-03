# Chapter 4 — Changes Only

**Base document:** `docs/chapters/chapter_04_our_approach.md` (keep all unlisted sections unchanged).

Apply the replacements and insertions below in order.

---

## REPLACE — Section 4.4.2 Process (entire subsection)

#### 4.4.2 Process

Member 3's process has three layers: **Shapley attribution** over the fused prediction, **trust-aware conflict explanation** when modalities disagree, and **web delivery** through a FastAPI backend and React dashboard.

**Shapley attribution and faithfulness.** A custom two-modality Shapley explainer treats physiological and video branches as players in a four-coalition game (neither, physio-only, video-only, both). Because only two modalities fuse at the decision level, Shapley values are computed in closed form — no sampling — over the known gated-fusion function applied to each coalition's probability inputs. The physiological Shapley contribution is split into separate EEG and GSR attributions using quality-weighted decomposition, reflecting Member 1's joint processing of both signals. A faithfulness score then masks modalities in order of attribution magnitude and correlates attribution rank with confidence drop, quantifying whether the explanation tracks the prediction (Section 2.5.4).

**Modality-conflict explanation.** When physiological and video argmax labels differ, the conflict layer re-derives the full gate trace — entropy confidence [3], quality penalty α, gate scores, and L1-normalised weights — and produces a structured explanation of which modality the fused decision sided with and why. Counterfactuals over signal-quality tiers (for example, whether the fused label would change if EEG quality were good rather than degraded) make the fusion rule inspectable without inspecting upstream model weights. This layer operates purely on the fused prediction dictionary and shared gate arithmetic, so it runs even when upstream models are loaded lazily or replaced.

**Backend.** A FastAPI application exposes prediction, explainability, session history, chat, dashboard, and conflict routes. Table 4.1 summarises the primary endpoints a user or integrator interacts with; additional routes (`/predict/synthetic`, `/auth/refresh`, `/health`, conflict-specific paths) support development, token renewal, and conflict-only workflows without changing the core contract.

**Table 4.1: Primary FastAPI endpoints (Member 3 backend)**

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register` | User registration |
| POST | `/auth/login` | JWT access + refresh token issue |
| POST | `/predict/multimodal` | Video + optional EEG/GSR upload → fused prediction + SHAP |
| POST | `/predict/video` | Video-only prediction |
| GET | `/explain/{session_id}` | SHAP explanation for a stored session |
| GET | `/sessions` | List user sessions |
| GET | `/sessions/{session_id}` | Session detail |
| POST | `/chat` | LLM chatbot message (SHAP-grounded) |
| GET | `/chat/history/{session_id}` | Chat history for a session |
| GET | `/dashboard/summary` | Aggregated dashboard statistics |

The `/predict/multimodal` route validates uploaded EEG arrays against shape `(32, 512)` and GSR against `(512,)`; malformed shapes return HTTP 422. When EEG/GSR are omitted, the same pipeline executes video-only fusion so Member 3's explainability code path stays identical regardless of which modalities were present.

**Frontend.** A React dashboard renders an emotion-trend line chart, a SHAP bar chart, a session-history panel, a chatbot panel, and the **modality-conflict explanation panel** — Member 3's novel contribution — which surfaces gate-trace reasoning when branches disagree. EEG/GSR file inputs and a "Run Multimodal Fusion" action were added during system integration (Chapter 6) alongside the pre-existing video-only upload path.

Figure 4.3 shows this architecture end to end.

**[INSERT IMAGE HERE: `docs/report_assets/fig4_3_m3_pipeline.png`]**

**Figure 4.3: Member 3 explainability and web application architecture**

---

## REPLACE — Section 4.4.3 Output (entire subsection)

#### 4.4.3 Output

The module's primary output is a `shap_output` dictionary — the `prediction_output` enriched with `shap_values` (EEG, GSR, video), `feature_importance` (absolute attributions), `faithfulness_score`, `signal_reliability` flags derived from `signal_quality`, and optional `coalition_values` for audit — persisted to the `shap_logs` table and returned from prediction routes.

When modalities conflict, a separate conflict explanation payload adds `gate_trace`, `modality_shapley`, `agrees_with`, and quality counterfactuals. Chat responses are either LLM-generated (Anthropic API when configured, with SHAP numbers injected into the prompt) or served from a template fallback when offline.

---

## REPLACE — Section 4.4.4 Evaluation (entire subsection)

#### 4.4.4 Evaluation

The web application was evaluated through a full in-process HTTP test against an isolated test database: registration and login; multimodal prediction with real video, EEG, and GSR uploads (HTTP 200, both modality weights strictly positive, populated SHAP and faithfulness fields); video-only prediction with physiological inputs omitted; malformed EEG shape rejected with HTTP 422; and chat history retrieval for a session with a prior prediction.

Conflict explanation was verified on synthetic prediction fixtures where physiological and video labels intentionally disagree, confirming that the panel reports the correct trusted modality and reproduces fusion weights to numerical tolerance. The frontend production build was confirmed clean with multimodal upload controls present. Full results are in Section 7.5.

---

## REPLACE — Section 4.5 Summary (final paragraph only)

Replace the last sentence of Section 4.5:

**Old:**
> The next chapter presents the system's analysis and design — the shared taxonomy, interface contracts, signal-quality thresholds and fusion formulation — that make these three independently-developed approaches compose into a single system.

**New:**
> The next chapter presents the system's analysis and design — the shared taxonomy, interface contracts, signal-quality thresholds, and fusion formulation — that make these three independently-developed approaches compose into a single system. Member 3's explainability and conflict layers are tied explicitly to that fusion design: they consume the same `prediction_output` schema and gate arithmetic rather than re-implementing trust logic independently.

---

## Notes (no chapter text — for your Word pass)

1. **Table 4.1 is new** in Chapter 4. Renumber if Table 4.x conflicts with your global table list; alternatively move this table to Chapter 6 as an expanded Table 6.3 and leave a cross-reference here — but only one canonical table should exist in the final report.
2. **No change required** to Sections 4.2–4.3 unless you want Table 4.1 duplicated in Chapter 6 — those sections in the base draft are already accurate.
3. **SHAP wording:** Base draft says "Kernel SHAP explainability layer" — the replacement clarifies it is a **custom closed-form Shapley** implementation following Kernel SHAP principles, not a call to the `shap` library at inference time.
