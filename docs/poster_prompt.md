<!--
This is the prompt to paste into Claude Desktop (with the Canva connector
enabled) to generate the MedOracle exhibition poster. Copy everything
between the "===== PROMPT START =====" and "===== PROMPT END =====" lines
and paste it as your message to Claude Desktop.
-->

===== PROMPT START =====

Create an academic final-year-project exhibition poster in Canva using the content and structure below. Follow the University of Moratuwa poster guidelines exactly.

## POSTER SPECS
- Size: A2 (420mm x 594mm), portrait orientation
- Clean, professional, well-organized layout
- One consistent colour scheme and font family throughout (suggest a navy/teal academic palette — e.g. dark navy #0D1B2A, teal #028090, teal-green accent #02C39A, light background #F2F8FA — but you can propose your own consistent scheme)
- Consistent heading structure (all section headers same style/size)
- All body content as bullet points, NOT paragraphs — keep every bullet short (one line where possible)
- All text must be legibly readable from ~1.5-2m viewing distance (exhibition setting)
- Every image/diagram must be properly labelled with a short caption
- Use my own uploaded image files for all diagrams/charts listed below (do not regenerate them from scratch) — upload and place each file at the exact file path given

## LAYOUT STRUCTURE (mirror this exact section order)

**1. Header band (top, full width):**
- Project Title (large, bold): "AI-Driven Multimodal Emotional Recognition System with Explainability"
- Group Name: MedOracle
- Group Number: 36

**2. Row 2 — two columns side by side:**
- Left column: "Problem in Brief" (bullets)
- Right column: "Aim & Objectives" (Aim as one sentence, Objectives as bullets), with the system architecture diagram placed to the right or below it

**3. Row 3 — three equal columns, one per module** (Module 01 / Module 02 / Module 03), each containing, top to bottom:
   - Module title + one-line description
   - "Methodology" — the module's pipeline diagram
   - "Evaluation & Results" — the module's chart(s) + 2-3 headline numbers as bold bullet stats
   - "Key Features" — 2-3 short bullets

**4. Row 4 — full width:** "Conclusion" (short paragraph or 3-4 bullets) + "Key Results / Outcomes" (bold headline stats, large text, visually prominent — this is what evaluators should notice first when walking past)

**5. Footer band (bottom, full width, smaller text):**
- University of Moratuwa
- Faculty of Information Technology
- BSc (Hons) in Information Technology
- Supervisor: Dr. Firdhous M.F.M.
- Group Members:
  - 214206G — Suhira Balarajan
  - 214215H — Vanaiyan Kirupagaran
  - 214024V — Adshaya Balarajah

## CONTENT

### Problem in Brief (bullets)
- Mental health assessment still relies mainly on subjective self-report; continuous, objective measurement is limited
- Single-modality tools (a camera OR a wearable, never both) inherit that one channel's blind spots — a face can be controlled, physiology can't, but physiology varies heavily between people
- Existing multimodal systems assume every input is always present and equally reliable, and fuse with fixed weights that can't adapt when a sensor is degraded or missing
- The two best available datasets for physiological (DEAP) and facial-video (CREMA-D) emotion recognition share no common subjects — genuine multimodal fusion can't be trained or evaluated the usual way

### Aim
To design, implement and rigorously evaluate a multimodal emotion recognition system that fuses physiological (EEG/GSR) and facial-video signals through a quality-aware gated fusion mechanism, and to make its predictions explainable through SHAP.

### Objectives (bullets)
- Physiological emotion recognition via a bidirectional cross-modal attention network (DEAP dataset)
- Video emotion recognition via ResNet50 + BiLSTM, evaluated actor-independently (CREMA-D)
- A quality-aware gated fusion mechanism with graceful degradation, verified by automated testing
- SHAP-based explainability + a FastAPI/React web app accepting real multimodal input
- An 8-condition ablation study with statistical significance testing — reporting all results honestly, including unfavourable ones

### System Architecture diagram
Upload and place: `docs/report_assets/fig_architecture.png`
Caption: "System architecture — physiological and video branches fused via quality-aware gating, explained via SHAP"

---

### MODULE 01 — Physiological Emotion Recognition
One-line description: Bidirectional cross-modal attention network over EEG (32-channel) and GSR, trained on DEAP, evaluated honestly under Leave-One-Subject-Out cross-validation.

Methodology diagram — upload and place: `docs/report_assets/fig4_1_m1_pipeline.png`

Evaluation & Results:
- Upload and place chart: `docs/report_assets/fig7_3_deap_distribution.png` (caption: "DEAP class distribution — severe imbalance, only 2.6% happy")
- Headline stats (bold, large):
  - LOSO macro-F1 (honest, subject-independent): **0.179**
  - Random-split macro-F1 (optimistic baseline): 0.417
- Bullet: Root cause diagnosed via 4 controlled experiments — traced to class imbalance and cross-subject EEG variability, NOT a feature-engineering or implementation defect

Key Features (bullets):
- Bidirectional cross-modal attention (EEG <-> GSR)
- Signal-quality grading (good/degraded/poor) feeding directly into system-level fusion
- Honest dual-protocol evaluation (random-split vs LOSO)

---

### MODULE 02 — Video Emotion Recognition + Gated Fusion
One-line description: YOLOv8 face detection (quality grading only) -> partially fine-tuned ResNet50 -> BiLSTM -> 5-class softmax, fused with Module 01's output via an adaptive, confidence-and-quality-aware gate.

Methodology diagram — upload and place: `docs/report_assets/fig4_2_m2_pipeline.png`

Evaluation & Results:
- Upload and place chart: `docs/report_assets/fig_ablation_bar.png` (caption: "Ablation study — fusion (C5) significantly beats both single modalities")
- Headline stats (bold, large):
  - Video held-out macro-F1: **0.6486** (best fold: 0.6615)
  - Fusion vs physio-only / video-only: **p < 0.001** (Wilcoxon, both significant)
- Bullet: Full-frame input outperforms YOLO-cropped face input (0.70 vs 0.56 macro-F1) — a genuine empirical finding from this project

Key Features (bullets):
- Actor-independent cross-validation (no actor overlap train/test)
- Entropy-based confidence x signal-quality penalty -> L1-normalised fusion weights
- Graceful degradation: falls back cleanly to a single modality when the other is missing/poor (verified by dedicated robustness tests)

---

### MODULE 03 — Explainability + Web Application
One-line description: Kernel SHAP over the fused prediction, served through a FastAPI backend (9 endpoints, JWT auth) and a React dashboard with a novel modality-conflict explanation panel and an LLM-backed chatbot.

Methodology diagram — upload and place: `docs/report_assets/fig4_3_m3_pipeline.png`

Evaluation & Results:
- Upload and place: `docs/report_assets/mockup_dashboard_placeholder.png` — THIS IS A MOCKUP, NOT A REAL SCREENSHOT (it is clearly labelled as such with a red "UI MOCKUP" banner). Place it exactly as-is, including its red border/banner — do not crop the banner out or present it as a genuine screenshot. I will replace this file with a real screenshot of the running application before the poster is printed.
- Headline stats (bold, large):
  - **82 automated tests** passing (fusion, pipeline, label harmonisation, full HTTP integration)
  - **9 REST API endpoints**, real multimodal (video + EEG + GSR) upload supported end-to-end

Key Features (bullets):
- Kernel SHAP + faithfulness scoring (perturbation-based)
- Modality-conflict panel — explains disagreements between physio and video using the same reasoning the fusion gate used internally
- Real file upload: video + EEG + GSR -> fused, explained prediction

---

### Key Results / Outcomes (full-width row, make this visually prominent — large bold numbers)
- Video module generalises to unseen actors: **0.6486 macro-F1**, near-zero train/test gap
- Multimodal fusion significantly outperforms either single modality (**p < 0.001**, Wilcoxon signed-rank)
- System degrades gracefully under missing/poor signal quality — verified, not assumed
- Physiological module's honest limitation (LOSO 0.179) is diagnosed with evidence, not hidden — four controlled experiments trace it to DEAP's class imbalance and cross-subject EEG variability
- **82/82 automated tests passing**; full working web application with real multimodal upload

### Conclusion
MedOracle demonstrates a working, honestly-evaluated multimodal emotion recognition system. Its quality-aware gated fusion significantly outperforms any single modality and degrades gracefully when a sensor is unreliable — precisely the scenario a real deployment must handle. Where a component (the physiological module, under strict subject-independent evaluation) falls short, this project treats that as a rigorously diagnosed finding rather than a hidden weakness, and shows the system is explicitly designed to remain robust to it.

## IMPORTANT NOTES FOR YOU (Claude Desktop)
- All image file paths above are relative to the project root: `/Users/vanaiyan/Documents/Claude/Projects/FYP/`
- Upload and place the real images at those exact paths — do not draw your own replacement diagrams for anything with a given file path
- The one placeholder (Module 03's app screenshot) should be a clearly labelled empty box, not a fabricated screenshot
- Proofread all text for spelling/grammar/consistency before finalising
- Remind me to bring my own poster stand/easel — the university does not provide one and it cannot be taped to any university property

===== PROMPT END =====
