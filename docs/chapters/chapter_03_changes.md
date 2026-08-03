# Chapter 3 — Changes Only

**Base document:** `docs/chapters/chapter_03_technology_adapted.md` (keep all unlisted sections unchanged).

Apply the replacements and insertions below in order.

---

## REPLACE — Section 3.1 Introduction (entire subsection)

### 3.1 Introduction

This chapter describes the development tools, programming languages, libraries, dependency management, and version-control practices used to implement MedOracle. Rather than presenting a bare technology inventory, each choice is explained in terms of the engineering constraint it addressed — GPU availability, module independence, asynchronous I/O, cross-session training limits, or the need to run the same codebase against SQLite in development and PostgreSQL in production.

---

## INSERT — New Section 3.4.0 (immediately before current Section 3.4, after Section 3.3)

### 3.4 Shared Foundation and Dependency Management

All three modules depend on a common `shared/` package (`label_harmonization.py`, `data_contracts.py`) and a shared NumPy/SciPy foundation (`requirements_shared.txt`). Member-specific dependencies are split into three requirement files — `requirements_m1.txt`, `requirements_m2.txt`, and `requirements_m3.txt` — so each member can install only what their module needs while still passing the shared interface-contract tests in `tests/`. System integration installs the union of all three files plus the shared base.

This split was deliberate: Member 1's DEAP preprocessing stack (MNE, pyedflib, h5py) is not required to run Member 3's FastAPI server, and Member 2's Ultralytics YOLO dependency is not required to execute Member 1's LOSO training script. Keeping dependencies module-scoped reduced environment conflicts during parallel development and made it possible to validate each module's contract compliance independently before full pipeline integration (Chapter 6).

---

## REPLACE — Section 3.4.1 Member 1 (entire subsection)

#### 3.4.1 Member 1 — Physiological Signal Processing and Deep Learning

PyTorch was used to implement the bidirectional cross-modal attention network, chosen for flexibility in defining custom attention layers and for interactive debugging during the LOSO diagnostic work (Chapter 7). scikit-learn supplied the SVM, Random Forest, and MLP baselines, class-weight utilities, and cross-validation helpers (`GroupKFold`, stratified splitting) used for the 32-fold Leave-One-Subject-Out protocol. SciPy provided Welch power spectral density estimation for the Differential Entropy features compared in Section 7.2.

DEAP-specific preprocessing used **MNE** for EEG bandpass filtering and referencing, **pyedflib** and **h5py**/**mat73** for reading the dataset's native file formats, and **pandas** for tabular handling of trial metadata and label mappings. These libraries were chosen because DEAP's release formats (.dat pickles, HDF5 `.mat` exports, and legacy MATLAB structures) are not handled uniformly by a single reader; isolating format-specific loading in `deap_loader.py` kept the training code independent of how the raw files were obtained.

---

## REPLACE — Section 3.4.2 Member 2 (entire subsection)

#### 3.4.2 Member 2 — Video Processing, Deep Learning and Evaluation

PyTorch and **torchvision** provided the pretrained ResNet50 backbone for partial fine-tuning (Chapter 6). **Ultralytics YOLOv8** was used for face detection — during early data preparation and, in the shipped system, exclusively for video signal-quality grading (Chapter 5), not for cropping the recognition model's input. **OpenCV** and **Pillow** handled frame decoding, resizing, and augmentation (including cutout). **pandas** and **NumPy** supported dataset indexing, actor-grouped splits, and construction of the cross-dataset synchronised evaluation set (Chapter 6).

SciPy's **Wilcoxon signed-rank** test was reused — via a small shared utility — to assess statistical significance in the C1–C8 ablation study (Chapter 7). **pytest** was used for fusion and pipeline unit tests (18 parametrized fusion tests against Member 3's reference implementation).

---

## REPLACE — Section 3.4.3 Member 3 (entire subsection)

#### 3.4.3 Member 3 — Explainability and Web Application

**Custom Shapley attribution (`kernel_shap.py`).** The production explainability path implements a closed-form two-modality Shapley explainer over the fused `prediction_output` — enumerating all four physio/video coalitions exactly, then splitting the physiological Shapley value into EEG and GSR contributions using quality-weighted attribution. This custom implementation was chosen because late fusion exposes a fixed, known combining function at the module boundary; exact Shapley values can therefore be computed without Monte Carlo sampling and without importing either Member 1's or Member 2's model internals. The `shap` Python package is listed in `requirements_m3.txt` for compatibility with exploratory attribution work (`attribution_engine.py`), but the deployed `/predict` and `/explain` paths use the custom explainer.

**FastAPI + Uvicorn.** The backend uses FastAPI for async request handling, Pydantic v2 schema validation, and auto-generated OpenAPI documentation. Uvicorn serves the ASGI application in development and test environments.

**Authentication and persistence.** **python-jose** issues and verifies JWT access/refresh tokens; **passlib[bcrypt]** hashes passwords. **SQLAlchemy 2.x** (async) maps the four-table schema (`users`, `sessions`, `shap_logs`, `chat_history`) with **aiosqlite** for development and **psycopg2-binary** for PostgreSQL in production. **Alembic** supports schema migrations when moving beyond the auto-created SQLite tables used during integration testing.

**LLM and chat.** The chatbot uses the **Anthropic** client when an API key is configured; **httpx** supports async HTTP calls. When no key is present, a deterministic template fallback in `llm_client.py` keeps the system runnable offline. A separate RAG-grounded generator in `llm_chatbot.py` (curated knowledge base + numeric faithfulness check) is used for conflict-aware explanations.

**Frontend.** **React** (Vite scaffold), **Axios**, and **Recharts** implement the dashboard panels (emotion trend, SHAP bar chart, session history, conflict explanation, chat).

**Testing.** **pytest** and **pytest-asyncio** drive the in-process FastAPI HTTP tests described in Section 7.5.

---

## INSERT — New Table 3.1 (after Figure 3.1, before Section 3.4.1)

**Table 3.1: Primary technologies by module**

| Layer | Member 1 | Member 2 | Member 3 | Shared |
|---|---|---|---|---|
| Language | Python | Python | Python + JavaScript (React) | Python |
| Deep learning | PyTorch | PyTorch, torchvision | — | — |
| Signal / video I/O | MNE, pyedflib, h5py | OpenCV, Ultralytics YOLOv8 | — | NumPy, SciPy |
| ML utilities | scikit-learn | scikit-learn | — | — |
| Explainability | — | — | Custom Shapley (`kernel_shap.py`), faithfulness perturbation | — |
| Backend | — | — | FastAPI, Uvicorn, SQLAlchemy, python-jose, passlib | Pydantic (via FastAPI) |
| Database | — | — | SQLite (dev), PostgreSQL (prod) | — |
| Frontend | — | — | React, Vite, Axios, Recharts | — |
| Testing | pytest | pytest | pytest, pytest-asyncio | pytest (contract + fusion tests) |
| GPU training | Kaggle Notebooks | Kaggle Notebooks | — | — |

---

## REPLACE — Section 3.5 (entire subsection)

### 3.5 Version Control System and Collaborative Workflow

Git and GitHub were used for source control. Each member developed on a feature branch scoped to their module, opening pull requests into a shared integration branch once interface-contract tests passed locally. Because the three modules communicate only through `shared/data_contracts.py` (Chapter 5), a member could merge preprocessing or model changes without touching the other members' packages, provided the exported dictionaries still validated.

Integration testing (`tests/test_pipeline.py`, `tests/test_fusion.py`, `tests/test_label_harmonization.py`) ran on the integration branch before demo deployments. The final codebase maintains **82 automated tests** (59 label-harmonisation, 18 fusion, 5 pipeline), all passing at submission time — giving a regression guard across module boundaries that prose interface contracts alone would not provide.

---

## REPLACE — Section 3.6 Summary (entire subsection)

### 3.6 Summary

This chapter described the tools, languages, libraries, dependency layout, and collaborative workflow used to build MedOracle, explaining why each choice matched the problem it solved — from Kaggle's 12-hour session cap shaping checkpoint-resume training, to module-scoped requirement files enabling parallel development, to a custom Shapley explainer operating on the fusion boundary rather than inside either upstream model. The next chapter describes each member's individual approach in terms of input, process, output, and evaluation.
