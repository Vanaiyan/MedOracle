<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 3 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 3" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 3" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "3.4 Libraries and Frameworks"): 12pt, bold.
- Subsection headings (e.g. "3.4.1 Member 1 ..."): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- Bulleted lists: use Word's standard bullet list style, still Times New
  Roman 12pt, 1.5 line spacing.
- There are no tables in this chapter. One figure is referenced as an image
  file path in this document — insert the actual image at that path,
  centered, with the caption "Figure 3.1: <caption text>" centered directly
  below it in bold:
    Figure 3.1: docs/report_assets/fig3_1_tech_stack.png
- This chapter does not use any citation markers ([n]) — that is
  intentional (technology/tooling choices are not typically literature-
  cited), so do not add any.
- If a Chapter 3 already exists in the document, REPLACE its content with
  this; otherwise insert it immediately after Chapter 2 and before
  Chapter 4 (if Chapter 4 already exists).
- Do not alter the numbering, content, or formatting of any other chapter.

=========================================================================
-->

# Chapter 3

## Technology Adapted

### 3.1 Introduction

This chapter describes the development tools, programming languages, libraries and version-control practices used to implement MedOracle, and — in keeping with the intent of this chapter rather than a bare inventory — explains why each choice was appropriate to the specific problem each module needed to solve, rather than listing technologies for their own sake.

### 3.2 Development Tools and IDEs

Kaggle Notebooks were used for all GPU-accelerated training: both the video model's ResNet50+BiLSTM training and the physiological model's 32-fold Leave-One-Subject-Out training required more compute than was available locally, and Kaggle's free GPU tier made this feasible. Kaggle sessions are capped at 12 hours, which is shorter than a full 32-fold LOSO training run or a multi-epoch video training run; this constraint directly shaped the implementation (Chapter 6), which persists model checkpoints across sessions and automatically resumes training from the last completed fold or epoch rather than assuming a single uninterrupted session.

Visual Studio Code was used as the primary IDE across all three modules' backend, frontend and model-development code, chosen for its strong Python and JavaScript tooling within a single environment, which suited a project where the same codebase spans PyTorch model code, a FastAPI backend, and a React frontend.

Claude Code (a command-line AI coding assistant) was used throughout the project as an iterative pair-programming aid — for debugging the video model's training instability, for scripting and running the evaluation and ablation study described in Chapter 7, and for the system-integration work that composes the three modules' independently-developed code into a single working pipeline.

### 3.3 Programming Languages

Python was used for all machine-learning, signal-processing and backend development, chosen for the maturity of its deep-learning ecosystem (PyTorch, scikit-learn, SciPy) and because all three team members' modules — physiological modelling, video modelling, and the FastAPI backend — could share a single language and a common shared package (Chapter 5) for the interface contracts and label taxonomy that tie the modules together. JavaScript, with React and JSX, was used for the frontend web application, where its component model suited the dashboard's multiple independent but data-linked panels (session history, SHAP chart, emotion trend, chat).

### 3.4 Libraries and Frameworks

Figure 3.1 summarises the libraries adopted by each member, grouped by ownership, sitting on top of the shared foundation described in Section 3.5.

**[INSERT IMAGE HERE: `docs/report_assets/fig3_1_tech_stack.png`]**

**Figure 3.1: Technology stack by module ownership**

#### 3.4.1 Member 1 — Physiological Signal Processing and Deep Learning

PyTorch was used to implement the bidirectional cross-modal attention network, chosen over alternative frameworks for its flexibility in defining custom attention mechanisms and its ease of debugging during the iterative process of diagnosing the network's Leave-One-Subject-Out performance (Chapter 7). scikit-learn provided the classical SVM, Random Forest and Logistic Regression baselines used specifically to establish whether a given evaluation result reflected a limitation of the neural architecture or of the underlying data, together with cross-validation utilities (StratifiedKFold, GroupKFold) needed to implement Leave-One-Subject-Out and actor-independent evaluation correctly. SciPy's Welch power spectral density implementation was used to compute the Differential Entropy features investigated as part of this diagnostic work (Section 7.2), and MNE-adjacent signal-processing routines were used for EEG bandpass filtering and referencing during preprocessing.

#### 3.4.2 Member 2 — Video Processing, Deep Learning and Evaluation

PyTorch and torchvision were used for the video model, providing a pretrained ResNet50 backbone that could be partially fine-tuned (Chapter 6) rather than trained from scratch, which was necessary given the comparatively modest size of the CREMA-D training set relative to the scale typically needed to train a deep convolutional network from random initialisation. Ultralytics' YOLOv8 was used for face detection — both during earlier data-preparation work and, in the final system, purely to grade video signal quality (Chapter 5) rather than to crop the recognition model's input, following the empirical finding described in Chapter 6 that cropping reduced recognition accuracy. OpenCV was used for video frame decoding and image processing throughout. SciPy's Wilcoxon signed-rank implementation, reused via a shared utility originally written for Member 1's evaluation, was used to test the statistical significance of the ablation study's results (Chapter 7), and NumPy and pandas were used throughout for array and tabular data handling, including the construction of the cross-dataset synchronised evaluation set (Chapter 6).

#### 3.4.3 Member 3 — Explainability and Web Application

The SHAP library was used to implement the Kernel SHAP explainability layer over the fused prediction, chosen because it provides a model-agnostic attribution mechanism that could be applied directly to the fused prediction_output dictionary without requiring access to either Member 1's or Member 2's model internals — an important property given that the three modules are developed independently and communicate only through the shared interface contracts (Chapter 5). FastAPI was used for the backend web framework because of its native asynchronous request handling (a good fit for I/O-bound operations such as database access and the LLM-backed chat endpoint), its built-in Pydantic-based request/response validation, and its automatic interactive API documentation, which eased integration testing against the nine-endpoint API surface (Chapter 6). SQLAlchemy's asynchronous ORM was used over the four-table schema (users, sessions, SHAP logs, chat history), chosen specifically because it allows the same code to run against SQLite in development and PostgreSQL in production without modification. python-jose provided JWT access and refresh token handling for authentication. On the frontend, React (scaffolded with Vite for fast local iteration) was used together with Axios for HTTP requests and Recharts for the dashboard's emotion-trend line chart and SHAP bar chart.

### 3.5 Version Control System and Collaborative Workflow

Git and GitHub were used throughout the project for source control. Each member developed their module on a dedicated branch, merging into a shared develop branch once a module reached a stable state, with pull-request-based review before merging into the branch used for system integration. Because the three modules communicate only through the interface contracts defined in shared/data_contracts.py (Chapter 5), each member's branch could be developed and tested largely independently of the others' implementation details, and integration work (Chapter 6) consisted primarily of composing already-tested modules behind a shared contract rather than resolving deep interdependencies at merge time.

### 3.6 Summary

This chapter described the tools, languages, libraries and collaborative workflow used to build MedOracle, and explained why each was appropriate to the specific problem it was applied to — from Kaggle's GPU access and session-limit constraints shaping the training pipeline's checkpoint-resume design, to SHAP's model-agnostic attribution suiting a system whose three modules are developed independently. The next chapter describes each member's individual approach in terms of input, process, output and evaluation.
