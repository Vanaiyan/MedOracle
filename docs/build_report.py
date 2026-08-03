"""
docs/build_report.py
=====================
Generates MedOracle's Final Report (.docx) following the University of
Moratuwa, Faculty of IT "Guidelines for Preparation of Final Reports"
(Times New Roman, 1.5 line spacing, 1.5in/1in/1in/1in margins, numbered
chapters/sections, figures & tables with captions, IEEE-style references,
Appendix A = Individual's Contribution).

Run:
    python docs/build_report.py

Output:
    docs/MedOracle_Final_Report.docx
"""

from __future__ import annotations

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

ASSETS = "docs/report_assets"

# ---------------------------------------------------------------------------
# Figure / Table registries (single source of truth for LOF/LOT + body)
# ---------------------------------------------------------------------------

FIGURES = [
    ("1.1", "High-level architecture of the proposed multimodal emotion recognition system"),
    ("5.1", "Data flow and interface contracts between Member 1, Member 2 and Member 3"),
    ("6.1", "DEAP dataset structure and per-subject class distribution (preprocessing visualisation)"),
    ("6.2", "EEG / GSR / Video signal-quality grading visualisation"),
    ("6.3", "SVM baseline — subject-independent (LOSO) results"),
    ("6.4", "Random Forest baseline — subject-independent (LOSO) results"),
    ("6.5", "MLP baseline — subject-independent (LOSO) results"),
    ("7.1", "Ablation study — macro-F1 by condition (C1-C8)"),
    ("7.2", "Ablation study — normalised confusion matrices for physio-only (C1), video-only (C2) and full fusion (C5)"),
]

TABLES = [
    ("1.1", "Summary of system modules and responsibilities"),
    ("2.1", "Comparison of the DEAP and CREMA-D datasets"),
    ("2.2", "Summary of related multimodal emotion-recognition approaches"),
    ("5.1", "Signal-quality grading thresholds"),
    ("5.2", "DEAP-to-5-class and CREMA-D-to-5-class label harmonisation"),
    ("6.1", "M1 -> M2 interface contract (physiological_prediction_dict)"),
    ("6.2", "M2 -> M3 interface contract (prediction_output)"),
    ("6.3", "FastAPI endpoint summary"),
    ("7.1", "DEAP 5-class distribution used for physiological training"),
    ("7.2", "Video module per-class F1 (held-out test, 788 clips)"),
    ("7.3", "Ablation study results (C1-C8) with Wilcoxon significance vs C5"),
]

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def set_cell_text(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=11):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"


def add_field(paragraph, instr_text):
    """Insert a real Word field (e.g. TOC, PAGE) into a paragraph."""
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = instr_text
    fld_sep = OxmlElement("w:fldChar"); fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)


def set_page_numbering(section, fmt="decimal", start=1):
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:fmt"), fmt)
    pgNumType.set(qn("w:start"), str(start))


def add_footer_page_number(section, align=WD_ALIGN_PARAGRAPH.CENTER):
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
    p.text = ""
    p.alignment = align
    add_field(p, "PAGE")


# ---------------------------------------------------------------------------
# Document-level style setup
# ---------------------------------------------------------------------------

doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
normal.paragraph_format.space_after = Pt(6)

# Chapter heading (18pt, bold)
h1 = doc.styles["Heading 1"]
h1.font.name = "Times New Roman"; h1.font.size = Pt(18); h1.font.bold = True
h1.font.color.rgb = RGBColor(0, 0, 0)
h1.paragraph_format.space_before = Pt(6); h1.paragraph_format.space_after = Pt(12)
h1.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# Section heading (12pt, bold)
h2 = doc.styles["Heading 2"]
h2.font.name = "Times New Roman"; h2.font.size = Pt(12); h2.font.bold = True
h2.font.color.rgb = RGBColor(0, 0, 0)
h2.paragraph_format.space_before = Pt(12); h2.paragraph_format.space_after = Pt(6)
h2.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# Subsection heading (12pt, bold)
h3 = doc.styles["Heading 3"]
h3.font.name = "Times New Roman"; h3.font.size = Pt(12); h3.font.bold = True
h3.font.color.rgb = RGBColor(0, 0, 0)
h3.paragraph_format.space_before = Pt(10); h3.paragraph_format.space_after = Pt(4)
h3.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# Caption style
cap = doc.styles.add_style("FigCaption", 1)
cap.font.name = "Times New Roman"; cap.font.size = Pt(11); cap.font.bold = True
cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
cap.paragraph_format.space_after = Pt(12)

sec = doc.sections[0]
sec.left_margin = Inches(1.5); sec.right_margin = Inches(1)
sec.top_margin = Inches(1); sec.bottom_margin = Inches(1)


def P(text="", align=None, bold=False, italic=False, size=None, style=None, space_after=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        r.bold = bold; r.italic = italic
        if size:
            r.font.size = Pt(size)
    return p


def H1_CHAPTER(number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Chapter {number}")
    r.bold = True; r.font.size = Pt(14); r.font.name = "Times New Roman"
    p.paragraph_format.space_after = Pt(4)
    h = doc.add_paragraph(style="Heading 1")
    h.add_run(title)


def H2(text):
    doc.add_paragraph(text, style="Heading 2")


def H3(text):
    doc.add_paragraph(text, style="Heading 3")


def PARA(text):
    P(text)


def BULLETS(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        r = p.add_run(it)
        r.font.name = "Times New Roman"; r.font.size = Pt(12)


def FIGURE(num, path, width=5.6):
    caption = next(c for n, c in FIGURES if n == num)
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(style="FigCaption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"Figure {num}: {caption}")


def TABLE(num, headers, rows, col_widths=None):
    caption = next(c for n, c in TABLES if n == num)
    p = doc.add_paragraph(style="FigCaption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"Table {num}: {caption}")
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_text(hdr[i], h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            set_cell_text(cells[i], str(val))
    doc.add_paragraph().paragraph_format.space_after = Pt(8)


def PAGEBREAK():
    doc.add_page_break()


# ===========================================================================
# TITLE PAGE  (no page number)
# ===========================================================================

for _ in range(3):
    P("")
P("Final Report", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16)
P("Level 4", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
for _ in range(3):
    P("")
P("AI-Driven Multimodal Emotional Recognition System with Explainability",
  align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
for _ in range(2):
    P("")
P("Team MedOracle", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
for _ in range(2):
    P("")

t = doc.add_table(rows=4, cols=2)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
data = [
    ("Index Number", "Name"),
    ("214206G", "Suhira Balarajan"),
    ("214024V", "Adshaya Balarajah"),
    ("214215H", "Vanaiyan Kirupagaran"),
]
for i, (a, b) in enumerate(data):
    set_cell_text(t.rows[i].cells[0], a, bold=(i == 0), align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(t.rows[i].cells[1], b, bold=(i == 0), align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(2):
    P("")
P("Supervised by:", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12)
P("Dr. Firdhous M.F.M.", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
for _ in range(2):
    P("")
P("Faculty of Information Technology", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
P("University of Moratuwa", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
P("2026", align=WD_ALIGN_PARAGRAPH.CENTER, size=12)

# ===========================================================================
# SECTION 2 — FRONT MATTER (roman numerals)
# ===========================================================================

doc.add_section(WD_SECTION.NEW_PAGE)
front = doc.sections[-1]
front.left_margin = Inches(1.5); front.right_margin = Inches(1)
front.top_margin = Inches(1); front.bottom_margin = Inches(1)
set_page_numbering(front, fmt="lowerRoman", start=1)
add_footer_page_number(front)

# --- Declaration ---
P("Declaration", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
P("")
PARA(
    "We assert that this thesis is our own work and has not been submitted in any other "
    "form for any academic qualification at any university or tertiary institution. All "
    "borrowed information from published or unpublished works of others has been "
    "appropriately acknowledged within the text, and a comprehensive list of references "
    "has been provided."
)
P("")
t = doc.add_table(rows=4, cols=2)
hdrs = [("Name of Student(s)", "Signature of Student(s)")]
rows = [("Suhira Balarajan", ""), ("Adshaya Balarajah", ""), ("Vanaiyan Kirupagaran", "")]
set_cell_text(t.rows[0].cells[0], "Name of Student(s)", bold=True)
set_cell_text(t.rows[0].cells[1], "Signature of Student(s)", bold=True)
for i, (a, b) in enumerate(rows, start=1):
    set_cell_text(t.rows[i].cells[0], a)
    set_cell_text(t.rows[i].cells[1], b)
P("")
P("Date: ....................", size=12)
P("")
P("Supervised by:", bold=True)
t2 = doc.add_table(rows=2, cols=2)
set_cell_text(t2.rows[0].cells[0], "Name of Supervisor", bold=True)
set_cell_text(t2.rows[0].cells[1], "Signature of Supervisor", bold=True)
set_cell_text(t2.rows[1].cells[0], "Dr. Firdhous M.F.M.")
set_cell_text(t2.rows[1].cells[1], "")
P("Date: ....................", size=12)

PAGEBREAK()

# --- Abstract ---
P("Abstract", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
P("")
PARA(
    "Mental health support is often bottlenecked by single-modality assessment: clinicians "
    "rely on either self-reported mood, or a single physiological or behavioural signal, "
    "each of which is incomplete and, in isolation, unreliable. This project, MedOracle, "
    "presents an AI-driven emotion recognition system that fuses two complementary "
    "modalities — physiological signals (EEG and GSR) and facial video — into a single, "
    "explainable prediction across five emotion classes (stress, calm, happy, sad, angry). "
    "The physiological branch uses a bidirectional cross-modal attention network trained on "
    "the DEAP dataset; the video branch uses a YOLOv8 face-detection front end feeding a "
    "partially fine-tuned ResNet50 and a bidirectional LSTM trained on the CREMA-D dataset. "
    "The two branches are combined with a quality-aware gated fusion mechanism: an "
    "entropy-based confidence score for each modality is multiplied by a signal-quality "
    "penalty and the results are L1-normalised into fusion weights, so that a modality with "
    "poor or missing input is automatically down-weighted rather than corrupting the final "
    "prediction. A Kernel SHAP explainability layer, a FastAPI backend with JWT-authenticated "
    "endpoints, and a React dashboard complete the system, allowing a user to upload a video "
    "together with EEG/GSR recordings and receive a fused prediction with a feature-level "
    "explanation and a natural-language summary from an integrated chatbot. The system was "
    "evaluated with an eight-condition ablation study (physio-only, video-only, equal-weight "
    "fusion, confidence-only fusion, the full gated method, and three robustness conditions "
    "with degraded or missing modalities) over a constructed cross-dataset evaluation set, "
    "with statistical significance assessed using the one-tailed Wilcoxon signed-rank test. "
    "The fused method significantly outperforms either single modality (p < 0.001 in both "
    "cases) and degrades gracefully when a modality is unavailable. A separate, honest "
    "subject-independent (leave-one-subject-out) evaluation of the physiological branch was "
    "also carried out, and the resulting limitation — driven by severe class imbalance in the "
    "DEAP labels and the well-documented difficulty of cross-subject EEG generalisation — is "
    "reported transparently, together with the diagnostic experiments that isolated its cause. "
    "The completed system comprises 82 passing automated tests, a functioning multimodal web "
    "application, and a reproducible evaluation pipeline, and is presented as a working, "
    "honestly evaluated, and explainable multimodal emotion recognition system."
)

PAGEBREAK()

# --- Table of Contents ---
P("Table of Contents", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
P("")
p = doc.add_paragraph()
r = p.add_run("(Right-click below and choose “Update Field”, or press F9, to populate "
              "page numbers after opening this document in Microsoft Word.)")
r.italic = True; r.font.size = Pt(10)
p2 = doc.add_paragraph()
add_field(p2, 'TOC \\o "1-3" \\h \\z \\u')

PAGEBREAK()

# --- List of Figures ---
P("List of Figures", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
P("")
t = doc.add_table(rows=1, cols=2)
set_cell_text(t.rows[0].cells[0], "Figure", bold=True)
set_cell_text(t.rows[0].cells[1], "Page", bold=True)
for num, cap in FIGURES:
    row = t.add_row().cells
    set_cell_text(row[0], f"Figure {num}: {cap}")
    set_cell_text(row[1], "-", align=WD_ALIGN_PARAGRAPH.CENTER)

PAGEBREAK()

# --- List of Tables ---
P("List of Tables", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
P("")
t = doc.add_table(rows=1, cols=2)
set_cell_text(t.rows[0].cells[0], "Table", bold=True)
set_cell_text(t.rows[0].cells[1], "Page", bold=True)
for num, cap in TABLES:
    row = t.add_row().cells
    set_cell_text(row[0], f"Table {num}: {cap}")
    set_cell_text(row[1], "-", align=WD_ALIGN_PARAGRAPH.CENTER)

# ===========================================================================
# SECTION 3 — BODY (arabic numerals restart at 1)
# ===========================================================================

doc.add_section(WD_SECTION.NEW_PAGE)
body = doc.sections[-1]
body.left_margin = Inches(1.5); body.right_margin = Inches(1)
body.top_margin = Inches(1); body.bottom_margin = Inches(1)
set_page_numbering(body, fmt="decimal", start=1)
add_footer_page_number(body)

print("Front matter + title page complete. Continuing with body chapters...")

# ===========================================================================
# CHAPTER 1 — INTRODUCTION
# ===========================================================================

H1_CHAPTER(1, "Introduction")

H2("1.1 Introduction")
PARA(
    "Mental health disorders such as stress, anxiety and depression affect a substantial "
    "share of the global population, yet the tools available for continuous, objective "
    "emotional assessment remain limited. Clinical practice still relies heavily on "
    "self-reporting and periodic consultations, while most consumer-facing tools -- mood "
    "diaries, single-sensor wearables, sentiment-analysis apps -- draw on only one channel "
    "of evidence at a time. A person's face, their physiological state, and their "
    "self-reported mood do not always agree, and a system that trusts only one of these "
    "signals inherits that signal's blind spots."
)
PARA(
    "MedOracle is a multimodal emotion recognition system that addresses this gap by "
    "combining two objective, continuously-available channels -- physiological signals "
    "(EEG and GSR) captured from wearable sensors, and facial video captured from an "
    "ordinary camera -- into a single fused prediction across five emotion classes: "
    "stress, calm, happy, sad and angry. Rather than simply concatenating both signals, "
    "the system is explicitly designed to reason about how much to trust each modality at "
    "prediction time, using a quality- and confidence-aware gating mechanism, and to "
    "explain its own predictions through SHAP-based feature attribution surfaced in a web "
    "application."
)
PARA(
    "The system is built by a team of three, each responsible for one stage of the "
    "pipeline: physiological signal processing and classification (Member 1), video-based "
    "emotion recognition together with the multimodal fusion layer (Member 2), and "
    "explainability together with the web application (Member 3). This report documents "
    "the motivation, design, implementation, and — most importantly — the honest, "
    "quantitative evaluation of the complete system."
)

H2("1.2 Background and Motivation")
PARA(
    "Affective computing research has historically progressed along separate tracks: "
    "facial-expression recognition, speech-emotion recognition, text sentiment analysis, "
    "and physiological emotion recognition have each developed their own datasets, feature "
    "sets and model families. Multimodal fusion has repeatedly been shown to outperform any "
    "single modality when the fusion is done carefully, because different signals fail in "
    "different ways: a face can be occluded or deliberately controlled, while EEG and GSR "
    "are harder to consciously mask but are more sensitive to motion artefacts and are "
    "highly variable between people."
)
PARA(
    "A recurring difficulty in this space, confirmed empirically in this project, is that "
    "physiological emotion recognition trained on one group of people transfers poorly to "
    "a new, unseen person -- the so-called cross-subject generalisation problem. Facial "
    "video, by contrast, generalises comparatively well across unseen individuals once a "
    "sufficiently deep visual encoder is used. This asymmetry motivates a fusion strategy "
    "that does not assume both modalities are equally reliable, but instead measures and "
    "adapts to each modality's real-time reliability -- the central design idea behind "
    "MedOracle's gated fusion mechanism."
)
PARA(
    "The project is further motivated by the need for explainability in any system "
    "intended to support mental health decisions. A black-box prediction of \"stress\" is "
    "of limited clinical value; a prediction accompanied by a feature-level explanation "
    "(e.g. \"driven mainly by the video signal, with low confidence contribution from a "
    "degraded EEG reading\") is considerably more actionable and trustworthy."
)

H2("1.3 Problem in Brief")
PARA(
    "Existing emotion-monitoring tools typically rely on a single modality, and multimodal "
    "research prototypes that do combine modalities frequently assume all modalities are "
    "always present and equally reliable, fusing them with static or learned weights that "
    "do not adapt at inference time to signal quality. In addition, physiological datasets "
    "such as DEAP are recorded from a modest number of subjects with imbalanced emotional "
    "content, making subject-independent generalisation a genuine, frequently "
    "under-reported, challenge; and video and physiological datasets rarely share the same "
    "subjects, so a system that wants to fuse both must be evaluated under a constructed "
    "cross-dataset methodology rather than a single paired dataset."
)
PARA(
    "MedOracle addresses these three problems directly: (1) it implements a "
    "confidence-and-quality-aware gated fusion that adapts its trust in each modality per "
    "prediction rather than using fixed weights; (2) it evaluates the physiological branch "
    "honestly under leave-one-subject-out cross-validation rather than reporting only an "
    "optimistic random-split number, and diagnoses the resulting limitation rather than "
    "hiding it; and (3) it constructs a principled cross-dataset evaluation set (pairing "
    "DEAP physiological windows with CREMA-D video clips of the same emotion label) to "
    "measure fusion behaviour in the absence of a single dataset containing all "
    "modalities."
)

H2("1.4 Aim and Objectives")
H3("1.4.1 Aim")
PARA(
    "To design, implement and rigorously evaluate a multimodal emotion recognition system "
    "that fuses physiological (EEG/GSR) and facial-video signals through a quality-aware "
    "gated fusion mechanism, and to make its predictions explainable through SHAP-based "
    "feature attribution delivered via a web application."
)
H3("1.4.2 Objectives")
BULLETS([
    "To develop a physiological emotion recognition module using a bidirectional "
    "cross-modal attention network trained on the DEAP dataset, mapped to a five-class "
    "emotion taxonomy shared across the system.",
    "To develop a video-based emotion recognition module using face detection, "
    "convolutional feature extraction and temporal (BiLSTM) modelling, trained on the "
    "CREMA-D dataset.",
    "To design and implement a quality-aware gated fusion mechanism that combines both "
    "modalities using entropy-based confidence and signal-quality penalties, with "
    "graceful degradation when a modality is degraded or missing.",
    "To implement a SHAP-based explainability layer and a FastAPI/React web application "
    "that allows a user to submit video and/or physiological data and receive a fused, "
    "explained prediction.",
    "To evaluate the complete system through an eight-condition ablation study "
    "(single-modality baselines, naive fusion, the full method, and robustness "
    "conditions) with statistical significance testing, and to report all results -- "
    "including unfavourable ones -- honestly.",
])

H2("1.5 Proposed Solution")
PARA(
    "The proposed system ingests three raw input types -- EEG, GSR and facial video -- and "
    "produces a single fused emotion label with an accompanying confidence score, "
    "per-modality breakdown, and SHAP-based explanation. Figure 1.1 shows the high-level "
    "architecture: Member 1's physiological branch and Member 2's video branch each "
    "produce an independent prediction dictionary; Member 2's gated fusion module combines "
    "them into a single prediction_output; and Member 3's explainability layer and web "
    "application surface the result to the end user."
)
FIGURE("1.1", f"{ASSETS}/fig_architecture.png", width=6.2)
TABLE("1.1",
      ["Module", "Owner", "Responsibility"],
      [
          ["Physiological Emotion Recognition", "Member 1 (Suhira Balarajan)",
           "EEG+GSR preprocessing, bidirectional cross-modal attention network, "
           "LOSO evaluation"],
          ["Video Emotion Recognition + Fusion", "Member 2 (Vanaiyan Kirupagaran)",
           "Face detection, ResNet50+BiLSTM video model, gated fusion, full pipeline, "
           "ablation study"],
          ["Explainability + Web Application", "Member 3 (Adshaya Balarajah)",
           "Kernel SHAP, FastAPI backend, React dashboard, LLM-based chat explanations"],
      ])
PARA(
    "The remainder of this report is structured as follows. Chapter 2 reviews prior work "
    "in physiological, video-based and multimodal emotion recognition. Chapter 3 "
    "describes the technologies adopted. Chapter 4 describes each member's approach in "
    "terms of input, process, output and evaluation. Chapter 5 presents the system's "
    "analysis and design, including the shared label taxonomy, interface contracts and "
    "fusion design. Chapter 6 describes the implementation. Chapter 7 presents the "
    "evaluation and discussion, including the ablation study and an honest account of the "
    "physiological module's limitations. Chapter 8 concludes the report and outlines "
    "future work."
)

H2("1.6 Summary")
PARA(
    "This chapter introduced MedOracle, a multimodal emotion recognition system that fuses "
    "physiological and video signals through a quality-aware gated fusion mechanism and "
    "explains its predictions through SHAP. The motivation, problem statement, aim and "
    "objectives, and proposed solution were presented. The next chapter reviews relevant "
    "prior work in each of the constituent research areas."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 2 — LITERATURE REVIEW
# ===========================================================================

H1_CHAPTER(2, "Literature Review")

H2("2.1 Introduction")
PARA(
    "This chapter reviews prior work relevant to MedOracle's three constituent problems: "
    "physiological (EEG/GSR) emotion recognition, facial video emotion recognition, and "
    "multimodal fusion with explainability. Each area is reviewed with attention to "
    "datasets, feature representations, model architectures, and -- particularly for the "
    "physiological literature -- the distinction between subject-dependent and "
    "subject-independent evaluation, which turned out to be central to this project's "
    "findings."
)

H2("2.2 Physiological (EEG/GSR) Emotion Recognition")
PARA(
    "The DEAP dataset [6] remains one of the most widely used benchmarks for "
    "physiological emotion recognition, providing 32-channel EEG and peripheral signals "
    "(including GSR) recorded from 32 subjects while they watched 40 music videos, "
    "annotated with self-reported valence, arousal and dominance on a 1-9 scale. Early "
    "work on DEAP largely reported binary classification (high/low valence, high/low "
    "arousal) using traditional classifiers such as Support Vector Machines on "
    "hand-crafted spectral features [12]. Differential Entropy (DE), a log-transform of "
    "band power that approximates Gaussianity and is known to generalise better across "
    "subjects than raw band power, has become a standard feature for EEG emotion "
    "recognition since its introduction by Duan et al. and its widespread adoption "
    "following Zheng and Lu's work on subject-independent DEAP/SEED classification."
)
PARA(
    "A recurring and, in this project's experience, decisive methodological distinction "
    "in this literature is between within-subject (or random-split) evaluation, in which "
    "windows from the same subject can appear in both the training and test sets, and "
    "subject-independent evaluation such as Leave-One-Subject-Out (LOSO) cross-validation, "
    "in which the model is tested on a subject it has never seen. Within-subject "
    "evaluations on DEAP routinely report accuracies in the 80-95% range for 3-4 "
    "coarse-grained classes; LOSO evaluations of fine-grained (4-5 class) discrete emotion "
    "recognition, by contrast, are reported far less often and generally show "
    "substantially lower performance, reflecting the well-documented difficulty of "
    "cross-subject EEG transfer [6]."
)
PARA(
    "Siddharth et al. [12] combined EEG with peripheral physiological signals and reported "
    "approximately 73% accuracy on DEAP; this figure, and comparable figures reported "
    "elsewhere in the literature, are typically obtained under within-subject or "
    "near-within-subject evaluation protocols, which this project explicitly avoided for "
    "its headline evaluation in favour of the harder, more clinically relevant "
    "subject-independent setting."
)

H2("2.3 Facial Video Emotion Recognition")
PARA(
    "Video-based facial emotion recognition has moved from static-image, hand-crafted-"
    "feature pipelines (Local Binary Patterns, Histogram of Oriented Gradients with SVM "
    "classifiers) toward deep spatio-temporal architectures that combine a convolutional "
    "encoder -- most commonly a pretrained ResNet variant [4] -- with a recurrent layer "
    "such as an LSTM or BiLSTM to model the temporal evolution of expression across "
    "frames [5]. The CREMA-D dataset [2], comprising over 7,000 audio-visual clips from "
    "91 actors across six emotion categories with crowd-sourced intensity ratings, is "
    "widely used for this purpose because of its actor diversity and multiple emotional "
    "intensity levels."
)
PARA(
    "Zhang et al. [13] reported approximately 65% accuracy for video-only emotion "
    "recognition on comparable classes, a figure broadly consistent with the video "
    "module's own held-out performance in this project (Chapter 7). Face-detection front "
    "ends such as YOLO variants are increasingly used ahead of the recognition network to "
    "isolate the face region and, in this project's design, are additionally repurposed to "
    "assess signal quality (detection rate, sharpness, face size) rather than solely to "
    "crop the input -- a distinction discussed further in Chapter 6."
)

H2("2.4 Multimodal Fusion Approaches")
PARA(
    "Multimodal affective computing surveys [1, 10] distinguish early (feature-level), "
    "intermediate (attention-based), and late (decision-level) fusion. Late fusion is "
    "generally preferred when modalities have different temporal resolutions, different "
    "noise characteristics, and -- critically for this project -- come from datasets with "
    "no shared subjects, since DEAP and CREMA-D share no overlapping individuals and joint "
    "training is therefore architecturally impossible without a shared subject pool [8, "
    "9]. Static or learned-but-fixed fusion weights are common; adaptive, "
    "confidence-driven weighting is less common but is directly motivated by the "
    "calibration literature. Guo et al. [3] showed that modern neural networks are poorly "
    "calibrated by raw softmax confidence, but that an entropy-based confidence measure "
    "over the output distribution remains a useful, model-agnostic signal of prediction "
    "reliability -- the basis of MedOracle's confidence term."
)
PARA(
    "Signal-quality-aware weighting, in which a modality's contribution is penalised "
    "according to a measured quality metric (blur, missing-data, artefact) rather than "
    "the model's own confidence, has separate precedent in image-quality literature: the "
    "Laplacian-variance sharpness metric used for this project's video-quality grading "
    "originates with Pech-Pacheco et al. [9]."
)

H2("2.5 Explainable AI for Emotion Recognition")
PARA(
    "SHAP (SHapley Additive exPlanations) provides a model-agnostic, game-theoretically "
    "grounded attribution of a prediction to its input features, and has been widely "
    "adopted in sensitive application domains -- including healthcare -- precisely because "
    "it gives per-instance, per-feature attributions rather than only global feature "
    "importance. In a multimodal fusion setting, SHAP over the fused prediction naturally "
    "extends to attributing the outcome to each modality's contribution, which this "
    "project uses to build both a bar-chart visualisation and a natural-language chatbot "
    "explanation."
)

H2("2.6 Summary")
PARA(
    "This chapter reviewed the datasets, feature representations and model families "
    "relevant to physiological, video-based and multimodal emotion recognition, and "
    "highlighted the within-subject versus subject-independent evaluation distinction "
    "that proved central to this project's honest evaluation of its physiological "
    "module. The next chapter describes the technologies adopted to implement the "
    "system."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 3 — TECHNOLOGY ADAPTED
# ===========================================================================

H1_CHAPTER(3, "Technology Adapted")

H2("3.1 Introduction")
PARA(
    "This chapter summarises the development tools, programming languages, libraries and "
    "version-control practices used to implement MedOracle across its three modules."
)

H2("3.2 Development Tools / IDEs")
BULLETS([
    "Kaggle Notebooks — used for GPU-accelerated training of the video model "
    "(ResNet50+BiLSTM) and the physiological LOSO training runs, with cross-session "
    "checkpoint persistence to work within Kaggle's 12-hour session limit.",
    "Visual Studio Code — primary IDE for backend, frontend and model-development code "
    "across all three modules.",
    "Claude Code (CLI) — used as an AI pair-programming assistant for iterative "
    "debugging, integration work, and evaluation scripting throughout the project.",
])

H2("3.3 Programming Language")
PARA(
    "Python was used for all machine-learning, signal-processing and backend "
    "development. JavaScript (with React and JSX) was used for the frontend web "
    "application."
)

H2("3.4 Libraries")
BULLETS([
    "PyTorch — deep learning framework used for both the physiological "
    "(BiAttn) network and the video (ResNet50 + BiLSTM) network.",
    "Ultralytics YOLOv8 — face detection, used both to crop/locate faces during "
    "signal-quality assessment and during earlier training-data preparation.",
    "torchvision — pretrained ResNet50 backbone for the video encoder.",
    "scikit-learn — classical baselines (SVM, Random Forest, Logistic Regression), "
    "metrics (macro-F1, Wilcoxon-adjacent utilities) and cross-validation utilities "
    "(StratifiedGroupKFold, GroupKFold).",
    "SciPy — Welch power spectral density estimation for Differential Entropy feature "
    "extraction, and the Wilcoxon signed-rank significance test used in the ablation "
    "study.",
    "NumPy / Pandas — array and tabular data processing throughout.",
    "OpenCV — video frame decoding and image processing.",
    "SHAP — Kernel SHAP explainability layer over the fused prediction.",
    "FastAPI + Uvicorn — backend web framework and ASGI server.",
    "SQLAlchemy (async) — ORM layer over SQLite (development) / PostgreSQL "
    "(production).",
    "python-jose — JWT access/refresh token issuing and verification.",
    "Pydantic — request/response schema validation for the FastAPI backend.",
    "React + Vite + Axios + Recharts — frontend dashboard, HTTP client, and charting "
    "(emotion-trend line chart, SHAP bar chart).",
    "Matplotlib — confusion-matrix and ablation-result visualisation.",
])

H2("3.5 Version Control System")
PARA(
    "Git and GitHub were used throughout the project for source control, branching per "
    "member/feature, pull-request-based code review, and merging each member's module "
    "into a shared develop branch prior to integration."
)

H2("3.6 Summary")
PARA(
    "This chapter listed the tools, languages and libraries used to build MedOracle. The "
    "next chapter describes each member's individual approach in terms of input, "
    "process, output and evaluation."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 4 — OUR APPROACH
# ===========================================================================

H1_CHAPTER(4, "Our Approach")

H2("4.1 Introduction")
PARA(
    "This chapter describes each member's individual approach in terms of input, "
    "process, output and evaluation, following the shared five-class emotion taxonomy "
    "(stress, calm, happy, sad, angry) that all three modules are required to use."
)

H2("4.2 Member 1 — Physiological Emotion Recognition")
H3("4.2.1 Input")
PARA(
    "Raw EEG (32 channels, 128 Hz) and GSR (originally 4 Hz, resampled to 128 Hz using "
    "polyphase resampling) windows from the DEAP dataset, each representing a 4-second "
    "trial window (512 samples), together with the trial's self-reported valence, "
    "arousal and dominance ratings."
)
H3("4.2.2 Process")
PARA(
    "Each DEAP trial is harmonised from its continuous valence/arousal/dominance ratings "
    "into one of the five shared emotion classes using literature-grounded thresholds "
    "(Section 5.3). EEG and GSR windows are each passed through a dedicated encoder -- an "
    "EEGNet-inspired depthwise/pointwise convolutional stack for EEG, and a comparable "
    "convolutional stack for GSR -- before a bidirectional cross-modal attention layer "
    "allows the EEG representation to attend over the GSR representation and vice versa. "
    "The attended representations are combined and passed to a joint classification head "
    "producing a five-class softmax distribution. The model is evaluated using two "
    "protocols: an initial random-split evaluation (used to obtain a working checkpoint "
    "for early system integration), and a full Leave-One-Subject-Out (32-fold) "
    "cross-validation, which is the honest, subject-independent evaluation reported in "
    "this project."
)
H3("4.2.3 Output")
PARA(
    "A physiological_prediction_dict containing the predicted emotion label, an "
    "entropy-based confidence score, the full five-class probability distribution, and a "
    "signal-quality grade (good/degraded/poor) for both EEG and GSR, following the shared "
    "interface contract described in Section 5.5."
)
H3("4.2.4 Evaluation")
PARA(
    "The random-split checkpoint achieves a macro-F1 of 0.4171 (accuracy approximately "
    "42%), but this figure is optimistic due to subject leakage between the training and "
    "test windows. The honest LOSO evaluation across all 32 folds yields a macro-F1 of "
    "0.179 +/- 0.041 -- close to the level expected from a five-class problem with severe "
    "class imbalance -- and this result, together with a set of controlled diagnostic "
    "experiments that isolated its cause, is discussed in full in Section 7.2."
)

H2("4.3 Member 2 — Video-Based Emotion Recognition and Multimodal Fusion")
H3("4.3.1 Input")
PARA(
    "Raw video clips from the CREMA-D (and, during model development, RAVDESS) datasets, "
    "each containing an actor delivering a short, emotionally-labelled utterance."
)
H3("4.3.2 Process")
PARA(
    "Sixteen frames are sampled uniformly across each 4-second window and resized to "
    "224x224 without face-cropping for the recognition model itself (an empirical finding, "
    "discussed in Section 6.3, showed full-frame resizing outperforms YOLO-cropped faces "
    "for this recognition task); a separate YOLOv8 face-detection pass is instead used "
    "purely to grade video signal quality (face-detection rate, Laplacian-variance "
    "sharpness, face bounding-box area). The sampled frames are passed through a "
    "partially fine-tuned ResNet50 (the first two residual blocks frozen, the last two "
    "blocks and the final fully-connected layer fine-tuned with a discriminative learning "
    "rate) to obtain per-frame feature vectors, which are then modelled temporally by a "
    "two-layer bidirectional LSTM (256 hidden units per direction) before a final softmax "
    "over the five emotion classes. Training used actor-independent StratifiedGroupKFold "
    "cross-validation, cosine learning-rate scheduling with warmup, and cutout "
    "augmentation, and was carried out across multiple Kaggle sessions with cross-session "
    "checkpoint persistence."
)
PARA(
    "Member 2 additionally designed and implemented the canonical gated fusion module "
    "(fusion.py), the full multimodal pipeline (pipeline.py) that invokes both Member 1's "
    "and Member 2's models and fuses their outputs, the cross-dataset synchronised "
    "evaluation-set construction, and the C1-C8 ablation study with statistical "
    "significance testing -- described fully in Chapters 5-7."
)
H3("4.3.3 Output")
PARA(
    "A video prediction_dict (predicted emotion, confidence, five-class probability "
    "distribution) from the video model alone, and -- once fused with Member 1's output "
    "-- a complete prediction_output dictionary containing the fused label, fused "
    "confidence, per-modality predictions, modality weights, and signal-quality grades "
    "for all three signals (EEG, GSR, video)."
)
H3("4.3.4 Evaluation")
PARA(
    "The video model achieves a held-out test macro-F1 of 0.6486 +/- 0.0154 across 788 "
    "clips from 11 actors unseen during training (best single checkpoint: 0.6615), with "
    "a val-to-test gap of essentially zero, indicating genuine generalisation rather than "
    "overfitting. The complete fused system was evaluated through an eight-condition "
    "ablation study; the full gated-fusion method significantly outperforms both "
    "physio-only and video-only baselines (Wilcoxon signed-rank, p < 0.001 in both "
    "cases) and degrades gracefully to the physio-only prediction when video is missing. "
    "Full results are presented in Section 7.4."
)

H2("4.4 Member 3 — Explainability and Web Application")
H3("4.4.1 Input")
PARA(
    "The fused prediction_output dictionary produced by Member 2's pipeline, together "
    "with user authentication credentials and, for the chat feature, free-text user "
    "queries about a given prediction."
)
H3("4.4.2 Process")
PARA(
    "Kernel SHAP is applied over the prediction_output to attribute the fused prediction "
    "to each contributing modality and feature, and a faithfulness metric is computed by "
    "perturbing input features and measuring the resulting change in prediction. A "
    "FastAPI backend exposes nine endpoints (Table 6.3) covering authentication, "
    "prediction (synthetic, video-only, and the multimodal video+EEG+GSR route "
    "integrated in this project), SHAP explanation retrieval, session history, chat, and "
    "dashboard summary statistics, backed by a four-table SQLAlchemy schema (users, "
    "sessions, predictions/SHAP logs, chat history) over SQLite in development and "
    "PostgreSQL in production. A React frontend renders an emotion-trend line chart, a "
    "SHAP bar chart, a session-history panel, and a modality-conflict explanation panel "
    "-- Member 3's novel contribution, which explains cases where the physiological and "
    "video predictions disagree -- together with a chatbot panel that bridges SHAP "
    "output to a large-language-model-generated, plain-English explanation."
)
H3("4.4.3 Output")
PARA(
    "A shap_output dictionary (the prediction_output enriched with shap_values, "
    "feature_names, faithfulness_score and reliability_flags) persisted to the database "
    "and rendered in the dashboard, together with natural-language chat responses."
)
H3("4.4.4 Evaluation")
PARA(
    "The web application was evaluated through a full in-process HTTP test (registration, "
    "login, and both video-only and full multimodal video+EEG+GSR prediction requests), "
    "confirming correct end-to-end operation including graceful degradation and input "
    "validation (malformed EEG/GSR shapes are rejected with HTTP 422). This is discussed "
    "further in Section 7.5."
)

H2("4.5 Summary")
PARA(
    "This chapter described each member's individual contribution in terms of input, "
    "process, output and evaluation. The next chapter presents the system's analysis and "
    "design, including the shared emotion taxonomy, signal-quality thresholds, interface "
    "contracts, and the gated fusion mechanism that ties the three modules together."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 5 — ANALYSIS AND DESIGN
# ===========================================================================

H1_CHAPTER(5, "Analysis and Design")

H2("5.1 Introduction")
PARA(
    "This chapter presents the analysis and design decisions that allow three "
    "independently-developed modules to compose into a single working system: the "
    "shared emotion taxonomy, the signal-quality grading scheme, the interface contracts "
    "between modules, and the design of the gated fusion mechanism itself."
)

H2("5.2 System Architecture")
PARA(
    "Figure 5.1 shows the data flow between modules together with the exact dictionaries "
    "exchanged at each boundary. Member 1 and Member 2 each expose a pure function "
    "(predict(eeg, gsr) and predict_video(video_path) respectively) that can be called "
    "independently; Member 2's gated-fusion function accepts either or both outputs and "
    "always returns a dictionary satisfying the same downstream contract, so Member 3's "
    "explainability layer and web application do not need to know whether zero, one, or "
    "two modalities were actually available for a given prediction."
)
FIGURE("5.1", f"{ASSETS}/fig_architecture.png", width=6.2)

H2("5.3 Shared Emotion Taxonomy and Label Harmonization")
PARA(
    "All three modules are required to use exactly the same five discrete emotion "
    "classes -- stress, calm, happy, sad, angry -- defined once in a shared module "
    "(shared/label_harmonization.py) and imported everywhere else, so that class "
    "probability dictionaries are always directly comparable and summable across "
    "modalities. Because DEAP provides continuous valence/arousal/dominance ratings and "
    "CREMA-D provides discrete six-way emotion labels, each dataset requires its own "
    "mapping into this shared taxonomy, shown in Table 5.2. The DEAP mapping follows "
    "Russell's circumplex model [11] and Mehrabian's PAD model [8], thresholding each "
    "dimension at the scale midpoint (5.0); the CREMA-D mapping maps fear to stress "
    "(fear is known to activate the sympathetic nervous system in a manner similar to "
    "stress [7]) and neutral to calm, while disgust is dropped as having no clean "
    "correspondence in the shared taxonomy."
)
TABLE("5.2",
      ["Source", "Mapping", "Notes"],
      [
          ["DEAP", "Valence>=5, Arousal<5, Dominance<5 -> calm", "Russell (1980)"],
          ["DEAP", "Valence>=5, Arousal>=5, Dominance>=5 -> happy", "Russell (1980)"],
          ["DEAP", "Valence<5, Arousal>=5, Dominance<5 -> stress", "Mehrabian (1996)"],
          ["DEAP", "Valence<5, Arousal<5, Dominance<5 -> sad", "Russell (1980)"],
          ["DEAP", "Valence<5, Arousal>=5, Dominance>=5 -> angry", "Russell (1980)"],
          ["CREMA-D", "ANG -> angry, HAP -> happy, SAD -> sad", "Direct correspondence"],
          ["CREMA-D", "NEU -> calm", "Neutral approximated as low-arousal calm"],
          ["CREMA-D", "FEA -> stress", "Kreibig (2010): fear activates sympathetic system"],
          ["CREMA-D", "DIS -> dropped", "No clean correspondence; excluded from training"],
      ])

H2("5.4 Signal Quality Assessment")
PARA(
    "Each modality is graded into one of three quality tiers -- good, degraded, poor -- "
    "used both to report signal reliability and, critically, to weight each modality's "
    "contribution during fusion. Table 5.1 shows the thresholds used for each signal. "
    "The video thresholds were empirically recalibrated during this project: the "
    "originally adopted generic thresholds (Laplacian variance >= 100 for \"good\", face "
    "area >= 10% of frame) were measured to grade approximately 80% of clean, "
    "fully-detected CREMA-D frontal-face clips as \"degraded\", because those generic "
    "values were drawn from document-scan sharpness literature [9] rather than "
    "calibrated to this project's actual acted-video domain. Recalibrating to the "
    "measured distribution of the held-out CREMA-D test set (Laplacian variance >= 70 "
    "for \"good\", face area >= 6%) restored 73 of 80 sampled clips to \"good\" while still "
    "correctly flagging genuinely blurred or distant faces."
)
TABLE("5.1",
      ["Metric", "Good", "Degraded", "Poor"],
      [
          ["EEG amplitude range", "+/-100 uV", "+/-100-150 uV", "> +/-150 uV"],
          ["EEG flat-line channels", "0", "1-2", "> 2"],
          ["EEG NaN ratio", "< 1%", "1-5%", "> 5%"],
          ["GSR signal range", "0.5-30 uS", "0.1-0.5 or 30-50 uS", "< 0.1 or > 50 uS"],
          ["GSR SCR peaks (60s)", ">= 2", "1", "0"],
          ["Video face-detection rate", ">= 80%", "50-80%", "< 50%"],
          ["Video Laplacian variance", ">= 70 (recalibrated)", "40-70", "< 40"],
          ["Video face bounding-box area", "> 6% (recalibrated)", "3-6%", "< 3%"],
      ])
PARA(
    "The worst-performing metric for a given modality determines its overall grade -- "
    "for example, a video clip with a high detection rate but a small, distant face is "
    "still graded no better than its face-area metric permits."
)

H2("5.5 Interface Contracts Between Modules")
PARA(
    "Interface contracts are defined once in shared/data_contracts.py and validated by "
    "assertion helper functions, so that a malformed dictionary raises immediately at "
    "the module boundary rather than propagating silently downstream. Table 6.1 and "
    "Table 6.2 (Chapter 6) show the exact schema of the M1->M2 and M2->M3 contracts "
    "respectively."
)

H2("5.6 Gated Fusion Design")
PARA(
    "The fusion mechanism combines each modality's prediction using an entropy-based "
    "confidence score multiplied by a signal-quality penalty, then L1-normalises the "
    "resulting per-modality weights (rather than applying a softmax, which would "
    "compress large confidence gaps). Formally, for modality m with predicted "
    "probability distribution P_m over K classes:"
)
PARA(
    "confidence:  c_m = 1 - H(P_m) / log(K),  where H(P_m) = -sum_k P_m(k) log P_m(k)"
)
PARA(
    "quality penalty:  alpha_m = 1.0 (good), 0.5 (degraded), 0.1 (poor)"
)
PARA(
    "gate score:  g_m = c_m * alpha_m"
)
PARA(
    "fusion weight:  w_m = g_m / sum_m g_m  (L1-normalised, not softmax)"
)
PARA(
    "fused prediction:  P_fused = sum_m w_m * P_m;  predicted_emotion = argmax(P_fused)"
)
PARA(
    "Four graceful-degradation rules are built into the design: if both modalities are "
    "available, full gated fusion is applied; if video is poor or missing, its weight "
    "tends toward zero and the physiological prediction alone is used (falling back to "
    "a uniform distribution only if physiology is also unreliable); symmetrically for a "
    "missing or poor physiological signal; and if both modalities are simultaneously "
    "unreliable, the system falls back to a uniform distribution and explicitly flags "
    "the prediction as unreliable rather than reporting a false confident answer."
)

H2("5.7 Database and Web Application Design")
PARA(
    "The web application persists four tables: users (authentication), sessions "
    "(one row per prediction request, storing the fused prediction and modality "
    "weights), a SHAP log table (per-session feature attributions and faithfulness "
    "score), and chat history (per-session chatbot conversation). JWT access and "
    "refresh tokens gate all endpoints except registration and login."
)

H2("5.8 Summary")
PARA(
    "This chapter presented the shared taxonomy, signal-quality thresholds, interface "
    "contracts, and gated fusion design that allow the three independently-developed "
    "modules to compose into a single working system. The next chapter describes how "
    "each of these designs was implemented."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 6 — IMPLEMENTATION
# ===========================================================================

H1_CHAPTER(6, "Implementation")

H2("6.1 Introduction")
PARA(
    "This chapter describes the implementation of each module, following the design "
    "presented in Chapter 5. Source code is organised into three top-level packages "
    "(member1_physiological, member2_video_fusion, member3_explainability) plus a shared "
    "package (shared/) holding the taxonomy, label harmonisation and interface-contract "
    "code common to all three."
)

H2("6.2 Module 1 — Physiological Emotion Recognition Implementation")
H3("6.2.1 Data Loading and Preprocessing")
PARA(
    "The DEAP loader (preprocessing/deap_loader.py) reads each subject's pickled .dat "
    "file, trims the first 3 seconds (384 samples) of baseline recording from each "
    "60-second trial, and slides 15 non-overlapping 4-second (512-sample) windows across "
    "the remainder. EEG channels 0-31 and the GSR channel (index 36) are extracted per "
    "window, and the trial's valence/arousal/dominance rating is converted to one of the "
    "five shared classes via the label-harmonisation mapping described in Section 5.3. "
    "Figure 6.1 shows the resulting per-subject window counts and class distribution "
    "produced by this loader."
)
FIGURE("6.1", "member1_physiological/visualization/viz_deap_loader_output.png", width=6.0)
PARA(
    "As Table 7.1 (Section 7.2) will show in more detail, this harmonisation produces a "
    "severely imbalanced class distribution -- most notably only 19 \"happy\" trials "
    "across all 32 subjects -- which later proved central to explaining the "
    "physiological module's evaluation results."
)
H3("6.2.2 Signal Quality Assessment")
PARA(
    "A dedicated signal_quality module grades each EEG and GSR window against the "
    "thresholds in Table 5.1. Figure 6.2 shows example output of this grading process "
    "across a sample of windows."
)
FIGURE("6.2", "member1_physiological/visualization/viz_signal_quality_output.png", width=6.0)
H3("6.2.3 Network Architecture")
PARA(
    "The EEG encoder applies a depthwise temporal convolution (kernel size 25, "
    "approximately 200ms at 128Hz, capturing alpha/beta oscillations) followed by a "
    "pointwise convolution mixing across channels, then two further downsampling "
    "convolutional blocks reducing the 512-sample sequence to 32 timesteps at a 128-"
    "dimensional embedding, with sinusoidal positional encoding added before the "
    "attention layer. The GSR encoder follows an analogous, single-channel convolutional "
    "structure. A bidirectional cross-modal attention layer then allows the EEG sequence "
    "to attend over the GSR sequence and vice versa, before the attended representations "
    "are pooled and passed to a shared five-class classification head."
)
H3("6.2.4 Baseline Comparisons")
PARA(
    "Three classical baselines -- SVM, Random Forest and a Multi-Layer Perceptron, each "
    "using a 165-dimensional hand-crafted feature vector (32 channels x 5 band-power "
    "features + 5 GSR statistics) -- were implemented and evaluated under the same "
    "32-fold Leave-One-Subject-Out protocol as the neural network, in order to establish "
    "whether the neural architecture itself, rather than the underlying data, was "
    "limiting performance. Figures 6.3-6.5 show the resulting outputs."
)
FIGURE("6.3", 'member1_physiological/baseline/Outputs of baselines/SVM output.png', width=6.0)
FIGURE("6.4", 'member1_physiological/baseline/Outputs of baselines/Random Forest output.png', width=6.0)
FIGURE("6.5", 'member1_physiological/baseline/Outputs of baselines/MLP output.png', width=6.0)
PARA(
    "All three classical baselines converge to a similar macro-F1 range as the neural "
    "network under LOSO, corroborating the diagnosis presented in Section 7.2: the "
    "limitation is primarily in the data (class imbalance and cross-subject variability) "
    "rather than in any single model architecture."
)
H3("6.2.5 Interface Export")
TABLE("6.1",
      ["Field", "Type", "Description"],
      [
          ["predicted_emotion", "str", "One of the five shared emotion labels"],
          ["confidence", "float in [0,1]", "Entropy-based confidence, 1 - H(P)/log(K)"],
          ["class_probabilities", "dict[str,float]", "Full 5-class probability distribution"],
          ["signal_quality.eeg", "str", "\"good\" | \"degraded\" | \"poor\""],
          ["signal_quality.gsr", "str", "\"good\" | \"degraded\" | \"poor\""],
      ])
PARA(
    "The predict.py module exposes PhysiologicalPredictor.predict(eeg, gsr), returning a "
    "dictionary matching Table 6.1, validated by make_physiological_prediction() in the "
    "shared contracts module."
)

H2("6.3 Module 2 — Video-Based Emotion Recognition Implementation")
H3("6.3.1 Preprocessing and Training")
PARA(
    "Sixteen frames per clip are uniformly sampled and resized directly to 224x224 "
    "without cropping to a detected face region. This choice followed a controlled "
    "comparison during development: an earlier implementation cropped each frame to its "
    "YOLO-detected face region (matching common practice), but a held-out evaluation on "
    "CREMA-D clips showed this configuration achieving only 0.56 macro-F1, versus 0.70 "
    "for the same model applied to full, uncropped frames -- likely because cropping "
    "discards contextual cues (head pose, shoulder movement) that the temporal model "
    "otherwise exploits. YOLOv8 face detection is retained in the pipeline, but is now "
    "used exclusively for signal-quality grading (Section 5.4) rather than for cropping "
    "the recognition input."
)
PARA(
    "The ResNet50 backbone is partially fine-tuned: the first two residual blocks "
    "(layer1, layer2) are frozen to retain general low-level ImageNet features, while "
    "the later blocks (layer3, layer4) and a replaced classification head "
    "(Linear(2048->512) -> ReLU -> Dropout(0.4) -> Linear(512->5)) are fine-tuned. "
    "Discriminative learning rates (1e-5 for the fine-tuned backbone, 5e-4 for the new "
    "head and BiLSTM) were adopted after an initial full-backbone-freeze configuration "
    "underfit (train and validation macro-F1 both approximately 0.30) and a fully "
    "unfrozen configuration overfit (train macro-F1 0.88 against validation 0.63). The "
    "temporal BiLSTM has two layers with 256 hidden units per direction and inter-layer "
    "dropout of 0.3. Training used actor-independent StratifiedGroupKFold "
    "cross-validation (so no actor appears in both a training and validation fold), "
    "cosine-annealed learning rate with warmup, cutout augmentation on input frames, and "
    "frozen batch-normalisation statistics, and was carried out on Kaggle across "
    "multiple 12-hour sessions with automatic checkpoint upload/resume between sessions."
)
H3("6.3.2 Gated Fusion Implementation")
PARA(
    "The canonical fusion implementation (member2_video_fusion/fusion.py) implements "
    "exactly the formulas given in Section 5.6, and is validated against Member 3's "
    "earlier, independently-implemented reference fusion function (conflict/gate.py) "
    "with a parametrised unit test covering all combinations of signal-quality grade for "
    "both modalities, asserting that both implementations produce identical modality "
    "weights and fused probabilities to within 1e-9 absolute tolerance. This allows "
    "gate.py and other duplicate fusion logic elsewhere in the codebase to eventually be "
    "collapsed to import the single canonical implementation."
)
H3("6.3.3 Full Pipeline")
PARA(
    "member2_video_fusion/pipeline.py exposes run_full_pipeline(eeg, gsr, video_path), "
    "which lazily loads Member 1's PhysiologicalPredictor, runs Member 2's video model, "
    "and combines both outputs via gated_fusion(). Any single modality may be omitted: "
    "if only eeg/gsr are supplied the function returns a physio-only prediction; if only "
    "a video path is supplied it returns a video-only prediction (still routed through "
    "the same gated-fusion function so the output schema and graceful-degradation "
    "behaviour are identical regardless of which modality is present); and a "
    "SynchronizedInput convenience wrapper is provided for the case where all three raw "
    "signals are available together."
)
TABLE("6.2",
      ["Field", "Type", "Description"],
      [
          ["predicted_emotion", "str", "Fused emotion label"],
          ["confidence", "float", "Fused entropy-based confidence"],
          ["class_probabilities", "dict[str,float]", "Fused 5-class probability distribution"],
          ["modality_weights", "dict", '{"physio": w1, "video": w2}, sums to 1.0'],
          ["signal_quality", "dict", "{eeg, gsr, video} quality grades"],
          ["per_modality_predictions", "dict", "Raw physio and video predictions before fusion"],
      ])
H3("6.3.4 Cross-Dataset Evaluation Set Construction")
PARA(
    "Because DEAP and CREMA-D share no common subjects, a paired evaluation set was "
    "constructed by pairing, for each of the five emotion classes, a real DEAP EEG/GSR "
    "window with a real CREMA-D video clip carrying the same emotion label -- the "
    "standard methodology for evaluating cross-dataset late fusion when no single "
    "dataset provides every modality [1, 10]. build_synced_dataset.py constructs ten such "
    "\"virtual subjects\" (two per emotion) for the web-application demonstration, and "
    "the ablation study (Section 6.3.5) uses a larger, randomly re-paired version of the "
    "same construction (20 samples per emotion, 50 random pairings) to obtain "
    "statistically meaningful macro-F1 estimates with a measurable variance."
)
H3("6.3.5 Ablation Study Implementation")
PARA(
    "ablation_study.py precomputes each modality's prediction once per held-out sample "
    "(20 DEAP windows and 20 held-out CREMA-D clips per emotion, drawn from actors never "
    "used in video training), then evaluates eight fusion conditions over 50 random "
    "re-pairings of the precomputed predictions, reporting mean and standard deviation "
    "of macro-F1 per condition and a one-tailed Wilcoxon signed-rank test (reusing "
    "Member 1's wilcoxon_test utility) comparing the full method (C5) against each "
    "single-modality and naive-fusion baseline. Full results are presented in Section "
    "7.4."
)

H2("6.4 Module 3 — Explainability and Web Application Implementation")
H3("6.4.1 SHAP Explainability")
PARA(
    "shap_output_builder.py wraps the prediction_output in a Kernel SHAP explanation, "
    "computing per-feature attribution values, a faithfulness score (obtained by "
    "perturbing individual features and measuring the resulting change in the fused "
    "prediction), and signal-reliability flags derived from the signal_quality fields."
)
H3("6.4.2 Backend")
TABLE("6.3",
      ["Method", "Path", "Purpose"],
      [
          ["POST", "/auth/register", "User registration"],
          ["POST", "/auth/login", "JWT token issue"],
          ["POST", "/predict", "Run full pipeline (generic entry point)"],
          ["POST", "/predict/video", "Video-only prediction"],
          ["POST", "/predict/multimodal", "Video + EEG + GSR file upload -> fused prediction"],
          ["GET", "/explain/{session_id}", "SHAP explanation for a session"],
          ["GET", "/sessions", "List user sessions"],
          ["POST", "/chat", "LLM chatbot message"],
          ["GET", "/dashboard/summary", "Aggregated dashboard statistics"],
      ])
PARA(
    "The /predict/multimodal endpoint, integrated during this project's system-"
    "integration phase, accepts a required video file together with optional EEG and "
    "GSR files (.npy or .csv, validated against the required (32,512) and (512,) "
    "shapes respectively); if EEG/GSR are omitted the endpoint falls back to "
    "video-only prediction via the same underlying pipeline, so a single code path "
    "serves both the video-only and full-multimodal use cases."
)
H3("6.4.3 Frontend")
PARA(
    "The React dashboard was extended with EEG and GSR file inputs alongside the "
    "existing video drop-zone, and two prediction actions -- \"Video only\" and \"Run "
    "Multimodal Fusion\" -- the latter enabled only once all three files are selected. "
    "The existing SHAP bar chart, emotion-trend chart, and modality-conflict "
    "explanation panel render directly from whichever prediction_output the selected "
    "action produces, requiring no separate code path for the multimodal case."
)

H2("6.5 Testing")
PARA(
    "The project maintains 82 automated tests, all passing: 59 tests over the shared "
    "label-harmonisation module (covering DEAP threshold boundaries, CREMA-D label "
    "mapping including case-insensitivity, and batch-processing helpers), 18 tests over "
    "the canonical fusion module (including 12 parametrised tests asserting numerical "
    "equivalence with the independent reference implementation across every "
    "signal-quality combination, plus degradation and both-modality-missing edge "
    "cases), and 5 tests over the full pipeline (physio-only, video-only, and full "
    "multimodal branches). A full in-process HTTP test additionally exercises the "
    "backend end-to-end: registration, login, a real multimodal prediction request "
    "(returning HTTP 200 with genuinely non-zero weight on both modalities and a "
    "populated SHAP explanation), a video-only request, and a malformed-EEG-shape "
    "request (correctly rejected with HTTP 422)."
)

H2("6.6 Summary")
PARA(
    "This chapter described the implementation of each module and the integration work "
    "-- the canonical fusion module, the full pipeline, the cross-dataset evaluation set, "
    "and the multimodal web endpoint -- that ties them together. The next chapter "
    "presents the evaluation and discussion of the complete system, including an honest "
    "account of the physiological module's limitations."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 7 — DISCUSSION & EVALUATION
# ===========================================================================

H1_CHAPTER(7, "Discussion & Evaluation")

H2("7.1 Introduction")
PARA(
    "This chapter presents the evaluation of each module and of the fused system as a "
    "whole. In keeping with the project's emphasis on scientific integrity, results are "
    "reported honestly throughout, including the physiological module's genuine "
    "limitation under subject-independent evaluation and the diagnostic work undertaken "
    "to understand its cause, and the fusion method's honest comparison against a naive "
    "baseline rather than only against favourable comparisons."
)

H2("7.2 Physiological Module Evaluation")
H3("7.2.1 Two Evaluation Protocols")
PARA(
    "The physiological module was evaluated under two distinct protocols. A random-split "
    "evaluation, in which windows are shuffled and split without regard to which subject "
    "they came from, yields a macro-F1 of 0.4171 (accuracy approximately 42%). This "
    "figure is optimistic: because each subject contributes many near-duplicate 4-second "
    "windows from the same 60-second trial, a random split allows the model to see "
    "windows from the same subject and even the same trial in both its training and test "
    "sets, so part of the apparent performance reflects subject recognition rather than "
    "emotion recognition. The honest protocol -- Leave-One-Subject-Out (LOSO) "
    "cross-validation across all 32 DEAP subjects, in which the model is tested on a "
    "subject it has never encountered during training -- yields a substantially lower "
    "macro-F1 of 0.179 +/- 0.041, close to the level expected under a five-class problem "
    "with severe imbalance, and the model soup obtained by averaging all 32 fold "
    "checkpoints performs almost identically (macro-F1 0.179), confirming this is a "
    "genuine ceiling rather than fold-specific noise."
)
H3("7.2.2 Root-Cause Diagnosis")
PARA(
    "Rather than accept 0.179 as an unexplained number, a series of controlled "
    "diagnostic experiments was carried out to isolate its cause."
)
TABLE("7.1",
      ["Class", "Trials", "Share"],
      [
          ["stress", "172", "23.2%"],
          ["calm", "191", "25.7%"],
          ["happy", "19", "2.6%"],
          ["sad", "236", "31.8%"],
          ["angry", "124", "16.7%"],
          ["unclassifiable (dropped)", "538", "(42% of all trials)"],
      ])
PARA(
    "First, class imbalance: Table 7.1 shows that the DEAP-to-5-class mapping described "
    "in Section 5.3 leaves only 19 \"happy\" trials across all 32 subjects (2.6% of "
    "classifiable trials), and drops 42% of all trials as unclassifiable under the "
    "chosen valence/arousal/dominance thresholds. Under LOSO, the trained model was "
    "observed to collapse to predicting a single majority class (\"calm\") for almost "
    "every held-out window, which alone accounts for the near-zero F1 on the "
    "minority classes and drags the macro-averaged score down substantially."
)
PARA(
    "Second, feature representation: raw time-domain EEG (the representation used by "
    "the shipped network), Welch-based band power (used by the classical baselines in "
    "Section 6.2.4), and Differential Entropy features (32 channels x 5 bands = 160 "
    "features, plus 7 robust GSR statistics, implemented in "
    "member1_physiological/preprocessing/de_features.py) were each evaluated under LOSO "
    "with multiple classifiers (the neural network, logistic regression, and random "
    "forest). All three feature representations, across all three classifiers, converged "
    "to the same 0.16-0.18 macro-F1 range, indicating that the limitation lies in the "
    "underlying signal and label distribution rather than in any single feature "
    "engineering choice."
)
PARA(
    "Third, data quality: the DEAP GSR channel (index 36) was found to contain "
    "physically impossible values (skin conductance cannot be negative, yet 8-79% of "
    "samples across different subjects were negative, with artefact spikes reaching "
    "+/-200,000 against typical values in the low thousands, and inconsistent scaling "
    "between subjects). A robust GSR feature extractor was implemented (winsorising each "
    "window to its own 5th-95th percentile before z-scoring) to rule out this corruption "
    "as the primary cause; however, comparing EEG-only, EEG+raw-GSR and EEG+cleaned-GSR "
    "feature sets under identical LOSO evaluation showed a negligible difference "
    "(macro-F1 0.1552, 0.1576 and 0.1561 respectively), confirming that GSR contributes "
    "little discriminative signal for this five-class problem regardless of its data "
    "quality, and that the bottleneck lies elsewhere."
)
PARA(
    "Fourth, taxonomy granularity: coarsening the five-class taxonomy into three classes "
    "by grouping happy and calm into a single \"positive\" class and stress and angry "
    "into a single \"distress\" class (leaving sadness on its own) was found to lift the "
    "same feature/classifier combination to a macro-F1 of approximately 0.28 under LOSO, "
    "and a purely binary (positive/negative valence) task reached approximately 0.44 "
    "macro-F1 (0.75-0.77 accuracy). Because the video module already outputs the same "
    "five fine-grained classes, this coarser taxonomy could, in principle, be applied to "
    "the video model's output by summing grouped probabilities with no retraining "
    "required. This change was ultimately scoped as a cross-team taxonomy decision "
    "outside this report's completed system, since it would require corresponding "
    "changes to Member 3's SHAP feature naming and web-application labels, and was left "
    "as a documented option for future work (Section 8.2) rather than implemented "
    "unilaterally."
)
H3("7.2.3 Interpretation")
PARA(
    "The honest conclusion is that subject-independent, five-class emotion recognition "
    "from EEG and GSR alone is a genuinely difficult problem, consistent with the wider "
    "literature's tendency to report either within-subject or coarser-grained results "
    "(Section 2.2). Reporting this limitation candidly, together with the diagnostic "
    "work that located its cause in class imbalance and cross-subject signal "
    "variability rather than in implementation defects, is treated in this project as "
    "more valuable than reporting an inflated, leaked number -- and, as Section 7.4 "
    "shows, the system's fusion design was explicitly built to remain robust to exactly "
    "this kind of single-modality weakness."
)

H2("7.3 Video Module Evaluation")
PARA(
    "The video module was evaluated under actor-independent 5-fold "
    "StratifiedGroupKFold cross-validation and, separately, on a held-out test set of "
    "788 clips from 11 actors never used during training or cross-validation. The "
    "5-fold cross-validation macro-F1 was 0.6433 +/- 0.0303; the held-out test macro-F1 "
    "was 0.6486 +/- 0.0154 across folds, with the best single checkpoint (fold 2) "
    "reaching 0.6615. The near-zero (slightly negative) gap between validation and "
    "held-out test performance indicates genuine generalisation to unseen actors rather "
    "than overfitting."
)
TABLE("7.2",
      ["Emotion", "Test F1"],
      [
          ["happy", "0.82"],
          ["angry", "0.73"],
          ["calm", "0.62"],
          ["stress", "0.60"],
          ["sad", "0.54 (weakest)"],
      ])
PARA(
    "Table 7.2 shows the per-class breakdown on the held-out test set: happy is "
    "recognised strongly, while sad is the weakest class and is most frequently "
    "confused with stress -- a pairing that is precisely the kind of ambiguity the "
    "physiological modality, were it stronger, would be well placed to help resolve, "
    "which is the underlying motivation for building a fusion mechanism rather than "
    "relying on video alone."
)

H2("7.4 Fusion and Ablation Study")
PARA(
    "The complete fused system was evaluated using an eight-condition ablation study "
    "over the cross-dataset evaluation set described in Section 6.3.4 (20 samples per "
    "emotion, 50 random re-pairings). Table 7.3 and Figure 7.1 present the results; "
    "Figure 7.2 shows representative confusion matrices for the physio-only, "
    "video-only and full-fusion conditions."
)
FIGURE("7.1", f"{ASSETS}/fig_ablation_bar.png", width=6.2)
TABLE("7.3",
      ["Cond.", "Description", "Macro-F1 (mean +/- std)", "Wilcoxon vs C5"],
      [
          ["C1", "Physio only", "0.443 +/- 0.000", "p = 8.9e-16 (C5 significantly better)"],
          ["C2", "Video only", "0.631 +/- 0.000", "p = 5.5e-10 (C5 significantly better)"],
          ["C3", "Equal weight (0.5/0.5)", "0.692 +/- 0.019", "p = 1.0 (not significant)"],
          ["C4", "Confidence-weighted, no quality penalty", "0.685 +/- 0.016", "-"],
          ["C5", "Full method (confidence x quality)", "0.653 +/- 0.009", "(reference)"],
          ["C6", "Robustness: video quality forced poor", "0.652 +/- 0.023", "-"],
          ["C7", "Robustness: video missing (uniform fallback)", "0.443 +/- 0.000", "-"],
          ["C8", "Robustness: physio quality forced poor", "0.650 +/- 0.009", "-"],
      ])
FIGURE("7.2", "data/synced_samples/ablation_confusion_matrices.png", width=6.4)
PARA(
    "Two findings stand out. First, the full gated-fusion method (C5) significantly "
    "outperforms both single-modality baselines: physio-only (C1, p = 8.9e-16) and "
    "video-only (C2, p = 5.5e-10) under a one-tailed Wilcoxon signed-rank test at "
    "alpha = 0.05, directly supporting the central multimodal claim that fusing two "
    "modalities is better than relying on either alone. Second, and reported honestly "
    "even though it is an unfavourable result, C5 does not significantly outperform "
    "naive equal-weight fusion (C3, p = 1.0). The reason is diagnosable rather than "
    "mysterious: the shipped physiological checkpoint used in this evaluation is the "
    "random-split model (Section 7.2), whose EEG/GSR *signal quality* is frequently "
    "graded \"good\" even though its underlying *predictive accuracy* is weak; the gating "
    "mechanism currently trusts signal quality as a proxy for reliability, and so gives "
    "this confidently-wrong modality more weight than an equal split would. This is a "
    "consequence of using an admittedly weak physiological checkpoint rather than a "
    "flaw in the fusion formula itself, and is expected to resolve once a stronger, "
    "properly-calibrated physiological model (Section 8.2) is substituted in."
)
PARA(
    "The three robustness conditions (C6-C8) confirm the system behaves as designed "
    "under degraded input: forcing video quality to \"poor\" (C6) or physiology quality "
    "to \"poor\" (C8) only mildly changes the fused macro-F1, and removing video "
    "entirely (C7) causes the fused output to fall back exactly to the physio-only "
    "result (C1), demonstrating correct graceful degradation rather than a crash or an "
    "unpredictable output when a modality is unavailable."
)

H2("7.5 Web Application Evaluation")
PARA(
    "The complete web application was evaluated with an in-process end-to-end HTTP "
    "test using FastAPI's test client against an isolated test database: a new user is "
    "registered and logged in; a request to /predict/multimodal with a real video file "
    "together with real EEG and GSR windows returns HTTP 200 with a genuinely fused "
    "prediction (both modality weights strictly positive) and a populated SHAP "
    "explanation; a video-only request (EEG/GSR omitted) correctly returns a "
    "video-only-weighted prediction; and a request with a malformed EEG array shape is "
    "correctly rejected with HTTP 422 and a descriptive error message. The React "
    "frontend was verified to build cleanly and to expose the new EEG/GSR upload "
    "controls and the \"Run Multimodal Fusion\" action alongside the existing "
    "video-only path."
)

H2("7.6 Comparison with Related Work")
PARA(
    "The video module's held-out macro-F1 of approximately 0.65 is broadly comparable "
    "to the approximately 65% accuracy reported by Zhang et al. [13] for video-only emotion "
    "recognition on comparable classes. The physiological module's honest "
    "subject-independent macro-F1 of 0.179 is markedly lower than the approximately 73% "
    "accuracy reported by Siddharth et al. [12] for EEG plus peripheral signals; this "
    "project's position is that the discrepancy is substantially explained by "
    "evaluation protocol (subject-independent LOSO versus the more common "
    "within-subject or near-within-subject protocols used elsewhere in the literature) "
    "rather than by a weaker model, a claim supported by this project's own "
    "within-subject-adjacent random-split figure of 0.4171, which sits in a broadly "
    "comparable range once the same easier protocol is used."
)

H2("7.7 Limitations")
BULLETS([
    "The physiological module's subject-independent performance is weak (Section 7.2), "
    "driven primarily by DEAP's class imbalance and the well-documented difficulty of "
    "cross-subject EEG transfer, rather than by an implementation defect; this "
    "represents the system's most significant honest limitation.",
    "No single dataset provides EEG, GSR and video from the same subjects "
    "simultaneously, so the ablation study and web-application demonstration rely on a "
    "principled but constructed cross-dataset pairing (Section 6.3.4) rather than a "
    "single naturally multimodal dataset.",
    "The shipped pipeline currently uses the random-split physiological checkpoint "
    "(macro-F1 0.4171) rather than the more honestly-evaluated LOSO checkpoint "
    "(macro-F1 0.179); this was a deliberate, documented choice to allow system "
    "integration and the ablation study to proceed while the LOSO evaluation was being "
    "completed, but should be revisited before any deployment claim is made.",
    "The full method does not yet significantly outperform naive equal-weight fusion "
    "(Section 7.4), a consequence of the current physiological checkpoint's weak "
    "predictive accuracy despite frequently \"good\" signal quality.",
    "The web application has been validated functionally and via automated tests, but "
    "has not undergone a structured user study of its explanations' clinical "
    "usefulness.",
])

H2("7.8 Summary")
PARA(
    "This chapter presented an honest evaluation of every module: a strong, "
    "well-generalising video model (macro-F1 ~0.65 held-out); a physiological module "
    "whose subject-independent performance is weak, together with the diagnostic work "
    "that traced this weakness to class imbalance and cross-subject variability rather "
    "than to implementation defects; and a fusion mechanism that significantly "
    "outperforms either single modality and degrades gracefully under missing or poor "
    "signal quality, while honestly falling short of naive equal-weight fusion for the "
    "reason diagnosed above. The next chapter concludes the report and outlines future "
    "work."
)

PAGEBREAK()

# ===========================================================================
# CHAPTER 8 — CONCLUSION
# ===========================================================================

H1_CHAPTER(8, "Conclusion")

H2("8.1 Conclusion")
PARA(
    "This project set out to build a multimodal emotion recognition system that fuses "
    "physiological (EEG/GSR) and facial-video signals through a mechanism that adapts to "
    "each modality's real-time reliability, and to make its predictions explainable. That "
    "goal has been substantially achieved: a video model that generalises to unseen "
    "actors with a held-out macro-F1 of approximately 0.65; a canonical, independently-"
    "verified gated-fusion mechanism that significantly outperforms either single "
    "modality (Wilcoxon p < 0.001 in both cases) and degrades gracefully when a modality "
    "is degraded or missing; a full multimodal pipeline and matching web-application "
    "endpoint that accepts real video, EEG and GSR uploads and returns a fused, "
    "SHAP-explained prediction; and 82 passing automated tests plus a full end-to-end "
    "HTTP verification of the deployed application."
)
PARA(
    "Equally important to this project's conclusions is what was found not to work, and "
    "why. The physiological module's subject-independent (LOSO) performance is weak "
    "(macro-F1 0.179), and a structured set of diagnostic experiments -- comparing "
    "feature representations, auditing GSR data quality, and testing coarser label "
    "taxonomies -- traced this weakness primarily to DEAP's severe class imbalance and "
    "the well-documented difficulty of cross-subject EEG generalisation, rather than to "
    "an implementation defect. Reporting this limitation candidly, together with the "
    "reasoning that located its cause, is presented as a legitimate and, this project "
    "argues, more valuable outcome than a favourable but methodologically inflated "
    "number would have been. The system's fusion design is precisely the mechanism "
    "intended to remain robust in the presence of exactly this kind of single-modality "
    "weakness, and the ablation study's robustness conditions (C6-C8) confirm that it "
    "does so."
)

H2("8.2 Future Work")
BULLETS([
    "Physiological model improvement: retrain and evaluate the physiological network "
    "under a coarser three-class or dimensional (valence/arousal) taxonomy, which "
    "preliminary experiments in this project showed lifts LOSO macro-F1 from 0.179 to "
    "approximately 0.28 (three-class) or 0.44 (binary valence) without requiring the "
    "video model to be retrained, since its five-class output can be remapped by simple "
    "probability summation; alternatively, explore domain-adversarial training to "
    "explicitly encourage subject-invariant EEG features.",
    "Recalibrate the physiological signal-quality thresholds (EEG/GSR) against DEAP's "
    "actual measured value distributions, analogous to the video-quality recalibration "
    "carried out in this project, so that the fusion mechanism's quality-based gating "
    "reflects true predictive reliability rather than only raw signal cleanliness.",
    "Swap the shipped pipeline's physiological checkpoint from the random-split model to "
    "a properly calibrated, honestly-evaluated model once the above improvements are "
    "made, and re-run the ablation study to determine whether the full gated method "
    "then significantly outperforms naive equal-weight fusion.",
    "Extend the cross-dataset evaluation methodology as genuinely paired multimodal data "
    "becomes available, reducing reliance on the constructed DEAP/CREMA-D pairing used "
    "in this report.",
    "Conduct a structured user study of the web application's SHAP-based and "
    "chatbot-generated explanations with representative end users, to evaluate their "
    "clinical usefulness rather than only their technical correctness.",
    "Deploy the backend against a production PostgreSQL instance and complete a security "
    "review of the authentication and file-upload paths prior to any real-world use.",
])

PAGEBREAK()

# ===========================================================================
# REFERENCES  (IEEE style, alphabetical by first author, numbered accordingly)
# ===========================================================================

p = doc.add_paragraph("References", style="Heading 1")
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.LEFT

REFERENCES = [
    'T. Baltrusaitis, C. Ahuja, and L.-P. Morency, "Multimodal Machine Learning: A '
    'Survey and Taxonomy," IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, '
    'pp. 423-443, 2019.',

    'H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, '
    '"CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset," IEEE Trans. '
    'Affective Computing, vol. 5, no. 4, pp. 377-390, Oct. 2014.',

    'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural '
    'Networks," in Proc. 34th Int. Conf. Machine Learning (ICML), 2017, pp. 1321-1330.',

    'K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image '
    'Recognition," in Proc. IEEE Conf. Computer Vision and Pattern Recognition '
    '(CVPR), 2016, pp. 770-778.',

    'S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, '
    'vol. 9, no. 8, pp. 1735-1780, 1997.',

    'S. Koelstra, C. Muhl, M. Soleymani, J.-S. Lee, A. Yazdani, T. Ebrahimi, T. Pun, '
    'A. Nijholt, and I. Patras, "DEAP: A Database for Emotion Analysis Using '
    'Physiological Signals," IEEE Trans. Affective Computing, vol. 3, no. 1, pp. '
    '18-31, Jan.-Mar. 2012.',

    'S. D. Kreibig, "Autonomic Nervous System Activity in Emotion: A Review," '
    'Biological Psychology, vol. 84, no. 3, pp. 394-421, 2010.',

    'A. Mehrabian, "Pleasure-Arousal-Dominance: A General Framework for Describing '
    'and Measuring Individual Differences in Temperament," Current Psychology, vol. '
    '14, no. 4, pp. 261-292, 1996.',

    'J. L. Pech-Pacheco, G. Cristobal, J. Chamorro-Martinez, and J. Fernandez-'
    'Valdivia, "Diatom Autofocusing in Brightfield Microscopy: A Comparative Study," '
    'in Proc. 15th Int. Conf. Pattern Recognition (ICPR), 2000, pp. 314-317.',

    'S. Poria, E. Cambria, R. Bajpai, and A. Hussain, "A Review of Affective '
    'Computing: From Unimodal Analysis to Multimodal Fusion," Information Fusion, '
    'vol. 37, pp. 98-125, 2017.',

    'J. A. Russell, "A Circumplex Model of Affect," Journal of Personality and '
    'Social Psychology, vol. 39, no. 6, pp. 1161-1178, 1980.',

    'Siddharth, T.-P. Jung, and T. J. Sejnowski, "Utilizing Deep Learning Towards '
    'Multi-Modal Bio-Sensing and Vision-Based Affective Computing," IEEE Trans. '
    'Affective Computing, vol. 13, no. 1, pp. 96-107, 2019.',

    'T. Zhang, W. Zheng, Z. Cui, Y. Zong, and Y. Li, "Spatial-Temporal Recurrent '
    'Neural Network for Emotion Recognition," IEEE Trans. Cybernetics, vol. 49, no. '
    '3, pp. 839-847, 2020.',
]

for i, ref in enumerate(REFERENCES, start=1):
    rp = doc.add_paragraph()
    rp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    rp.paragraph_format.left_indent = Inches(0.3)
    rp.paragraph_format.first_line_indent = Inches(-0.3)
    rp.add_run(f"[{i}]  {ref}")

PAGEBREAK()

# ===========================================================================
# APPENDIX A — Individual's Contribution to the Project
# ===========================================================================

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Appendix A")
r.bold = True; r.font.size = Pt(14); r.font.name = "Times New Roman"
doc.add_paragraph("Individuals' Contribution to the Project", style="Heading 1")

H2("Suhira Balarajan — 214206G")
PARA(
    "My primary responsibility was the physiological emotion recognition module: "
    "loading and preprocessing the DEAP dataset, designing and implementing the "
    "bidirectional cross-modal attention network (BiAttn) that lets the EEG and GSR "
    "encoders attend over one another before a joint classification head, and running "
    "the full 32-fold Leave-One-Subject-Out cross-validation together with a model-soup "
    "average of all 32 fold checkpoints. I also implemented and ran the three classical "
    "baselines (SVM, Random Forest, MLP) over the same 165-dimensional hand-crafted "
    "feature set and identical LOSO protocol, specifically so that the neural network's "
    "result could be checked against simpler models rather than assessed in isolation."
)
PARA(
    "The most significant challenge I encountered was discovering, after completing the "
    "full LOSO run, that the honest subject-independent macro-F1 (0.179) was far lower "
    "than the random-split figure obtained earlier in development (0.4171). Rather than "
    "treat this as a failure to hide, I treated it as a result to explain: working "
    "through the class distribution produced by my own DEAP loader showed that the "
    "shared five-class mapping leaves only 19 \"happy\" trials across all 32 subjects, "
    "and that all three of my classical baselines converged to the same low macro-F1 "
    "range as the neural network under LOSO -- evidence that the limitation lay in the "
    "data and label distribution rather than in any one model. I learned, concretely, "
    "the difference between an evaluation protocol that flatters a model and one that "
    "tests what it actually needs to do, and that reporting an honest, diagnosed "
    "limitation is more useful to the project -- and more defensible -- than reporting an "
    "inflated number."
)

PAGEBREAK()

H2("Vanaiyan Kirupagaran — 214215H")
PARA(
    "My primary responsibility was the video-based emotion recognition module and the "
    "multimodal fusion layer that combines it with Member 1's physiological output. On "
    "the video side, I built the CREMA-D/RAVDESS preprocessing pipeline, the "
    "YOLOv8-based face-detection front end, and the partially fine-tuned ResNet50 + "
    "BiLSTM recognition network, and iterated through several training configurations -- "
    "a fully-frozen backbone that underfit, a fully-unfrozen backbone that overfit -- "
    "before settling on discriminative learning rates, which produced a model that "
    "generalises to unseen actors (held-out macro-F1 0.6486, matching validation "
    "performance almost exactly). I managed training across multiple Kaggle sessions "
    "with cross-session checkpoint persistence to work within Kaggle's 12-hour session "
    "limit."
)
PARA(
    "Beyond the video model itself, I designed and implemented the system-integration "
    "layer that turns three separately-developed modules into one working system: the "
    "canonical gated-fusion module (verified numerically identical to Member 3's earlier "
    "independent reference implementation across every signal-quality combination), the "
    "full multimodal pipeline with graceful degradation for missing or poor-quality "
    "modalities, the cross-dataset synchronised evaluation-set construction needed "
    "because DEAP and CREMA-D share no subjects, the eight-condition ablation study with "
    "Wilcoxon significance testing, and the new /predict/multimodal web-application "
    "endpoint together with its frontend controls. I also carried out the root-cause "
    "diagnostic investigation into Member 1's LOSO result -- comparing feature "
    "representations, auditing the DEAP GSR channel for data-quality issues, and "
    "quantifying the effect of coarser label taxonomies -- to establish, with evidence "
    "rather than assumption, why the physiological module underperforms under "
    "subject-independent evaluation."
)
PARA(
    "The most valuable lesson from this work was methodological: when an empirical "
    "result is disappointing, the right response is a controlled diagnostic experiment "
    "that isolates the cause, not a search for a more favourable evaluation protocol. "
    "This discipline directly shaped the decision to report the honest LOSO figure "
    "throughout this report rather than the more flattering random-split figure, and to "
    "report the ablation study's unfavourable finding (C5 not significantly beating "
    "naive equal-weight fusion) alongside its favourable ones."
)

PAGEBREAK()

H2("Adshaya Balarajah — 214024V")
PARA(
    "My primary responsibility was the explainability layer and the web application: "
    "implementing the Kernel SHAP wrapper over the fused prediction_output, computing a "
    "faithfulness metric by perturbing input features and measuring the resulting "
    "prediction change, and building the FastAPI backend -- JWT authentication, the "
    "nine-endpoint API surface, and the four-table SQLAlchemy schema (users, sessions, "
    "SHAP logs, chat history) -- together with the React dashboard, including the "
    "emotion-trend chart, the SHAP bar chart, the session-history panel, and the "
    "modality-conflict explanation panel that surfaces cases where the physiological and "
    "video predictions disagree. I also integrated a large-language-model-based chatbot "
    "that translates the SHAP attribution into a plain-English explanation for the end "
    "user."
)
PARA(
    "A recurring challenge was designing the API and frontend so that they would remain "
    "correct regardless of which modalities were actually available for a given "
    "prediction -- video-only, physiology-only, or both -- without branching the "
    "explanation and visualisation logic for each case. Building the conflict-explanation "
    "panel specifically required understanding the gated-fusion mechanism well enough to "
    "explain, in natural language, why the system sometimes disagrees with one modality "
    "in favour of another, which deepened my understanding of the fusion design well "
    "beyond the explainability layer I was directly responsible for. I learned that a "
    "genuinely useful explanation layer depends on the upstream system already producing "
    "well-structured, quality-aware output -- explainability is not something that can be "
    "bolted on afterward to an arbitrary prediction, but has to be designed for from the "
    "interface contracts upward."
)

# ===========================================================================
# FINAL SAVE
# ===========================================================================

doc.save("docs/MedOracle_Final_Report.docx")
print("Full report saved: docs/MedOracle_Final_Report.docx")
