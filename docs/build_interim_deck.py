"""
build_interim_deck.py
=====================
Generates MedOracle_Interim_Presentation.pptx — a 14-slide, 15-minute interim
review deck for the MedOracle FYP, grounded in the Interim Report and the latest
Member 2 training results. Equal 3-way split across Members 1, 2, 3.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1F, 0x3B, 0x5B)   # primary
TEAL   = RGBColor(0x2A, 0x9D, 0x8F)   # accent
AMBER  = RGBColor(0xE9, 0xA8, 0x4A)   # secondary accent
DARK   = RGBColor(0x22, 0x2A, 0x33)   # body text
GREY   = RGBColor(0x5A, 0x63, 0x6E)   # muted
LIGHT  = RGBColor(0xF4, 0xF6, 0xF8)   # panel bg
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

FONT = "Calibri"


def _set(run, size, color, bold=False, italic=False, font=FONT):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def rect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def footer(slide, n):
    rect(slide, 0, SH - Inches(0.32), SW, Inches(0.32), NAVY)
    tf = textbox(slide, Inches(0.4), SH - Inches(0.34), Inches(9), Inches(0.3),
                 MSO_ANCHOR.MIDDLE)
    r = tf.paragraphs[0].add_run()
    r.text = "MedOracle  ·  Interim Review 2026  ·  University of Moratuwa"
    _set(r, 10, WHITE)
    tf2 = textbox(slide, SW - Inches(1.2), SH - Inches(0.34), Inches(0.8), Inches(0.3),
                  MSO_ANCHOR.MIDDLE)
    p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r2 = p.add_run(); r2.text = str(n); _set(r2, 10, WHITE)


def header(slide, title, tag=None, tag_color=TEAL):
    """Navy header bar with title + optional section tag chip."""
    rect(slide, 0, 0, SW, Inches(1.15), NAVY)
    rect(slide, 0, Inches(1.15), SW, Inches(0.07), TEAL)
    tf = textbox(slide, Inches(0.5), Inches(0.18), Inches(9.8), Inches(0.85),
                 MSO_ANCHOR.MIDDLE)
    r = tf.paragraphs[0].add_run(); r.text = title; _set(r, 28, WHITE, bold=True)
    if tag:
        chip = rect(slide, SW - Inches(3.4), Inches(0.34), Inches(2.9), Inches(0.5), tag_color)
        ctf = chip.text_frame; ctf.word_wrap = True
        ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
        cp = ctf.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
        cr = cp.add_run(); cr.text = tag; _set(cr, 13, WHITE, bold=True)


def bullets(slide, items, x=Inches(0.6), y=Inches(1.5),
            w=Inches(12.1), h=Inches(5.5), size=17, gap=6):
    """items: list of (text, level) tuples or strings (level 0)."""
    tf = textbox(slide, x, y, w, h)
    first = True
    for it in items:
        text, lvl = (it if isinstance(it, tuple) else (it, 0))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap); p.space_before = Pt(0)
        p.level = lvl
        if lvl == 0:
            r = p.add_run(); r.text = "▸ "; _set(r, size, TEAL, bold=True)
            r2 = p.add_run(); r2.text = text; _set(r2, size, DARK, bold=False)
        elif lvl == 1:
            r = p.add_run(); r.text = "     – "; _set(r, size - 2, AMBER, bold=True)
            r2 = p.add_run(); r2.text = text; _set(r2, size - 2, GREY)
        else:
            r = p.add_run(); r.text = "          • "; _set(r, size - 3, GREY)
            r2 = p.add_run(); r2.text = text; _set(r2, size - 3, GREY)
    return tf


def two_panel(slide, left_title, left_items, right_title, right_items,
              ltag=TEAL, rtag=NAVY):
    """Two side-by-side panels for compact comparisons."""
    pw = Inches(6.0); ph = Inches(5.3); top = Inches(1.5)
    lx = Inches(0.45); rx = Inches(6.85)
    for (px, ptitle, pitems, pc) in [(lx, left_title, left_items, ltag),
                                     (rx, right_title, right_items, rtag)]:
        rect(slide, px, top, pw, Inches(0.55), pc)
        ttf = textbox(slide, px + Inches(0.2), top, pw - Inches(0.4), Inches(0.55),
                      MSO_ANCHOR.MIDDLE)
        tr = ttf.paragraphs[0].add_run(); tr.text = ptitle; _set(tr, 15, WHITE, bold=True)
        rect(slide, px, top + Inches(0.55), pw, ph - Inches(0.55), LIGHT)
        bullets(slide, pitems, x=px + Inches(0.2), y=top + Inches(0.72),
                w=pw - Inches(0.4), h=ph - Inches(0.8), size=14, gap=5)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(4.55), SW, Inches(0.09), TEAL)
rect(s, 0, Inches(4.64), SW, Inches(0.05), AMBER)

tf = textbox(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.2))
r = tf.paragraphs[0].add_run(); r.text = "MedOracle"; _set(r, 60, WHITE, bold=True)
tf2 = textbox(s, Inches(0.9), Inches(2.7), Inches(11.5), Inches(1.6))
r = tf2.paragraphs[0].add_run()
r.text = "A Multimodal Emotion Recognition System with Explainability"
_set(r, 26, TEAL, bold=True)
p = tf2.add_paragraph(); p.space_before = Pt(8)
r = p.add_run(); r.text = "Fusing EEG · GSR · Facial Video → 5 emotions, with SHAP explanations"
_set(r, 16, RGBColor(0xC9, 0xD3, 0xDE), italic=True)

tf3 = textbox(s, Inches(0.9), Inches(4.95), Inches(11.5), Inches(2.0))
rows = [
    ("Suhira Balarajan", "214206G", "Member 1 — Physiological (EEG + GSR)"),
    ("Vanaiyan Kirupagaran", "214215H", "Member 2 — Video + Fusion"),
    ("Adshaya Balarajah", "214024V", "Member 3 — Explainability + Web App"),
]
first = True
for name, idn, role in rows:
    p = tf3.paragraphs[0] if first else tf3.add_paragraph(); first = False
    p.space_after = Pt(4)
    r = p.add_run(); r.text = f"{name}  "; _set(r, 15, WHITE, bold=True)
    r = p.add_run(); r.text = f"({idn})   "; _set(r, 14, AMBER, bold=True)
    r = p.add_run(); r.text = role; _set(r, 14, RGBColor(0xC9, 0xD3, 0xDE))
p = tf3.add_paragraph(); p.space_before = Pt(12)
r = p.add_run(); r.text = "Supervisor: Dr. Firdhous M.F.M.   ·   Faculty of Information Technology, University of Moratuwa   ·   Level 4 FYP 2026"
_set(r, 12, RGBColor(0x9F, 0xAE, 0xC0), italic=True)
notes(s, "Welcome. We are team MedOracle. Our project is a multimodal emotion "
         "recognition system that fuses physiological signals and facial video, "
         "and crucially explains its predictions. I'm [name], with me are [names]. "
         "Over the next 15 minutes we'll cover the problem, our architecture, the "
         "datasets, then each of us presents our module's progress, and finish "
         "with results and remaining work.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem & Motivation
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "The Problem & Motivation")
bullets(s, [
    "Emotion recognition matters for mental-health monitoring, human–computer interaction, and personalised healthcare.",
    "Most existing systems rely on a SINGLE modality — either facial expression OR physiological signals.",
    ("Facial-only systems break under poor lighting, occlusion, and head pose.", 1),
    ("Physiological-only systems are sensitive to sensor noise and motion artefacts.", 1),
    "In the real world, signal quality degrades — a robust system must handle this gracefully.",
    "Black-box predictions are unacceptable in healthcare — clinicians need to know WHY.",
    "Our answer: fuse complementary modalities, weight them by signal quality, and explain every prediction.",
], size=18, gap=10)
footer(s, 2)
notes(s, "Single-modality emotion recognition is brittle. Cameras fail in bad "
         "lighting; EEG/GSR sensors pick up noise. Real deployments face degraded "
         "signals. And in healthcare a black-box answer isn't enough — you need "
         "explainability. MedOracle tackles all three: multimodal fusion, "
         "quality-aware weighting, and SHAP-based explanations. ~1 min.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Aim & Objectives
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Aim & Objectives")
tf = textbox(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(0.9))
r = tf.paragraphs[0].add_run()
r.text = ("Aim:  Build a robust, explainable system that classifies five emotional "
          "states by fusing EEG, GSR, and facial video.")
_set(r, 18, NAVY, bold=True)

# 5 emotion chips
emos = ["Stress", "Calm", "Happy", "Sad", "Angry"]
cw = Inches(2.3); gap = Inches(0.2); x0 = Inches(0.7); y0 = Inches(2.45)
for i, e in enumerate(emos):
    c = rect(s, x0 + i * (cw + gap), y0, cw, Inches(0.7), TEAL if i % 2 == 0 else NAVY)
    ctf = c.text_frame; ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
    cp = ctf.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
    cr = cp.add_run(); cr.text = e; _set(cr, 17, WHITE, bold=True)

bullets(s, [
    "Objective 1 — Physiological module: emotion recognition from EEG + GSR (DEAP).",
    "Objective 2 — Video module: emotion recognition from facial video (CREMA-D + RAVDESS).",
    "Objective 3 — Quality-aware gated fusion that degrades gracefully when a modality is poor or missing.",
    "Objective 4 — Explainability layer (Kernel SHAP) + web dashboard with an LLM chatbot.",
    "Objective 5 — Rigorous evaluation: per-module cross-validation + a 5-condition ablation study.",
], y=Inches(3.4), size=16, gap=8)
footer(s, 3)
notes(s, "Our aim: classify five emotions — stress, calm, happy, sad, angry — by "
         "fusing three signal types. Five objectives, one per major component: the "
         "physiological model, the video model, the quality-aware fusion, the "
         "explainability + web app, and a rigorous evaluation plan. Note we locked "
         "to exactly 5 classes so all modules speak the same language. ~1 min.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — System Architecture
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "System Architecture — End-to-End Pipeline")

def flowbox(x, y, w, h, title, sub, fill):
    b = rect(s, x, y, w, h, fill)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = title; _set(r, 14, WHITE, bold=True)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = sub; _set(r, 10, RGBColor(0xE6, 0xEC, 0xF2))
    return b

def arrow(x, y, w):
    from pptx.enum.shapes import MSO_SHAPE
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, Inches(0.35))
    a.fill.solid(); a.fill.fore_color.rgb = AMBER; a.line.fill.background()
    a.shadow.inherit = False

yA = Inches(2.1); bh = Inches(1.5); bw = Inches(3.0)
flowbox(Inches(0.5), yA, bw, bh, "M1 · Physiological", "EEG+GSR → Bi-Cross-Attention", NAVY)
arrow(Inches(3.6), yA + Inches(0.55), Inches(0.7))
flowbox(Inches(4.45), yA, bw, bh, "M2 · Video + Fusion", "ResNet50→BiLSTM + Gated Fusion", TEAL)
arrow(Inches(7.55), yA + Inches(0.55), Inches(0.7))
flowbox(Inches(8.4), yA, bw, bh, "M3 · Explainability", "Kernel SHAP + FastAPI + React", NAVY)
arrow(Inches(11.5), yA + Inches(0.55), Inches(0.7))
# video input feeding M2
flowbox(Inches(4.45), Inches(0.0) + Inches(3.95), bw, Inches(0.8),
        "Facial video", "T=16 frames", GREY)
# small connector note
bullets(s, [
    "Late (decision-level) fusion — forced by design: DEAP and CREMA-D share NO subjects, so joint training is impossible.",
    "Modules communicate through strict interface contracts (validated Python dicts) — enabling independent development.",
    "Output: predicted emotion + class probabilities + per-modality weights + signal-quality flags + SHAP attributions.",
], y=Inches(5.0), size=15, gap=8)
footer(s, 4)
notes(s, "Three modules in a pipeline. Member 1 turns EEG and GSR into a "
         "physiological prediction. Member 2 turns facial video into a video "
         "prediction and then fuses both. Member 3 explains the result and serves "
         "it through a web app. A key design point: we use LATE fusion because our "
         "two datasets have no overlapping subjects — joint training is impossible, "
         "so fusing at the decision level is architecturally forced, not a "
         "weakness. The modules talk through strict contracts, which let us build "
         "in parallel. ~1.5 min.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Datasets & Label Harmonisation
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Datasets & Label Harmonisation")
two_panel(s,
    "DEAP  (physiological)",
    [
        "32 subjects · EEG (32ch, 128 Hz) + GSR",
        "Continuous Valence / Arousal / Dominance",
        ("Mapped to 5 classes via V/A/D thresholds", 1),
        ("e.g. low-V / high-A / low-D → stress", 2),
        ("Grounded in Russell (1980), Mehrabian (1996)", 1),
    ],
    "CREMA-D + RAVDESS  (video)",
    [
        "CREMA-D: 91 actors · 7,442 clips → 6,171",
        "RAVDESS: 24 actors  (added for diversity)",
        ("ANG→angry, HAP→happy, SAD→sad", 1),
        ("NEU→calm, FEA→stress (Kreibig 2010)", 1),
        ("DIS (disgust) → dropped: no clean mapping", 1),
    ])
tf = textbox(s, Inches(0.45), Inches(6.85), Inches(12.4), Inches(0.5))
r = tf.paragraphs[0].add_run()
r.text = ("Shared label_harmonization.py enforces one 5-class taxonomy across all members — "
          "validated by 59 passing unit tests.")
_set(r, 14, NAVY, bold=True)
footer(s, 5)
notes(s, "Two very different datasets. DEAP gives physiological signals labelled "
         "on continuous valence-arousal-dominance scales; we threshold those into "
         "our 5 classes using established affective-computing theory. The video side "
         "uses CREMA-D plus RAVDESS; we map their categorical labels, mapping fear "
         "to stress and neutral to calm, and dropping disgust which has no clean "
         "match. A shared harmonisation module — with 59 unit tests — guarantees all "
         "three modules use exactly the same 5 classes. ~1.5 min.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Member 1 — Preprocessing
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Physiological Module — Signal Pipeline", tag="Member 1 · Suhira", tag_color=NAVY)
bullets(s, [
    "DEAP dataset (Queen Mary U.) downloaded & verified against official spec — all 32 subjects, channel config, sampling rate, V/A/D ranges.",
    "EEG + GSR preprocessing implemented in Python / NumPy / PyTorch (Google Colab):",
    ("Subject-wise loading · epoch segmentation into 4-second windows", 1),
    ("EEG channel extraction · GSR signal extraction", 1),
    "Feature extraction designed:",
    ("EEG — Differential Entropy across delta/theta/alpha/beta/gamma bands (all 32 channels)", 1),
    ("GSR — temporal & variability statistics (physiological arousal)", 1),
    "Signal-quality grading (good / degraded / poor) designed to feed Member 2's gated fusion.",
    "Shared 5-class label harmonisation integrated into the physiological pipeline.",
], size=16, gap=7)
footer(s, 6)
notes(s, "[Member 1 presents] I own the physiological module. First I verified the "
         "DEAP dataset matched the official Queen Mary spec — channels, sampling "
         "rate, label ranges — with automated checks. Then I built the EEG and GSR "
         "preprocessing: loading per subject, cutting signals into 4-second epochs, "
         "and extracting features — Differential Entropy across the five EEG "
         "frequency bands, and statistical features for GSR. I also designed the "
         "signal-quality grading that Member 2's fusion uses. ~2 min total for my part.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Member 1 — Model & Status
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Physiological Module — Model & Status", tag="Member 1 · Suhira", tag_color=NAVY)
two_panel(s,
    "Architecture (designed)",
    [
        "Bidirectional Cross-Modal Attention Network",
        ("EEG attends to GSR, and GSR attends to EEG", 1),
        ("Captures interaction between the two signals", 1),
        "Joint classification head → 5-class softmax",
        "Exports predict_physiological() → contract dict",
    ],
    "Evaluation & status",
    [
        "LOSO — Leave-One-Subject-Out (32 folds)",
        ("Tests subject-independent generalisation", 1),
        "Class weighting for imbalance (no SMOTE on EEG)",
        "Metric: mean ± std macro-F1 across 32 folds",
        "Status: preprocessing validated; training & LOSO in progress",
    ], ltag=NAVY, rtag=TEAL)
footer(s, 7)
notes(s, "[Member 1] My model is a bidirectional cross-modal attention network: "
         "EEG attends to GSR and vice-versa, so the model learns how the two "
         "signals interact, before a joint head outputs the 5 classes. For "
         "evaluation I use Leave-One-Subject-Out across all 32 subjects — the "
         "strictest test of generalising to unseen people — reporting mean macro-F1. "
         "Preprocessing is validated; training and LOSO evaluation are underway. "
         "Handing over to Member 2.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Member 2 — Video Pipeline & Model
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Video Module — Preprocessing & Model", tag="Member 2 · Vanaiyan", tag_color=TEAL)
two_panel(s,
    "Preprocessing pipeline (built & verified)",
    [
        "YOLOv8 face detection (+ Haar cascade fallback)",
        "Uniform sampling of T=16 frames per clip",
        "Crop + pad → resize to 224×224",
        "Quality assessment: Laplacian variance (sharpness)",
        ("+ face bounding-box area (good/degraded/poor)", 1),
        "Verified on all 5 emotion classes locally",
    ],
    "Model: ResNet50 → BiLSTM",
    [
        "ResNet50 encoder, partial fine-tune (ImageNet)",
        ("2048-d feature per frame", 1),
        "2-layer Bidirectional LSTM (temporal modelling)",
        ("mean-pooled over 16 frames → 5 logits", 1),
        "~30M params · all smoke tests pass (CPU/MPS/CUDA)",
        "Actor-independent 5-fold (no actor in train+val)",
    ], ltag=TEAL, rtag=NAVY)
footer(s, 8)
notes(s, "[Member 2 — Vanaiyan presents] I own the video and fusion module. The "
         "preprocessing detects the face with YOLOv8, samples 16 frames evenly "
         "across the clip, crops to 224×224, and grades video quality using image "
         "sharpness and face size. The model is a ResNet50 that turns each frame "
         "into a feature vector, fed into a bidirectional LSTM that models how the "
         "expression evolves over time. Critically, my 5-fold split is "
         "actor-independent — no actor appears in both train and test — so the model "
         "can't cheat by memorising faces. ~2 min for my part.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Member 2 — Gated Fusion
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Quality-Aware Gated Fusion", tag="Member 2 · Vanaiyan", tag_color=TEAL)
bullets(s, [
    "Each modality gets a confidence score from prediction entropy:  c = 1 − H(P)/log K   (Guo et al., 2017).",
    "A quality penalty α scales confidence by signal quality:  good = 1.0 · degraded = 0.5 · poor = 0.1.",
    "Gate score per modality:  g = c × α.   Weights via L1 normalisation:  w = g / Σ g.",
    ("L1 (not softmax) preserves large confidence gaps between modalities.", 1),
    "Fused prediction:  P_fused = Σ w·P  →  argmax = final emotion.",
    "Graceful degradation:",
    ("Video poor/missing → rely on physiological;  physiological poor/missing → rely on video", 1),
    ("Both poor → uniform distribution, flagged unreliable", 1),
], size=16, gap=9)
footer(s, 9)
notes(s, "[Member 2] This is the heart of robustness. Each modality gets a "
         "confidence from how peaked its probability distribution is. We multiply "
         "that by a quality penalty — a clean signal keeps full weight, a poor one "
         "is cut to 10%. Those gate scores become weights via L1 normalisation, "
         "which preserves big confidence gaps better than softmax. We take the "
         "weighted average of the two predictions. And if a modality drops out "
         "entirely, the system gracefully falls back to the other one rather than "
         "failing. I designed this fusion algorithm and the quality thresholds for "
         "the whole team.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Member 2 — Preliminary Results
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Video Model — Preliminary Results", tag="Member 2 · Vanaiyan", tag_color=TEAL)

# Results table
from pptx.enum.shapes import MSO_SHAPE
tbl_x, tbl_y = Inches(0.5), Inches(1.5)
rows_data = [
    ("Run", "Setup", "Val macro-F1"),
    ("v1 baseline", "CREMA-D only", "0.640"),
    ("v3.1  Fold 1", "CREMA-D + RAVDESS, anti-overfit", "0.605"),
    ("v3.1  Fold 2", "CREMA-D + RAVDESS, anti-overfit", "0.663"),
    ("v3.1  Folds 3–5", "5-fold run in progress", "pending"),
]
gt = s.shapes.add_table(len(rows_data), 3, tbl_x, tbl_y, Inches(7.0), Inches(2.6)).table
gt.columns[0].width = Inches(2.0); gt.columns[1].width = Inches(3.4); gt.columns[2].width = Inches(1.6)
for ci in range(3):
    cell = gt.cell(0, ci); cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    p = cell.text_frame.paragraphs[0]; r = p.add_run(); r.text = rows_data[0][ci]
    _set(r, 13, WHITE, bold=True)
for ri in range(1, len(rows_data)):
    for ci in range(3):
        cell = gt.cell(ri, ci)
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
        p = cell.text_frame.paragraphs[0]; r = p.add_run(); r.text = rows_data[ri][ci]
        bold = (ci == 2 and ri in (2, 3))
        _set(r, 12, NAVY if ci == 2 else DARK, bold=bold)

bullets(s, [
    "Diagnosed over-fitting (train≫val) — model was memorising actor identity.",
    "Fixes applied:",
    ("Frozen-BatchNorm + discriminative LR (layer4 @1e-5, head @5e-4)", 1),
    ("Cutout augmentation · BiLSTM mean-pooling · StratifiedGroupKFold", 1),
    "Train/val gap now small & stable — healthy generalisation.",
    "Strong per-class: happy 0.80, angry 0.70;  fusion should lift sad/stress.",
], x=Inches(7.8), y=Inches(1.5), w=Inches(5.3), h=Inches(5.0), size=14, gap=7)
footer(s, 10)
notes(s, "[Member 2] Honest preliminary results. My first baseline hit 0.64 "
         "macro-F1 but was over-fitting — the train score was far above validation, "
         "meaning it was memorising specific actors' faces rather than emotions. I "
         "diagnosed that and applied targeted fixes: freezing BatchNorm, training "
         "the backbone with a much lower learning rate than the head, cutout "
         "augmentation, and stratified actor-independent folds. The current v3.1 run "
         "shows Fold 1 at 0.605 and Fold 2 at 0.663, with a small healthy train/val "
         "gap. Happy and angry are already strong; sad and stress are the weak spots "
         "— exactly where fusion with the physiological signal should help. Over to "
         "Member 3.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Member 3 — Explainability
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Explainability Layer — Kernel SHAP", tag="Member 3 · Adshaya", tag_color=NAVY)
bullets(s, [
    "Kernel SHAP on the fused pipeline — each modality (EEG, GSR, video) treated as one feature.",
    ("Evaluates all 4 coalitions → signed SHAP value = each modality's contribution to the winning class.", 1),
    "Challenge: gated fusion uses discrete quality grades → non-differentiable → unstable SHAP values.",
    ("Solved by fixing quality grades during each SHAP pass; only probabilities vary → stable, reproducible.", 1),
    "Faithfulness metric — does the explanation reflect the real decision?",
    ("Masks modalities by SHAP importance, measures confidence drop, Spearman-correlates the two orderings.", 1),
    ("Epsilon guard fixed a divide-by-zero on near-uniform distributions; all synthetic tests pass.", 1),
    "Pydantic contract (7 fields, prob-sum & range checks) + synthetic data generator (4 scenarios) underpin all testing.",
], size=15, gap=7)
footer(s, 11)
notes(s, "[Member 3 — Adshaya presents] My job is to make the system "
         "explainable and put it on the web. I use Kernel SHAP, treating each "
         "modality as a feature, to compute how much EEG, GSR, and video each "
         "contributed to the predicted emotion. A real challenge: our fusion uses "
         "discrete quality grades, which makes it non-differentiable and made SHAP "
         "values unstable — I fixed that by holding quality grades constant during "
         "each SHAP pass. I also built a faithfulness metric that checks the "
         "explanation actually matches the model's behaviour. To develop "
         "independently of the other modules, I built a Pydantic contract and a "
         "synthetic data generator. ~2 min for my part.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Member 3 — Backend & Web App
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Backend & Web Application", tag="Member 3 · Adshaya", tag_color=NAVY)
two_panel(s,
    "Backend — FastAPI (done / in progress)",
    [
        "Auth: register + login, bcrypt hashing, JWT tokens ✓",
        "JWT middleware on all protected routes ✓",
        "4-table schema (users, sessions, shap_logs, chat) ✓",
        ("SQLAlchemy ORM + Alembic migrations", 1),
        "POST /predict endpoint implemented & tested ✓",
        "Remaining 8 endpoints — in progress",
    ],
    "Frontend + Chatbot (in progress)",
    [
        "React dashboard — emotion trends & SHAP charts",
        ("Recharts: line chart + SHAP bar chart", 1),
        "Session history + detail views",
        "LLM chatbot — Anthropic Claude API",
        ("Translates SHAP values → plain-English answers", 1),
        "Primary remaining deliverable for the final stage",
    ], ltag=NAVY, rtag=TEAL)
footer(s, 12)
notes(s, "[Member 3] On the backend, the FastAPI authentication is complete — "
         "registration, login, JWT, and a four-table database — and the main "
         "/predict endpoint works. The remaining endpoints are in progress. The "
         "React dashboard will visualise emotion trends and SHAP attributions, and "
         "an LLM chatbot using the Claude API will turn the SHAP numbers into "
         "plain-English explanations a clinician can read. The frontend and chatbot "
         "are my main deliverables for the final stage. Handing back for the wrap-up.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Progress Summary & Evaluation Plan
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Progress Summary & Evaluation Plan")
two_panel(s,
    "Done at interim",
    [
        "Full architecture + interface contracts finalised",
        "Label harmonisation — 59 unit tests pass",
        "M1: DEAP verified, EEG/GSR preprocessing + features",
        "M2: full video pipeline + model; training underway",
        "M3: Kernel SHAP + faithfulness; auth + /predict done",
    ],
    "Remaining work",
    [
        "M1: train Bi-attention net, LOSO over 32 subjects",
        "M2: finish 5-fold, build fusion.py + pipeline.py",
        "M3: 8 endpoints, React dashboard, LLM chatbot",
        "Integration: run_full_pipeline() across all modules",
        "Ablation study C1–C8 + Wilcoxon significance tests",
    ], ltag=TEAL, rtag=AMBER)
tf = textbox(s, Inches(0.45), Inches(6.85), Inches(12.4), Inches(0.5))
r = tf.paragraphs[0].add_run()
r.text = ("Evaluation: per-module CV (LOSO / 5-fold macro-F1) → full-pipeline ablation (C1–C5) "
          "+ robustness (C6–C8) on a held-out test set.")
_set(r, 13, NAVY, bold=True)
footer(s, 13)
notes(s, "[Wrap-up] Where we stand: the architecture, contracts, and label "
         "harmonisation are done. Each module's core is built — physiological "
         "preprocessing, the full video pipeline with training underway, and the "
         "SHAP layer plus backend auth. Remaining: finish training and evaluation "
         "for both models, build the fusion and integration code, and complete the "
         "web app. Evaluation runs in three phases — each module on its own, then "
         "the full pipeline under our 5-condition ablation plus robustness tests, "
         "with Wilcoxon significance testing. ~1 min.")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Differentiation & Conclusion
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(1.6), SW, Inches(0.07), TEAL)
tf = textbox(s, Inches(0.7), Inches(0.5), Inches(12), Inches(1.0))
r = tf.paragraphs[0].add_run(); r.text = "What Makes MedOracle Different"
_set(r, 32, WHITE, bold=True)
bullets(s, [
    "Cross-dataset late fusion — explicitly handles the realistic case of no shared subjects across modalities.",
    "Signal-quality-aware gated fusion with graceful degradation — robust when inputs are poor or missing.",
    "Explainability built in — Kernel SHAP + a faithfulness metric + an LLM chatbot for plain-English reasons.",
], x=Inches(0.7), y=Inches(1.9), w=Inches(12), h=Inches(2.6), size=18, gap=14)
tf = textbox(s, Inches(0.7), Inches(4.9), Inches(12), Inches(1.4))
r = tf.paragraphs[0].add_run()
r.text = ("On track: architecture complete, all three modules implemented and validated, "
          "preliminary video results promising. Remaining work is training, fusion, and the web app.")
_set(r, 16, RGBColor(0xC9, 0xD3, 0xDE), italic=True)
tf2 = textbox(s, Inches(0.7), Inches(6.1), Inches(12), Inches(1.0))
r = tf2.paragraphs[0].add_run(); r.text = "Thank you  —  Questions?"
_set(r, 26, AMBER, bold=True)
notes(s, "To conclude, three things set MedOracle apart: we tackle cross-dataset "
         "fusion honestly, our fusion is quality-aware and degrades gracefully, and "
         "explainability is a first-class feature, not an afterthought. We're on "
         "track — the architecture is done, all three modules are implemented and "
         "validated, and early video results are promising. Remaining work is "
         "finishing training, the fusion code, and the web app. Thank you — we're "
         "happy to take questions.")

out = "MedOracle_Interim_Presentation.pptx"
prs.save(out)
print(f"✓ saved {out} with {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
