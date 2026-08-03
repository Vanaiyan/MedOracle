"""
docs/build_m2_slides.py
========================
Expands the single "M2 - Video + Gated Fusion" slide in
MedOracle_Final_Presentation_updated.pptx into three slides, matching the
existing deck's visual theme exactly (colors, fonts, box styles):

  Slide 8 (existing, retitled/trimmed): M2 - Video Recognition Pipeline
      - keeps the existing 5-box pipeline diagram + 0.6615 stat callout
      - replaces the old bottom two panels with:
          left  = Training Strategy & Key Fixes
          right = Per-Class Test F1 (held-out, dark panel)

  Slide 9 (NEW, inserted): M2 - Gated Confidence Fusion
      - the original equations panel + graceful-degradation panel,
        enlarged to use the full slide body

  Slide 10 (NEW, inserted): M2 - Ablation Study Results
      - real C1-C8 macro-F1 table (styled like the existing Evaluation
        Summary table) + a dark "Key Finding" callout panel

Run:
    python docs/build_m2_slides.py
"""

from __future__ import annotations

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

SRC = "MedOracle_Final_Presentation_updated.pptx"

# ---------------------------------------------------------------------------
# Palette / fonts (extracted from the existing deck)
# ---------------------------------------------------------------------------

BG           = "F2F8FA"
NAVY         = "0D1B2A"
TEAL_DARK    = "1C6E7A"
TEAL         = "028090"
TEAL_BRIGHT  = "02C39A"
CYAN_TEXT    = "C8E6F0"
GRAY_BLUE    = "4A6070"
DARK_ROW     = "1A2A3A"
BODY_TEXT    = "1A1A2E"
WHITE        = "FFFFFF"
ROW_ALT1     = "DFF2EC"
ROW_ALT2     = "E0EEF2"

F_HEAD  = "Cambria"
F_BODY  = "Calibri"


def rgb(hexstr):
    return RGBColor.from_string(hexstr)


def add_rect(slide, x, y, w, h, fill_hex):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill_hex)
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=None):
    """lines: list of (text, size_pt, bold, color_hex, font_name)"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    for i, (text, size, bold, color, font) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.name = font
        r.font.color.rgb = rgb(color)
    return box


def add_header(slide, title, bar_width=1.9):
    """Background + title + accent bar + subtitle, matching the deck's M-section pattern."""
    add_rect(slide, 0, 0, 10.0, 5.62, BG)
    add_text(slide, 0.50, 0.25, 9.0, 0.55, [(title, 34, True, NAVY, F_HEAD)])
    add_rect(slide, 0.50, 0.82, bar_width, 0.06, TEAL)
    add_text(slide, 0.50, 0.90, 9.0, 0.25,
             [("Vanaiyan Kirupagaran — 214215H", 11, False, GRAY_BLUE, F_BODY)])


def new_slide_like(prs, template_slide):
    return prs.slides.add_slide(template_slide.slide_layout)


# ---------------------------------------------------------------------------
# Load + snapshot the template layout/slide before mutating slide 8
# ---------------------------------------------------------------------------

prs = Presentation(SRC)
slide8 = prs.slides[8]

# ---------------------------------------------------------------------------
# STEP 1 — Trim slide 8: remove old bottom panels (shapes 25..50), retitle
# ---------------------------------------------------------------------------

shapes_xml = slide8.shapes._spTree
all_shapes = list(slide8.shapes)
for shp in all_shapes[25:]:
    shapes_xml.remove(shp._element)

# Retitle (shape 1 = title textbox)
title_shape = list(slide8.shapes)[1]
title_shape.text_frame.paragraphs[0].runs[0].text = "M2 — Video Recognition Pipeline"

print(f"Slide 8 trimmed to {len(list(slide8.shapes))} shapes; retitled.")

# ---------------------------------------------------------------------------
# STEP 2 — Slide 8 new bottom-left panel: Training Strategy & Key Fixes
# ---------------------------------------------------------------------------

add_rect(slide8, 0.30, 2.60, 5.50, 2.60, WHITE)
add_text(slide8, 0.50, 2.70, 5.10, 0.32,
         [("Training Strategy & Key Fixes", 14, True, TEAL, F_HEAD)])

training_rows = [
    "Discriminative LR: backbone (layer3+4) @1e-5, head+BiLSTM @5e-4",
    "Fixed overfitting (train 0.88/val 0.63) → healthy 0.62-0.66 generalisation",
    "Frozen BatchNorm + cutout augmentation for robustness",
    "Actor-independent StratifiedGroupKFold — no actor leaks across folds",
    "Full-frame input outperforms YOLO-cropped face (0.70 vs 0.56 macro-F1)",
]
y = 3.08
for row in training_rows:
    add_rect(slide8, 0.50, y + 0.04, 0.10, 0.10, TEAL)
    add_text(slide8, 0.70, y, 4.90, 0.32, [(row, 10.5, False, BODY_TEXT, F_BODY)])
    y += 0.36

# ---------------------------------------------------------------------------
# STEP 3 — Slide 8 new bottom-right panel: Per-Class Test F1 (dark panel)
# ---------------------------------------------------------------------------

add_rect(slide8, 6.00, 2.60, 3.65, 2.60, NAVY)
add_text(slide8, 6.15, 2.70, 3.35, 0.32,
         [("Per-Class Test F1 (788 held-out clips)", 13, True, WHITE, F_HEAD)])

per_class = [
    ("happy", "0.82", "strongest"),
    ("angry", "0.73", ""),
    ("calm", "0.62", ""),
    ("stress", "0.60", ""),
    ("sad", "0.54", "weakest — confused with stress"),
]
y = 3.10
for cls, f1, note in per_class:
    add_rect(slide8, 6.12, y, 3.38, 0.36, DARK_ROW)
    add_text(slide8, 6.20, y + 0.04, 1.30, 0.28,
             [(cls, 11, False, CYAN_TEXT, F_BODY)])
    add_text(slide8, 7.55, y + 0.04, 0.75, 0.28,
             [(f1, 11, True, TEAL_BRIGHT, F_BODY)])
    add_text(slide8, 8.35, y + 0.06, 1.05, 0.24,
             [(note, 7.5, False, GRAY_BLUE if not note else CYAN_TEXT, F_BODY)])
    y += 0.40

print("Slide 8 (Video Recognition Pipeline) complete.")

# ---------------------------------------------------------------------------
# STEP 4 — NEW Slide A: M2 — Gated Confidence Fusion (enlarged)
# ---------------------------------------------------------------------------

slideA = new_slide_like(prs, slide8)
add_header(slideA, "M2 — Gated Confidence Fusion", bar_width=2.3)

# Left panel: equations, enlarged
add_rect(slideA, 0.30, 1.15, 4.55, 4.05, WHITE)
add_text(slideA, 0.50, 1.28, 4.15, 0.36,
         [("Quality-Aware Fusion Formula", 16, True, TEAL, F_HEAD)])

eq_rows = [
    "Confidence:  c_m = 1 − H(P_m) / log(K)   (H = entropy, K = 5 classes)",
    "Guo et al. (2017): entropy-based confidence calibrates better than raw softmax",
    "Quality penalty α:  good = 1.0  ·  degraded = 0.5  ·  poor = 0.1",
    "Gate score:  g_m = c_m × α_m",
    "Weights:  w_m = g_m / Σ g_m   (L1-normalised — preserves extreme gaps,",
    "unlike softmax, which compresses large confidence differences)",
    "Fused prediction:  P_fused = Σ w_m × P_m  →  argmax = final emotion",
]
y = 1.80
for row in eq_rows:
    is_sub = row.startswith("Guo") or row.startswith("unlike")
    if not is_sub:
        add_rect(slideA, 0.50, y + 0.05, 0.10, 0.10, TEAL)
        add_text(slideA, 0.70, y, 4.00, 0.36, [(row, 12, False, BODY_TEXT, F_BODY)])
    else:
        add_text(slideA, 0.90, y, 3.80, 0.32, [(row, 9.5, False, GRAY_BLUE, F_BODY)])
    y += 0.42 if not is_sub else 0.32

# Right panel: graceful degradation, enlarged
add_rect(slideA, 5.15, 1.15, 4.50, 4.05, NAVY)
add_text(slideA, 5.35, 1.28, 4.10, 0.36,
         [("Graceful Degradation", 16, True, WHITE, F_HEAD)])

degrad_rows = [
    ("Both modalities good", "→ Full gated fusion"),
    ("Video poor / missing", "→ Physio only"),
    ("Physio poor / missing", "→ Video only"),
    ("Both poor / missing", "→ Uniform dist. + flag"),
]
y = 1.75
for cond, outcome in degrad_rows:
    add_rect(slideA, 5.30, y, 4.10, 0.55, DARK_ROW)
    add_text(slideA, 5.45, y + 0.11, 2.30, 0.34,
             [(cond, 12, False, CYAN_TEXT, F_BODY)])
    add_text(slideA, 7.55, y + 0.11, 1.75, 0.34,
             [(outcome, 12, True, TEAL_BRIGHT, F_BODY)])
    y += 0.62

add_text(slideA, 5.35, y + 0.10, 4.10, 0.55,
         [("Verified: 18 unit tests confirm this implementation is numerically",
           9.5, False, CYAN_TEXT, F_BODY),
          ("identical (±1e-9) to Member 3's independent reference fusion.",
           9.5, False, CYAN_TEXT, F_BODY)])

print("New Slide A (Gated Confidence Fusion) complete.")

# ---------------------------------------------------------------------------
# STEP 5 — NEW Slide B: M2 — Ablation Study Results
# ---------------------------------------------------------------------------

slideB = new_slide_like(prs, slide8)
add_header(slideB, "M2 — Ablation Study Results", bar_width=2.2)

# Results table (left, wide)
add_rect(slideB, 0.30, 1.05, 5.80, 3.55, WHITE)
add_text(slideB, 0.50, 1.13, 5.40, 0.30,
         [("Macro-F1 by Condition (k=20/emotion, n=50 pairings)", 12.5, True, TEAL, F_HEAD)])

rows = [
    ("Cond.", "Description", "Macro-F1", True, TEAL, WHITE),
    ("C1", "Physio only", "0.443", False, WHITE, BODY_TEXT),
    ("C2", "Video only", "0.631", False, ROW_ALT2, BODY_TEXT),
    ("C3", "Equal weight (0.5/0.5)", "0.692", False, WHITE, BODY_TEXT),
    ("C4", "Confidence only (no quality)", "0.685", False, ROW_ALT2, BODY_TEXT),
    ("C5", "FULL METHOD (conf × quality)", "0.653", False, ROW_ALT1, TEAL_DARK),
    ("C6", "Robust: video poor", "0.652", False, WHITE, BODY_TEXT),
    ("C7", "Robust: video missing", "0.443", False, ROW_ALT2, BODY_TEXT),
    ("C8", "Robust: physio poor", "0.650", False, WHITE, BODY_TEXT),
]
y = 1.50
for cond, desc, f1, is_header, bg, textcolor in rows:
    h = 0.30 if is_header else 0.315
    fill = bg if is_header else bg
    add_rect(slideB, 0.44, y, 5.52, h, fill)
    bold = is_header or cond == "C5"
    fsz = 10.5 if is_header else 10
    add_text(slideB, 0.55, y + 0.03, 0.60, h - 0.05,
             [(cond, fsz, bold, WHITE if is_header else textcolor, F_BODY)])
    add_text(slideB, 1.20, y + 0.03, 3.30, h - 0.05,
             [(desc, fsz, bold, WHITE if is_header else textcolor, F_BODY)])
    add_text(slideB, 4.55, y + 0.03, 1.30, h - 0.05,
             [(f1, fsz, True, WHITE if is_header else textcolor, F_BODY)], align=PP_ALIGN.CENTER)
    y += h + 0.02

add_text(slideB, 0.50, y + 0.06, 5.30, 0.55,
         [("Wilcoxon signed-rank (one-tailed, α=0.05): C5 > C1 p=8.9e-16 ✓  ·  ",
           8.5, False, GRAY_BLUE, F_BODY),
          ("C5 > C2 p=5.5e-10 ✓  ·  C5 vs C3 p=1.0 (not significant)",
           8.5, False, GRAY_BLUE, F_BODY)])

# Right: Key Finding callout (dark panel)
add_rect(slideB, 6.25, 1.05, 3.40, 3.55, NAVY)
add_text(slideB, 6.42, 1.18, 3.05, 0.32,
         [("Key Finding", 15, True, WHITE, F_HEAD)])

finding_rows = [
    ("Fusion (C5) significantly beats BOTH single "
     "modalities (p < 0.001 in both cases).", TEAL_BRIGHT),
    ("Correctly degrades to physio-only (C7 = C1) "
     "when video is unavailable — no crash, no "
     "unpredictable output.", CYAN_TEXT),
    ("Does not yet beat naive equal-weight (C3): "
     "the shipped physio checkpoint has \"good\" "
     "signal quality but weak accuracy, so gating "
     "currently over-trusts it.", CYAN_TEXT),
    ("Diagnosed, not hidden — expected to resolve "
     "once a properly-calibrated physio model "
     "replaces the current checkpoint.", TEAL_BRIGHT),
]
y = 1.62
for text, color in finding_rows:
    add_rect(slideB, 6.42, y + 0.03, 0.09, 0.09, TEAL_BRIGHT)
    add_text(slideB, 6.60, y, 2.90, 0.75, [(text, 10, False, color, F_BODY)])
    y += 0.80

print("New Slide B (Ablation Study Results) complete.")

# ---------------------------------------------------------------------------
# STEP 6 — Reorder: move Slide A, Slide B to positions 9, 10 (right after 8)
# ---------------------------------------------------------------------------

xml_slides = prs.slides._sldIdLst
entries = list(xml_slides)
elmA, elmB = entries[-2], entries[-1]
xml_slides.remove(elmA)
xml_slides.remove(elmB)
xml_slides.insert(9, elmA)
xml_slides.insert(10, elmB)

print(f"Reordered. Total slides now: {len(prs.slides)}")

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

prs.save(SRC)
print(f"Saved: {SRC}")
