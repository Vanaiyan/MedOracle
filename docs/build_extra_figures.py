"""
docs/build_extra_figures.py
============================
Generates the additional diagrams/charts needed to give every chapter (2-7)
at least one figure, per the university guideline that figures/tables must
be captioned and cited in text. Reuses the same colour palette as
fig_architecture.png / fig_ablation_bar.png for visual consistency.

Run: python docs/build_extra_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "docs/report_assets"

# Palette (matches fig_architecture.png)
C_INPUT   = "#fdf1e7"
C_PROC    = "#e7f0fd"
C_FUSION  = "#fde7ec"
C_OUTPUT  = "#eaf7ee"
C_EDGE    = "#2b4a7a"
C_HILITE  = "#2b7a4b"
C_GREY    = "#b0b8c4"

def box(ax, x, y, w, h, text, fc=C_PROC, ec=C_EDGE, fs=9.5, weight="normal", ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.06",
                        fc=fc, ec=ec, lw=1.3, linestyle=ls)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, weight=weight)

def arrow(ax, x1, y1, x2, y2, color=C_EDGE, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                         lw=1.2, color=color)
    ax.add_patch(a)

def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

# ---------------------------------------------------------------------------
# Figure 2.1 — Fusion taxonomy: early, intermediate, late
# ---------------------------------------------------------------------------
fig, ax = new_ax(11, 4.6)
panels = [
    (0.3, "Early (Feature-Level)", ["Modality A\nfeatures", "Modality B\nfeatures"], "concat", "Classifier", False),
    (4.0, "Intermediate (Attention-Level)", ["Modality A", "Modality B"], "cross-attend", "Shared Decision", False),
    (7.7, "Late (Decision-Level)", ["Modality A\n-> Model A -> P_A", "Modality B\n-> Model B -> P_B"], "combine", "Fusion", True),
]
for x0, title, inputs, mid_label, final_label, hilite in panels:
    ax.text(x0 + 1.35, 4.35, title, ha="center", fontsize=9.5, weight="bold")
    box(ax, x0, 3.2, 1.3, 0.6, inputs[0], fc=C_INPUT, fs=7.5)
    box(ax, x0 + 1.4, 3.2, 1.3, 0.6, inputs[1], fc=C_INPUT, fs=7.5)
    arrow(ax, x0 + 0.65, 3.2, x0 + 1.35, 2.55)
    arrow(ax, x0 + 2.05, 3.2, x0 + 1.35, 2.55)
    fc_mid = C_FUSION if hilite else C_PROC
    box(ax, x0 + 0.55, 1.95, 1.6, 0.55, mid_label, fc=fc_mid, fs=8)
    arrow(ax, x0 + 1.35, 1.95, x0 + 1.35, 1.35)
    box(ax, x0 + 0.35, 0.75, 2.0, 0.55, final_label, fc=C_OUTPUT if not hilite else C_FUSION,
        fs=8, weight="bold" if hilite else "normal")
ax.text(9.05, 0.15, "MedOracle: hybrid design — intermediate fusion within physiology,\nlate fusion across physio + video (this panel)",
        ha="center", fontsize=7.5, style="italic", color=C_HILITE)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_1_fusion_taxonomy.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig2_1_fusion_taxonomy.png")

# ---------------------------------------------------------------------------
# Figure 3.1 — Technology stack by module ownership
# ---------------------------------------------------------------------------
fig, ax = new_ax(10, 5.6)
ax.text(5, 5.3, "Technology Stack by Module Ownership", ha="center", fontsize=12, weight="bold")

cols = [
    (0.3, "Member 1\nPhysiological", ["PyTorch", "scikit-learn", "SciPy (Welch PSD)", "MNE (filtering)"]),
    (3.55, "Member 2\nVideo + Fusion", ["PyTorch + torchvision", "Ultralytics YOLOv8", "OpenCV", "NumPy / pandas"]),
    (6.8, "Member 3\nExplainability + Web", ["SHAP", "FastAPI + SQLAlchemy", "python-jose (JWT)", "React + Recharts"]),
]
for x0, title, items in cols:
    box(ax, x0, 4.55, 2.9, 0.55, title, fc=C_PROC, fs=10, weight="bold")
    y = 3.9
    for it in items:
        box(ax, x0, y, 2.9, 0.55, it, fc="#ffffff", fs=8.5)
        y -= 0.68

ax.plot([0.3, 9.7], [1.15, 1.15], color=C_EDGE, lw=1.2)
box(ax, 1.6, 0.35, 6.8, 0.6, "Shared foundation: Python  ·  Git / GitHub  ·  shared/data_contracts.py",
    fc=C_OUTPUT, fs=9.5, weight="bold")
for x0, *_ in cols:
    arrow(ax, x0 + 1.45, 1.15, 5.0, 0.95)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_1_tech_stack.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig3_1_tech_stack.png")

# ---------------------------------------------------------------------------
# Figure 4.1 — Member 1 physiological pipeline
# ---------------------------------------------------------------------------
fig, ax = new_ax(10, 3.8)
box(ax, 0.2, 2.3, 1.7, 0.8, "EEG\n(32ch, 512)", fc=C_INPUT, fs=8.5)
box(ax, 0.2, 0.6, 1.7, 0.8, "GSR\n(512,)", fc=C_INPUT, fs=8.5)
box(ax, 2.3, 2.3, 1.9, 0.8, "EEG Encoder\n(depthwise+pointwise\nconv)", fc=C_PROC, fs=7.8)
box(ax, 2.3, 0.6, 1.9, 0.8, "GSR Encoder\n(conv stack)", fc=C_PROC, fs=8)
arrow(ax, 1.9, 2.7, 2.3, 2.7)
arrow(ax, 1.9, 1.0, 2.3, 1.0)
box(ax, 4.6, 1.45, 2.2, 1.0, "Bidirectional\nCross-Modal\nAttention", fc=C_FUSION, fs=8.5, weight="bold")
arrow(ax, 4.2, 2.7, 4.6, 2.15)
arrow(ax, 4.2, 1.0, 4.6, 1.75)
box(ax, 7.2, 1.45, 1.9, 1.0, "Classification\nHead\n(softmax, K=5)", fc=C_PROC, fs=8)
arrow(ax, 6.8, 1.95, 7.2, 1.95)
box(ax, 7.2, 0.15, 1.9, 0.9, "physiological_\nprediction_dict", fc=C_OUTPUT, fs=8, weight="bold")
arrow(ax, 8.15, 1.45, 8.15, 1.05)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_1_m1_pipeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig4_1_m1_pipeline.png")

# ---------------------------------------------------------------------------
# Figure 4.2 — Member 2 video + fusion pipeline
# ---------------------------------------------------------------------------
fig, ax = new_ax(11, 4.2)
box(ax, 0.2, 2.7, 1.4, 0.8, "Video", fc=C_INPUT, fs=9)
box(ax, 1.9, 2.7, 1.6, 0.8, "16 Frames\nUniform\nSampling", fc=C_PROC, fs=7.8)
box(ax, 3.8, 2.7, 1.7, 0.8, "ResNet50\n(layer3+4\nfine-tuned)", fc=C_PROC, fs=7.8)
box(ax, 5.8, 2.7, 1.5, 0.8, "BiLSTM\n(2L, 256h)", fc=C_PROC, fs=8)
box(ax, 7.6, 2.7, 1.5, 0.8, "Softmax\n5-class\nP_video", fc=C_PROC, fs=8)
for x1, x2 in [(1.6, 1.9), (3.5, 3.8), (5.5, 5.8), (7.3, 7.6)]:
    arrow(ax, x1, 3.1, x2, 3.1)

box(ax, 0.2, 0.4, 3.0, 0.8, "physiological_prediction_dict\n(from Member 1, Figure 4.1)", fc=C_INPUT, fs=7.5)
box(ax, 4.4, 0.4, 2.4, 0.9, "Gated Fusion\n(confidence x quality,\nL1-normalised)", fc=C_FUSION, fs=8, weight="bold")
arrow(ax, 3.2, 0.8, 4.4, 0.8)
arrow(ax, 8.35, 2.7, 5.9, 1.05)
box(ax, 7.6, 0.4, 2.4, 0.8, "prediction_output\n(fused)", fc=C_OUTPUT, fs=8.5, weight="bold")
arrow(ax, 6.8, 0.8, 7.6, 0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_2_m2_pipeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig4_2_m2_pipeline.png")

# ---------------------------------------------------------------------------
# Figure 4.3 — Member 3 explainability + web app
# ---------------------------------------------------------------------------
fig, ax = new_ax(11, 3.6)
box(ax, 0.2, 2.2, 2.0, 0.9, "prediction_\noutput", fc=C_INPUT, fs=8.5)
box(ax, 2.6, 2.2, 2.0, 0.9, "Kernel SHAP\n+ faithfulness\nscore", fc=C_FUSION, fs=8, weight="bold")
arrow(ax, 2.2, 2.65, 2.6, 2.65)
box(ax, 5.0, 2.2, 2.3, 0.9, "FastAPI backend\n9 endpoints, JWT,\n4-table DB", fc=C_PROC, fs=8)
arrow(ax, 4.6, 2.65, 5.0, 2.65)
box(ax, 7.7, 2.2, 3.0, 0.9, "React Dashboard\nSHAP chart · trend chart ·\nconflict panel · chatbot", fc=C_OUTPUT, fs=7.8, weight="bold")
arrow(ax, 7.3, 2.65, 7.7, 2.65)
ax.text(5.5, 0.9, "Modality-conflict panel explains disagreements between\nphysio and video predictions using the same confidence/quality\nreasoning the gated fusion mechanism (Fig. 4.2) used internally.",
        ha="center", fontsize=8, style="italic", color=C_HILITE)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_3_m3_pipeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig4_3_m3_pipeline.png")

# ---------------------------------------------------------------------------
# Figure 5.2 — Signal quality tiers -> fusion gate effect
# ---------------------------------------------------------------------------
fig, ax = new_ax(9, 3.6)
tiers = [("Good", 1.0, C_OUTPUT), ("Degraded", 0.5, "#fff3d6"), ("Poor", 0.1, "#fde2e2")]
y = 2.9
for name, alpha, color in tiers:
    box(ax, 0.3, y, 1.6, 0.55, name, fc=color, fs=9, weight="bold")
    ax.barh(y + 0.275, alpha * 4.5, height=0.4, left=2.2, color=C_EDGE, alpha=0.75)
    ax.text(2.2 + alpha * 4.5 + 0.15, y + 0.275, f"alpha = {alpha}", va="center", fontsize=8.5)
    y -= 0.75
ax.text(0.3, 0.35, "gate score  g_m = confidence_m x alpha_m   ->   used directly in fusion weight w_m",
        fontsize=8.5, weight="bold", color=C_HILITE)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_2_quality_tiers.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig5_2_quality_tiers.png")

# ---------------------------------------------------------------------------
# Figure 5.3 — Graceful degradation decision flow
# ---------------------------------------------------------------------------
fig, ax = new_ax(10, 4.6)
box(ax, 3.8, 3.8, 2.4, 0.6, "Both modalities available\nand reliable?", fc=C_PROC, fs=8)
box(ax, 6.8, 3.8, 2.5, 0.6, "-> Full gated fusion", fc=C_OUTPUT, fs=8.5, weight="bold")
arrow(ax, 6.2, 4.1, 6.8, 4.1)
ax.text(6.5, 4.35, "yes", fontsize=7.5, color=C_HILITE)

box(ax, 3.8, 2.7, 2.4, 0.6, "Video poor / missing?", fc=C_PROC, fs=8)
arrow(ax, 5.0, 3.8, 5.0, 3.3)
ax.text(5.15, 3.55, "no", fontsize=7.5)
box(ax, 6.8, 2.7, 2.5, 0.6, "-> Physio only", fc=C_OUTPUT, fs=8.5, weight="bold")
arrow(ax, 6.2, 3.0, 6.8, 3.0)
ax.text(6.5, 3.25, "yes", fontsize=7.5, color=C_HILITE)

box(ax, 3.8, 1.6, 2.4, 0.6, "Physio poor / missing?", fc=C_PROC, fs=8)
arrow(ax, 5.0, 2.7, 5.0, 2.2)
ax.text(5.15, 2.45, "no", fontsize=7.5)
box(ax, 6.8, 1.6, 2.5, 0.6, "-> Video only", fc=C_OUTPUT, fs=8.5, weight="bold")
arrow(ax, 6.2, 1.9, 6.8, 1.9)
ax.text(6.5, 2.15, "yes", fontsize=7.5, color=C_HILITE)

box(ax, 3.8, 0.5, 2.4, 0.6, "(both poor / missing)", fc="#fde2e2", fs=8)
arrow(ax, 5.0, 1.6, 5.0, 1.1)
ax.text(5.15, 1.35, "no", fontsize=7.5)
box(ax, 6.8, 0.5, 2.5, 0.6, "-> Uniform distribution\n+ reliability flag", fc="#fde2e2", fs=8, weight="bold")
arrow(ax, 6.2, 0.8, 6.8, 0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_3_degradation_flow.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig5_3_degradation_flow.png")

# ---------------------------------------------------------------------------
# Figure 6.6 — Discriminative LR fix: train/val macro-F1 gap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
configs = ["Frozen backbone\n(underfit)", "Fully unfrozen\n(overfit)"]
train_vals = [0.30, 0.88]
val_vals = [0.30, 0.63]
x = np.arange(len(configs))
w = 0.32
ax.bar(x - w/2, train_vals, width=w, label="Train macro-F1", color=C_GREY, edgecolor="#333")
ax.bar(x + w/2, val_vals, width=w, label="Val macro-F1", color=C_EDGE, edgecolor="#333")
for i, (t, v) in enumerate(zip(train_vals, val_vals)):
    ax.text(i - w/2, t + 0.02, f"{t:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9.5)
ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
ax.set_title("Effect of Fine-Tuning Strategy on Train/Validation Gap", fontsize=11, weight="bold")
ax.text(0.5, -0.24,
        "Fix: discriminative LR (backbone 1e-5, head 5e-4) -> Val (5-fold CV) 0.6433, "
        "Held-out test 0.6486 -> gap approx. 0",
        transform=ax.transAxes, ha="center", fontsize=8.5, style="italic", color=C_HILITE)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_6_discriminative_lr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig6_6_discriminative_lr.png")

# ---------------------------------------------------------------------------
# Figure 7.3 — DEAP 5-class distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.3))
classes = ["stress", "calm", "happy", "sad", "angry"]
counts = [172, 191, 19, 236, 124]
colors = [C_EDGE if c != "happy" else "#c0392b" for c in classes]
bars = ax.bar(classes, counts, color=colors, edgecolor="#333")
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, c + 4, str(c), ha="center", fontsize=9.5)
ax.set_ylabel("Trials")
ax.set_title("DEAP 5-Class Distribution (538 additional trials dropped as unclassifiable)",
              fontsize=10.5, weight="bold")
ax.annotate("2.6% of classifiable trials\n(12.4x imbalance ratio)", xy=(2, 19), xytext=(2.7, 90),
            fontsize=8.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_3_deap_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig7_3_deap_distribution.png")

# ---------------------------------------------------------------------------
# Figure 7.4 — Feature representation comparison under LOSO
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.3))
reps = ["Raw signal\n(neural network)", "Differential Entropy\n+ Logistic Regression", "Differential Entropy\n+ Random Forest"]
vals = [0.179, 0.162, 0.161]
bars = ax.bar(reps, vals, color=[C_EDGE, "#5b8fc7", "#8fb4dc"], edgecolor="#333")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9.5)
ax.axhspan(0.16, 0.18, color=C_HILITE, alpha=0.12)
ax.set_ylabel("Macro-F1 (LOSO)")
ax.set_ylim(0, 0.25)
ax.set_title("Feature Representation Comparison — All Converge to ~0.16-0.18 Macro-F1",
              fontsize=10, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_4_feature_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig7_4_feature_comparison.png")

# ---------------------------------------------------------------------------
# Figure 7.5 — Video per-class F1
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.3))
classes = ["happy", "angry", "calm", "stress", "sad"]
f1s = [0.82, 0.73, 0.62, 0.60, 0.54]
colors = [C_HILITE if c == "happy" else ("#c0392b" if c == "sad" else C_EDGE) for c in classes]
bars = ax.bar(classes, f1s, color=colors, edgecolor="#333")
for b, v in zip(bars, f1s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5)
ax.set_ylabel("Test F1"); ax.set_ylim(0, 1.0)
ax.set_title("Video Module Per-Class F1 (788 held-out clips, 11 unseen actors)",
              fontsize=10.5, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_5_video_perclass_f1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig7_5_video_perclass_f1.png")

print("\nAll 11 new figures generated.")
