<!--
PROMPT FOR CLAUDE (paste this file into your Claude-in-Word chat if you
want to do a single, dedicated "insert all figures now" pass, instead of
handling images chapter-by-chapter as you paste each chapter file)
=========================================================================
This is a master checklist of every figure in the Final Report — 20
figure placements across 19 unique image files (one image is reused twice).
Every one of these figures is ALREADY cited in the correct place in the
corresponding chapter's .md file (with a placeholder line
"[INSERT IMAGE HERE: <path>]" immediately followed by a bold caption
line) — this file exists only as a consolidated cross-reference so you
can (a) verify nothing is missing after pasting all 8 chapters, or
(b) hand it to Claude-in-Word as a single pass to double check every
figure placeholder in the current document has a real image inserted at
that location, replacing the "[INSERT IMAGE HERE: ...]" placeholder text
itself (the placeholder text should not remain visible in the final
document — only the image and its caption should).

All file paths are relative to the project root:
/Users/vanaiyan/Documents/Claude/Projects/FYP/

For each figure: centre the image, and put the bold caption
"Figure X.Y: <caption>" directly below it, exactly as already specified
in that chapter's own instruction block.
=========================================================================
-->

# Master Image Index — MedOracle Final Report

20 figure placements across 8 chapters (19 unique image files — one image, the system architecture diagram, is used twice: as Figure 1.1 and Figure 5.1). "Source" tells you whether the
image is a **real output** (an actual screenshot/plot produced by running
the project's own code) or a **generated diagram** (drawn to illustrate a
design/architecture described in prose — still accurate to the real
system, just not a captured runtime output).

## Chapter 1 — Introduction

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 1.1 | 1.5 | Generated diagram | `docs/report_assets/fig_architecture.png` | High-level architecture of the proposed multimodal emotion recognition system |

## Chapter 2 — Literature Review

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 2.1 | 2.4.1 | Generated diagram | `docs/report_assets/fig2_1_fusion_taxonomy.png` | Fusion taxonomy — early, intermediate and late fusion |

## Chapter 3 — Technology Adapted

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 3.1 | 3.4 | Generated diagram | `docs/report_assets/fig3_1_tech_stack.png` | Technology stack by module ownership |

## Chapter 4 — Our Approach

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 4.1 | 4.2.2 | Generated diagram | `docs/report_assets/fig4_1_m1_pipeline.png` | Member 1 physiological emotion recognition pipeline |
| 4.2 | 4.3.2 | Generated diagram | `docs/report_assets/fig4_2_m2_pipeline.png` | Member 2 video recognition and gated fusion pipeline |
| 4.3 | 4.4.2 | Generated diagram | `docs/report_assets/fig4_3_m3_pipeline.png` | Member 3 explainability and web application architecture |

## Chapter 5 — Analysis and Design

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 5.1 | 5.2 | Generated diagram (reused from 1.1) | `docs/report_assets/fig_architecture.png` | Data flow and interface contracts between Member 1, Member 2 and Member 3 |
| 5.2 | 5.4.1 | Generated diagram | `docs/report_assets/fig5_2_quality_tiers.png` | Signal-quality tiers and their effect on the fusion gate score |
| 5.3 | 5.6 | Generated diagram | `docs/report_assets/fig5_3_degradation_flow.png` | Graceful degradation decision flow |

## Chapter 6 — Implementation

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 6.1 | 6.2.1 | **Real output** (Member 1's preprocessing viz) | `member1_physiological/visualization/viz_deap_loader_output.png` | DEAP dataset structure and per-subject class distribution (preprocessing visualisation) |
| 6.2 | 6.2.2 | **Real output** | `member1_physiological/visualization/viz_signal_quality_output.png` | EEG / GSR / Video signal-quality grading visualisation |
| 6.3 | 6.2.4 | **Real output** (actual SVM baseline run) | `member1_physiological/baseline/Outputs of baselines/SVM output.png` | SVM baseline — subject-independent (LOSO) results |
| 6.4 | 6.2.4 | **Real output** (actual Random Forest run) | `member1_physiological/baseline/Outputs of baselines/Random Forest output.png` | Random Forest baseline — subject-independent (LOSO) results |
| 6.5 | 6.2.4 | **Real output** (actual MLP run) | `member1_physiological/baseline/Outputs of baselines/MLP output.png` | MLP baseline — subject-independent (LOSO) results |
| 6.6 | 6.3.1 | Generated chart (real numbers) | `docs/report_assets/fig6_6_discriminative_lr.png` | Effect of fine-tuning strategy on the train/validation macro-F1 gap |

## Chapter 7 — Discussion & Evaluation

| Figure | Section | Source | File path | Caption |
|---|---|---|---|---|
| 7.1 | 7.2.2 | Generated chart (real numbers, Table 7.1) | `docs/report_assets/fig7_3_deap_distribution.png` | DEAP 5-class distribution used for physiological training |
| 7.2 | 7.2.3 | Generated chart (real numbers) | `docs/report_assets/fig7_4_feature_comparison.png` | Feature representation comparison under LOSO — all converge to ~0.16-0.18 macro-F1 |
| 7.3 | 7.3 | Generated chart (real numbers, Table 7.2) | `docs/report_assets/fig7_5_video_perclass_f1.png` | Video module per-class F1 (held-out test, 788 clips) |
| 7.4 | 7.4.2 | **Real output** (actual ablation study run) | `docs/report_assets/fig_ablation_bar.png` | Ablation study — macro-F1 by condition (C1-C8) |
| 7.5 | 7.4.2 | **Real output** (actual ablation study run) | `data/synced_samples/ablation_confusion_matrices.png` | Ablation study — normalised confusion matrices for physio-only (C1), video-only (C2) and full fusion (C5) |

## Chapter 8 — Conclusion

No figures in this chapter.

---

## Note on filenames vs. figure numbers (Chapter 7 only)

The three new Chapter 7 image *filenames* (`fig7_3_...`, `fig7_4_...`, `fig7_5_...`) do **not** match their final in-report figure *numbers* (7.1, 7.2, 7.3) — this is because they were generated before the chapter was renumbered to keep all figures in reading order (the two pre-existing figures moved from 7.1/7.2 to 7.4/7.5 to make room). This is expected and already handled correctly in `chapter_07_discussion_and_evaluation.md` — just use the table above (indexed by figure number) rather than assuming the filename's number matches the caption's number.

## Quick verification checklist

After inserting all chapters + images in Word, confirm:
- [ ] Every chapter has at least one figure except Chapter 8 (Conclusion) and Chapter 3 has exactly one (3.1)
- [ ] No `[INSERT IMAGE HERE: ...]` placeholder text remains visible anywhere in the document
- [ ] Figure 1.1 and Figure 5.1 are visibly the *same* image (architecture diagram) — this is intentional, not a duplication error
- [ ] Chapter 7's five figures read 7.1 → 7.5 in the order they appear on the page, matching the order in this index
- [ ] The List of Figures (front matter) is updated/regenerated to include all 19 entries once pagination is final
