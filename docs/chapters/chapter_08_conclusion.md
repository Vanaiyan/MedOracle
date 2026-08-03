<!--
PROMPT FOR CLAUDE (paste this entire file into your Claude-in-Word chat)
=========================================================================
Please insert the content below as Chapter 8 of my Final Report, following
the University of Moratuwa, Faculty of IT "Guidelines for Preparation of
Final Reports". Formatting rules to apply:

- Font: Times New Roman throughout.
- Body text: 12pt, 1.5 line spacing.
- Chapter heading ("Chapter 8" + title): 18pt, bold, RIGHT-ALIGNED. Put
  "Chapter 8" on its own right-aligned line above the title, and
  right-align the title itself too (both lines 18pt bold, right-aligned).
- Section headings (e.g. "8.2 Future Work"): 12pt, bold.
- Subsection headings (e.g. "8.1.1 Achievement of Objectives"): 12pt, bold.
- Do not add any heading decoration, borders, or colour — plain bold text.
- Bulleted lists: use Word's standard bullet list style, still Times New
  Roman 12pt, 1.5 line spacing.
- There are no figures or tables in this chapter.
- If a Chapter 8 already exists in the document (an earlier, shorter
  draft), REPLACE its content entirely with this enhanced version.
- Do not alter the numbering, content, or formatting of any other chapter.
- This chapter uses no citation markers ([n]).
- IMPORTANT — this chapter is typically the LAST chapter before the
  References list. After inserting it, check that the document's
  References section (alphabetical by first author, [1] Baltrusaitis
  through [13] Zhang) and Appendix A ("Individuals' Contribution to the
  Project") still follow immediately after this chapter, in that order.
  Do not remove or reorder them.

=========================================================================
-->

# Chapter 8

## Conclusion

### 8.1 Conclusion

#### 8.1.1 Achievement of Objectives

This project set out, in Section 1.4.2, five specific objectives. Each is revisited here against the evidence presented in Chapters 6 and 7.

The objective to develop a physiological emotion recognition module using a bidirectional cross-modal attention network trained on DEAP, evaluated under the shared five-class taxonomy, was met: the network was implemented, trained, and evaluated under both a random-split and a full 32-fold Leave-One-Subject-Out protocol. The honest LOSO result (macro-F1 0.179) is weak, but this is reported as a finding about subject-independent EEG/GSR emotion recognition under this taxonomy, arrived at through four controlled diagnostic experiments (Section 7.2), rather than as a failure to build or evaluate the module as intended.

The objective to develop a video-based emotion recognition module using face detection, convolutional feature extraction and BiLSTM temporal modelling, evaluated under an actor-independent protocol, was met and exceeded expectations: the module achieves a held-out macro-F1 of 0.6486 (best checkpoint 0.6615) on clips from actors never seen during training, with a near-zero validation-to-test gap confirming genuine generalisation rather than overfitting.

The objective to design and implement a quality-aware gated fusion mechanism with graceful degradation, verified through automated testing, was met: the mechanism is implemented exactly as designed in Chapter 5, its four degradation rules were each confirmed by the ablation study's robustness conditions (C6-C8, Section 7.4.5), and its correctness was independently verified against a second, separately-implemented reference fusion function to within 1e-9 numerical tolerance across 18 automated tests.

The objective to implement a SHAP-based explainability layer and a FastAPI/React web application supporting real multimodal input, including an explanation of modality disagreement, was met: the /predict/multimodal endpoint accepts real video, EEG and GSR uploads and returns a fused, SHAP-explained prediction, verified through a full end-to-end HTTP test, and the modality-conflict explanation panel surfaces the fusion mechanism's own reasoning to the end user.

The objective to evaluate the complete system through an eight-condition ablation study with statistical significance testing, reporting all findings honestly, was met: the study was carried out over 50 random cross-dataset re-pairings with a formal one-tailed Wilcoxon signed-rank test, and both the favourable result (fusion significantly beats either single modality) and the unfavourable one (fusion does not yet significantly beat naive equal-weight fusion) are reported and diagnosed in Section 7.4, rather than only the favourable result being presented.

#### 8.1.2 Overall Reflection

Taken together, this project set out to build a multimodal emotion recognition system that fuses physiological (EEG/GSR) and facial-video signals through a mechanism that adapts to each modality's real-time reliability, and to make its predictions explainable. That goal has been substantially achieved: a video model that generalises to unseen actors; a canonical, independently-verified gated-fusion mechanism that significantly outperforms either single modality and degrades gracefully when a modality is degraded or missing; a full multimodal pipeline and matching web-application endpoint that accepts real video, EEG and GSR uploads and returns a fused, SHAP-explained prediction; and 82 passing automated tests plus a full end-to-end HTTP verification of the deployed application.

Equally important to this project's conclusions is what was found not to work, and why. The physiological module's subject-independent (LOSO) performance is weak (macro-F1 0.179), and a structured set of diagnostic experiments — comparing feature representations, auditing GSR data quality, and testing coarser label taxonomies — traced this weakness primarily to DEAP's severe class imbalance and the well-documented difficulty of cross-subject EEG generalisation, rather than to an implementation defect. Reporting this limitation candidly, together with the reasoning that located its cause, is presented as a legitimate and, this project argues, more valuable outcome than a favourable but methodologically inflated number would have been. The system's fusion design is precisely the mechanism intended to remain robust in the presence of exactly this kind of single-modality weakness, and the ablation study's robustness conditions (C6-C8) confirm that it does so.

### 8.2 Future Work

#### 8.2.1 Physiological Module Improvements

Future work on the physiological module should prioritise retraining and evaluating the network under a coarser three-class or dimensional (valence/arousal) taxonomy, which preliminary experiments in this project showed lifts LOSO macro-F1 from 0.179 to approximately 0.28 (three-class) or 0.44 (binary valence) without requiring the video model to be retrained, since its five-class output can be remapped by simple probability summation. Recalibrating the physiological signal-quality thresholds (EEG/GSR) against DEAP's actual measured value distributions — analogous to the video-quality recalibration already carried out in this project (Section 5.4.3) — would allow the fusion mechanism's quality-based gating to reflect true predictive reliability rather than only raw signal cleanliness. Domain-adversarial training, which explicitly encourages subject-invariant EEG features, is a further direction worth exploring for the cross-subject generalisation problem specifically.

#### 8.2.2 System-Level Improvements

Once the physiological module is improved, the shipped pipeline's checkpoint should be swapped from the random-split model to a properly calibrated, honestly-evaluated one, and the ablation study re-run to determine whether the full gated method then significantly outperforms naive equal-weight fusion — closing the one unfavourable finding in Section 7.4. The cross-dataset evaluation methodology used throughout this report should also be extended, or replaced, as genuinely paired multimodal data becomes available, reducing this project's reliance on the constructed DEAP/CREMA-D pairing.

#### 8.2.3 Application and Deployment

Beyond the modelling work, a structured user study of the web application's SHAP-based and chatbot-generated explanations with representative end users would evaluate their clinical usefulness rather than only their technical correctness. Before any real-world use, the backend should be deployed against a production PostgreSQL instance rather than the development SQLite database, and a security review of the authentication and file-upload paths should be completed.
