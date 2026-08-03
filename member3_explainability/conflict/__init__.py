"""
member3_explainability/conflict
===============================
Trust-Aware Modality-Conflict Explanation for Quality-Gated
Multimodal Emotion Recognition.


It sits on top of the existing SHAP layer and the M2 -> M3 prediction_output
contract, and answers three questions the base system cannot:

  1. WHY do the physiological and video modalities disagree?
  2. HOW does the quality gate resolve that disagreement?
  3. WHICH modality should be trusted, and how much?

It also produces counterfactuals over the quality/gate variables
("if EEG signal quality had been good, the decision would flip to stress").


"""

from member3_explainability.conflict.conflict_explainer import (
    detect_conflict,
    explain_conflict,
    ConflictExplainer,
)

__all__ = ["detect_conflict", "explain_conflict", "ConflictExplainer"]
