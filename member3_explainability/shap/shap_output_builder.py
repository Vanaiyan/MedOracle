"""
member3_explainability/shap/shap_output_builder.py
===================================================
Assembles the final shap_output dict (M3 → FastAPI contract).

Combines prediction_output + KernelSHAP result + faithfulness score
into a single flat dict ready to be stored in the shap_logs table
and returned from the /predict and /explain endpoints.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

from typing import Dict, Optional

from member3_explainability.shap.kernel_shap import KernelSHAPExplainer, explain as shap_explain
from member3_explainability.shap.faithfulness import compute_faithfulness


def build_shap_output(
    prediction_output: dict,
    background_physio: Optional[Dict[str, float]] = None,
    background_video:  Optional[Dict[str, float]] = None,
) -> dict:
    """
    Run the full SHAP + faithfulness pipeline and return the shap_output dict.

    Parameters
    ----------
    prediction_output : dict
        The M2 → M3 prediction_output (from Member 2's fusion pipeline).
    background_physio : dict, optional
        Training-set mean physio class probabilities (uniform if not provided).
    background_video  : dict, optional
        Training-set mean video class probabilities (uniform if not provided).

    Returns
    -------
    dict — shap_output with the following guaranteed keys:

        predicted_emotion     : str
        confidence            : float
        class_probabilities   : {stress, calm, happy, sad, angry}
        modality_weights      : {physio: float, video: float}
        signal_quality        : {eeg: str, gsr: str, video: str}
        per_modality_predictions : {physio: {...}, video: {...}}
        shap_values           : {EEG: float, GSR: float, video: float}
        feature_importance    : {EEG: float, GSR: float, video: float}
        faithfulness_score    : float  [0, 1]
        signal_reliability    : {eeg: str, gsr: str, video: str}
        coalition_values      : {empty, physio, video, full}
        baseline              : float
        model_output          : float
    """
    # 1. Run Kernel SHAP
    shap_result = shap_explain(
        prediction_output,
        background_physio=background_physio,
        background_video=background_video,
    )

    # 2. Compute faithfulness
    faithfulness_score = compute_faithfulness(
        prediction_output,
        shap_result,
        background_physio=background_physio,
        background_video=background_video,
    )

    # 3. Signal reliability — pass-through from prediction_output signal_quality
    signal_reliability = prediction_output["signal_quality"]

    # 4. Assemble final dict
    shap_output = {
        # Pass-through from prediction_output
        "predicted_emotion":        prediction_output["predicted_emotion"],
        "confidence":               prediction_output["confidence"],
        "class_probabilities":      prediction_output["class_probabilities"],
        "modality_weights":         prediction_output["modality_weights"],
        "signal_quality":           prediction_output["signal_quality"],
        "per_modality_predictions": prediction_output["per_modality_predictions"],

        # SHAP outputs
        "shap_values":              shap_result["shap_values"],
        "feature_importance":       shap_result["feature_importance"],
        "coalition_values":         shap_result["coalition_values"],
        "baseline":                 shap_result["baseline"],
        "model_output":             shap_result["model_output"],

        # Faithfulness
        "faithfulness_score":       faithfulness_score,

        # Signal reliability (same as signal_quality, kept as spec alias)
        "signal_reliability":       signal_reliability,
    }

    return shap_output
