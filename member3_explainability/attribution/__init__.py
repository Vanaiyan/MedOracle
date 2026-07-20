"""
member3_explainability/attribution
==================================
Phase 2 — the feature-level ATTRIBUTION ENGINE and the answer to the evaluator's
question "why did you use SHAP, it's used so many times?".

Instead of defaulting to Kernel SHAP, this package:
  * uses **Integrated Gradients** (Sundararajan et al., 2017) as the PRIMARY
    attribution method — axiomatic, faithful, and cheap for differentiable nets;
  * keeps **Kernel SHAP** (Lundberg & Lee, 2017) as a COMPARED BASELINE;
  * reads the model's **intrinsic attention** weights as corroboration;
  * SELECTS the method per measured **faithfulness** (comprehensiveness), so the
    choice of attribution primitive is a data-backed decision, not a default.

The real M1/M2 networks plug in via `ModelWrapper`; a small `DummyEmotionModel`
lets the whole engine run and be validated before those models are ready.

Author : Adshaya Balarajah (214024V)
"""

from member3_explainability.attribution.model_wrapper import (
    DummyEmotionModel,
    PHYSIO_FEATURES,
    VIDEO_FEATURES,
    ALL_FEATURES,
)
from member3_explainability.attribution.attribution_engine import AttributionEngine

__all__ = [
    "AttributionEngine",
    "DummyEmotionModel",
    "PHYSIO_FEATURES",
    "VIDEO_FEATURES",
    "ALL_FEATURES",
]
