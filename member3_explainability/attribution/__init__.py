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


def __getattr__(name: str):
    """
    Lazy package attributes so `evaluation/reference_numpy.py` (deliberately
    zero heavy deps) can be imported/run without torch installed.

    Importing `member3_explainability.attribution` used to eagerly import
    model_wrapper.py (which does `import torch` at module level), so touching
    the package AT ALL required torch -- even just to reach the numpy-only
    reference implementation. This defers that import until one of the
    torch-dependent names below is actually accessed.
    """
    if name in ("DummyEmotionModel", "PHYSIO_FEATURES", "VIDEO_FEATURES", "ALL_FEATURES"):
        from member3_explainability.attribution import model_wrapper
        return getattr(model_wrapper, name)
    if name == "AttributionEngine":
        from member3_explainability.attribution.attribution_engine import AttributionEngine
        return AttributionEngine
    if name == "FusedEmotionModel":
        from member3_explainability.attribution.fused_ig import FusedEmotionModel
        return FusedEmotionModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttributionEngine",
    "DummyEmotionModel",
    "PHYSIO_FEATURES",
    "VIDEO_FEATURES",
    "ALL_FEATURES",
    "FusedEmotionModel",
]
