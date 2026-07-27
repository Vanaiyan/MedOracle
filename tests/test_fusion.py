"""
tests/test_fusion.py
====================
Verifies member2_video_fusion/fusion.gated_fusion:
  1. Produces IDENTICAL weights + fused probs to the proven
     member3_explainability/conflict/gate.run_gate (both modalities present).
  2. Handles graceful degradation (missing physio / missing video).
  3. Always returns a contract-valid prediction_output.
"""

from __future__ import annotations

import pytest

from member2_video_fusion.fusion import (
    gated_fusion,
    physio_quality_of,
    entropy_confidence,
    EMOTIONS,
)
from member3_explainability.conflict.gate import run_gate
from shared.data_contracts import make_prediction_output


def _physio(eeg="good", gsr="good"):
    return {
        "predicted_emotion": "stress",
        "confidence": 0.34,
        "class_probabilities": {"stress": 0.60, "calm": 0.10, "happy": 0.10,
                                "sad": 0.10, "angry": 0.10},
        "signal_quality": {"eeg": eeg, "gsr": gsr},
    }


def _video():
    return {
        "predicted_emotion": "happy",
        "confidence": 0.29,
        "class_probabilities": {"stress": 0.10, "calm": 0.10, "happy": 0.60,
                                "sad": 0.10, "angry": 0.10},
    }


# ---------------------------------------------------------------------------
# 1. Numerical equivalence to run_gate (the whole point of a canonical module)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("video_quality", ["good", "degraded", "poor"])
@pytest.mark.parametrize("eeg,gsr", [("good", "good"), ("degraded", "poor"),
                                     ("poor", "good"), ("degraded", "degraded")])
def test_matches_run_gate(video_quality, eeg, gsr):
    physio, video = _physio(eeg, gsr), _video()

    out = gated_fusion(physio, video, video_quality=video_quality)

    pq = physio_quality_of(eeg, gsr)
    trace = run_gate(physio["class_probabilities"], video["class_probabilities"],
                     pq, video_quality)

    assert out["modality_weights"]["physio"] == pytest.approx(trace.w_physio, abs=1e-9)
    assert out["modality_weights"]["video"]  == pytest.approx(trace.w_video,  abs=1e-9)
    for e in EMOTIONS:
        assert out["class_probabilities"][e] == pytest.approx(trace.fused_probs[e], abs=1e-9)
    assert out["predicted_emotion"] == trace.fused_emotion


# ---------------------------------------------------------------------------
# 2. Graceful degradation
# ---------------------------------------------------------------------------

def test_video_only_when_physio_missing():
    out = gated_fusion(None, _video(), video_quality="good")
    assert out["modality_weights"] == {"physio": 0.0, "video": 1.0}
    assert out["predicted_emotion"] == "happy"          # = video argmax
    assert out["signal_quality"]["eeg"] == "poor"
    assert out["per_modality_predictions"]["physio"]["confidence"] == 0.0


def test_physio_only_when_video_missing():
    out = gated_fusion(_physio(), None)
    assert out["modality_weights"] == {"physio": 1.0, "video": 0.0}
    assert out["predicted_emotion"] == "stress"         # = physio argmax
    assert out["signal_quality"]["video"] == "poor"


def test_both_missing_raises():
    with pytest.raises(ValueError):
        gated_fusion(None, None)


def test_both_uniform_gives_uniform_and_half_weights():
    uni = {"predicted_emotion": "stress", "confidence": 0.0,
           "class_probabilities": {e: 0.2 for e in EMOTIONS}}
    uni_p = dict(uni, signal_quality={"eeg": "poor", "gsr": "poor"})
    out = gated_fusion(uni_p, uni, video_quality="poor")
    assert out["modality_weights"]["physio"] == pytest.approx(0.5)
    assert out["modality_weights"]["video"]  == pytest.approx(0.5)
    for e in EMOTIONS:
        assert out["class_probabilities"][e] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 3. Output is always a valid prediction_output
# ---------------------------------------------------------------------------

def test_output_passes_contract_validation():
    out = gated_fusion(_physio("degraded", "good"), _video(), video_quality="degraded")
    # re-running make_prediction_output on the output must not raise
    make_prediction_output(
        predicted_emotion        = out["predicted_emotion"],
        confidence               = out["confidence"],
        class_probabilities      = out["class_probabilities"],
        modality_weights         = out["modality_weights"],
        signal_quality           = out["signal_quality"],
        per_modality_predictions = out["per_modality_predictions"],
    )
    assert abs(sum(out["class_probabilities"].values()) - 1.0) < 1e-6
    assert abs(out["modality_weights"]["physio"] + out["modality_weights"]["video"] - 1.0) < 1e-9


def test_good_physio_beats_poor_video():
    # physio good + confident, video poor → physio should dominate the weight
    out = gated_fusion(_physio("good", "good"), _video(), video_quality="poor")
    assert out["modality_weights"]["physio"] > out["modality_weights"]["video"]
    assert out["predicted_emotion"] == "stress"
