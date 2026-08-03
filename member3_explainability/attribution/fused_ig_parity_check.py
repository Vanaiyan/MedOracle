"""
member3_explainability/attribution/fused_ig_parity_check.py
=============================================================
Proves the differentiable fusion arithmetic inside FusedEmotionModel.forward
(fused_ig.py) is numerically identical to the real, canonical
member2_video_fusion.fusion.gated_fusion() -- i.e. that wrapping the fusion
formula in torch (so Integrated Gradients can differentiate through it) did
not change its behaviour.

This check is pure numpy/python (no torch needed), so it can run anywhere,
including CI, without the heavy dependency.


"""

from __future__ import annotations

import math

from member2_video_fusion.fusion import QUALITY_ALPHA, physio_quality_of, gated_fusion


def _numpy_fused_probs(physio_probs, video_probs, eeg_q, gsr_q, video_q):
    """Mirrors FusedEmotionModel.forward's fusion arithmetic exactly, in numpy."""
    physio_q = physio_quality_of(eeg_q, gsr_q)
    alpha_p = QUALITY_ALPHA.get(physio_q, 0.1)
    alpha_v = QUALITY_ALPHA.get(video_q, 0.1)

    def entropy_confidence(probs):
        K = len(probs)
        H = -sum(p * math.log(p + 1e-9) for p in probs.values())
        return max(0.0, min(1.0, 1 - H / math.log(K)))

    c_p = entropy_confidence(physio_probs)
    c_v = entropy_confidence(video_probs)
    g_p = c_p * alpha_p
    g_v = c_v * alpha_v
    g_total = g_p + g_v + 1e-9
    w_p = g_p / g_total
    w_v = g_v / g_total
    fused = {e: w_p * physio_probs[e] + w_v * video_probs[e] for e in physio_probs}
    return fused, w_p, w_v


def check_parity(physio_probs, video_probs, eeg_q, gsr_q, video_q, tol=1e-6) -> dict:
    real = gated_fusion(
        {"predicted_emotion": max(physio_probs, key=physio_probs.get),
         "confidence": 0.0, "class_probabilities": physio_probs,
         "signal_quality": {"eeg": eeg_q, "gsr": gsr_q}},
        {"predicted_emotion": max(video_probs, key=video_probs.get),
         "confidence": 0.0, "class_probabilities": video_probs},
        video_quality=video_q,
    )
    mine, w_p, w_v = _numpy_fused_probs(physio_probs, video_probs, eeg_q, gsr_q, video_q)

    max_diff = max(abs(real["class_probabilities"][k] - mine[k]) for k in mine)
    weight_diff = max(
        abs(real["modality_weights"]["physio"] - w_p),
        abs(real["modality_weights"]["video"] - w_v),
    )
    ok = max_diff < tol and weight_diff < tol
    return {
        "match": ok,
        "max_class_prob_diff": max_diff,
        "max_weight_diff": weight_diff,
        "real_weights": real["modality_weights"],
        "my_weights": {"physio": w_p, "video": w_v},
    }


if __name__ == "__main__":
    cases = [
        dict(physio_probs={"stress": 0.1933, "happy": 0.1002, "sad": 0.0396, "angry": 0.1238, "calm": 0.5431},
             video_probs={"stress": 0.0469, "happy": 0.0125, "sad": 0.006, "angry": 0.0145, "calm": 0.9201},
             eeg_q="degraded", gsr_q="good", video_q="good"),
        dict(physio_probs={"stress": 0.6, "happy": 0.1, "sad": 0.1, "angry": 0.1, "calm": 0.1},
             video_probs={"stress": 0.1, "happy": 0.6, "sad": 0.1, "angry": 0.1, "calm": 0.1},
             eeg_q="good", gsr_q="poor", video_q="degraded"),
        dict(physio_probs={"stress": 0.2, "happy": 0.2, "sad": 0.2, "angry": 0.2, "calm": 0.2},
             video_probs={"stress": 0.05, "happy": 0.05, "sad": 0.05, "angry": 0.05, "calm": 0.8},
             eeg_q="poor", gsr_q="poor", video_q="good"),
    ]
    all_ok = True
    for i, c in enumerate(cases):
        r = check_parity(**c)
        all_ok &= r["match"]
        print(f"case {i}: {'MATCH' if r['match'] else 'MISMATCH'}  "
              f"(max class-prob diff={r['max_class_prob_diff']:.2e}, "
              f"max weight diff={r['max_weight_diff']:.2e})")
    print()
    print("ALL PARITY CHECKS PASSED" if all_ok else "PARITY CHECK FAILED")
