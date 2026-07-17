"""
member3_explainability/attribution/model_wrapper.py
===================================================
The differentiable model interface the attribution engine attributes over,
plus a small stand-in model so Phase 2 runs before M1/M2 ship.

Feature layout (interpretable, 10-dim)
--------------------------------------
Physiological (M1) : eeg_alpha, eeg_beta, eeg_theta, gsr_scr, gsr_tonic
Video         (M2) : au_smile, au_brow, au_eye, head_pose, face_valence

`ModelWrapper` is the contract: any object exposing
    forward(x: Tensor[B, F]) -> logits Tensor[B, 5]
can be attributed.  Member 2's real ResNet50+BiLSTM head / Member 1's
attention network implement this later; nothing in the engine changes.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

PHYSIO_FEATURES: List[str] = ["eeg_alpha", "eeg_beta", "eeg_theta", "gsr_scr", "gsr_tonic"]
VIDEO_FEATURES: List[str] = ["au_smile", "au_brow", "au_eye", "head_pose", "face_valence"]
ALL_FEATURES: List[str] = PHYSIO_FEATURES + VIDEO_FEATURES

EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]


class DummyEmotionModel(nn.Module):
    """
    Deterministic, differentiable stand-in for the real emotion head, with a
    PLANTED feature->class structure so attributions have real reliance to
    recover (a randomly initialised net has none, making faithfulness noise).

    Planted drivers (mirrors reference_numpy.py so the two demos agree):
        stress <- eeg_beta, gsr_scr      calm  <- eeg_alpha, gsr_tonic
        happy  <- au_smile, face_valence sad   <- au_brow, eeg_theta
        angry  <- au_brow, gsr_scr
    A mild quadratic term makes it nonlinear (so Integrated Gradients is
    non-trivial). The per-feature softmax attention gate is kept as an
    (intentionally unaligned) intrinsic signal to corroborate/compare against.
    """

    def __init__(self, n_features: int = 10, n_classes: int = 5, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.attn = nn.Linear(n_features, n_features)   # attention (unaligned)
        idx = {f: i for i, f in enumerate(ALL_FEATURES)}
        M = torch.zeros(n_classes, n_features)
        planted = {
            0: ["eeg_beta", "gsr_scr"],       # stress
            1: ["eeg_alpha", "gsr_tonic"],    # calm
            2: ["au_smile", "face_valence"],  # happy
            3: ["au_brow", "eeg_theta"],      # sad
            4: ["au_brow", "gsr_scr"],        # angry
        }
        for c, feats in planted.items():
            for f in feats:
                M[c, idx[f]] = 2.0
        M = M + 0.05 * torch.randn(n_classes, n_features)
        self.register_buffer("M", M)          # fixed planted map (not trained)
        self._last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.attn(x), dim=-1)   # per-feature attention weights
        self._last_attn = a.detach()
        g = x * a
        return g @ self.M.t() + 0.5 * (g ** 2) @ self.M.t()

    @torch.no_grad()
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return the intrinsic per-feature attention weights for input x."""
        self.forward(x)
        return self._last_attn
