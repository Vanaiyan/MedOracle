"""
member3_explainability/attribution/fused_ig.py
================================================
REAL, end-to-end differentiable Integrated Gradients over the FUSED
prediction -- both modalities combined, not one at a time.

Why this exists
----------------
kernel_shap.py explains the fused decision by treating physio and video as
two "players" and swapping real/background data in and out of the SAME
gated-fusion formula Member 2 uses (coalition-based attribution). This module
explains the exact same fused decision a different way: by chaining Member
1's real PhysiologicalNet, Member 2's real VideoEmotionModel, and a
differentiable re-implementation of the gated-fusion formula into ONE
`nn.Module`, then running captum's IntegratedGradients straight through all
of it with the real raw EEG, GSR, and video tensors as a single multi-input
attribution call. No synthetic/dummy model, no per-modality shortcut -- one
gradient path from raw sensor data to the fused probability.

Both real models are imported read-only from member1_physiological /
member2_video_fusion (never modified here).

Preprocessing parity
---------------------
EEG/GSR z-score and ImageNet frame normalisation are re-implemented here in
torch ops (elementwise affine transforms) so IG can integrate in raw,
human-meaningful units (raw EEG uV, raw GSR ADC, raw 0-255 pixels) while the
underlying computation matches member1_physiological/preprocessing exactly
and member2_video_fusion/preprocessing/multi_corpus_dataset.normalise_and_to_tensor
exactly (verified in evaluation/test scripts -- see fused_ig_parity_check.py).

Fusion formula parity
-----------------------
The gated-fusion math (entropy confidence -> quality alpha -> L1-normalised
weights -> weighted sum) is the SAME formula as member2_video_fusion/fusion.py,
rewritten in torch so autograd can flow through it. Quality tiers (good/
degraded/poor) are passed in as fixed values per explanation call -- they are
discrete real-world grades assessed once from the raw signal (face-detection
rate, amplitude thresholds, etc.), not something IG differentiates through --
same treatment kernel_shap.py already gives them.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the real quality-alpha constants from Member 2's canonical fusion
# module rather than duplicating the dict (single source of truth).
from member2_video_fusion.fusion import QUALITY_ALPHA, physio_quality_of

EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# The combined, differentiable, fused-prediction model
# ---------------------------------------------------------------------------

class FusedEmotionModel(nn.Module):
    """
    forward(eeg_raw, gsr_raw, video_raw) -> fused_probs (B, 5)

    eeg_raw   : (B, 32, 512)         raw EEG, NOT pre-normalised
    gsr_raw   : (B, 512)             raw GSR, NOT pre-normalised
    video_raw : (B, T, 3, 224, 224)  raw RGB pixels in [0, 255], NOT ImageNet-normalised

    physio_model / video_model are the REAL loaded checkpoints (see
    build_fused_model() below) -- this class only adds the differentiable
    preprocessing + fusion glue around them.
    """

    def __init__(
        self,
        physio_model: nn.Module,
        video_model: nn.Module,
        eeg_mean: np.ndarray,
        eeg_std: np.ndarray,
        gsr_mean: float,
        gsr_std: float,
        eeg_quality: str,
        gsr_quality: str,
        video_quality: str,
    ):
        super().__init__()
        self.physio_model = physio_model
        self.video_model = video_model

        self.register_buffer(
            "eeg_mean", torch.as_tensor(eeg_mean, dtype=torch.float32).view(1, 32, 1)
        )
        self.register_buffer(
            "eeg_std", torch.as_tensor(eeg_std, dtype=torch.float32).view(1, 32, 1)
        )
        self.register_buffer("gsr_mean", torch.tensor(float(gsr_mean)))
        self.register_buffer("gsr_std", torch.tensor(float(gsr_std)))
        self.register_buffer(
            "imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1)
        )

        # Quality tiers are FIXED per explanation call (see module docstring).
        physio_quality = physio_quality_of(eeg_quality, gsr_quality)
        self.alpha_physio = QUALITY_ALPHA.get(physio_quality, 0.1)
        self.alpha_video = QUALITY_ALPHA.get(video_quality, 0.1)

    # -- differentiable preprocessing (mirrors member1_physiological/preprocessing) --

    def _preprocess_eeg(self, eeg_raw: torch.Tensor) -> torch.Tensor:
        window_mean = eeg_raw.mean(dim=-1, keepdim=True)          # (B, 32, 1)
        centered = eeg_raw - window_mean
        return (centered - self.eeg_mean) / self.eeg_std

    def _preprocess_gsr(self, gsr_raw: torch.Tensor) -> torch.Tensor:
        window_mean = gsr_raw.mean(dim=-1, keepdim=True)          # (B, 1)
        centered = gsr_raw - window_mean
        return (centered - self.gsr_mean) / self.gsr_std

    def _preprocess_video(self, video_raw: torch.Tensor) -> torch.Tensor:
        frames = video_raw / 255.0
        return (frames - self.imagenet_mean) / self.imagenet_std

    # -- differentiable gated fusion (mirrors member2_video_fusion/fusion.py) --

    @staticmethod
    def _entropy_confidence(probs: torch.Tensor) -> torch.Tensor:
        """c = 1 - H(P)/log(K), computed per batch row. probs: (B, K)."""
        K = probs.shape[-1]
        H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        c = 1.0 - H / math.log(K)
        return c.clamp(0.0, 1.0)

    def forward(
        self,
        eeg_raw: torch.Tensor,
        gsr_raw: torch.Tensor,
        video_raw: torch.Tensor,
    ) -> torch.Tensor:
        eeg_n = self._preprocess_eeg(eeg_raw)
        gsr_n = self._preprocess_gsr(gsr_raw)
        physio_logits = self.physio_model(eeg_n, gsr_n)                 # (B, 5)
        physio_probs = F.softmax(physio_logits, dim=-1)

        video_n = self._preprocess_video(video_raw)
        video_out = self.video_model(video_n)
        video_logits = video_out["logits"] if isinstance(video_out, dict) else video_out
        video_probs = F.softmax(video_logits, dim=-1)

        c_p = self._entropy_confidence(physio_probs)                    # (B,)
        c_v = self._entropy_confidence(video_probs)                     # (B,)
        g_p = c_p * self.alpha_physio
        g_v = c_v * self.alpha_video
        g_total = g_p + g_v + 1e-9
        w_p = (g_p / g_total).unsqueeze(-1)                             # (B, 1)
        w_v = (g_v / g_total).unsqueeze(-1)

        fused_probs = w_p * physio_probs + w_v * video_probs            # (B, 5)
        return fused_probs


# ---------------------------------------------------------------------------
# Builder -- loads the REAL checkpoints (read-only reuse of M1/M2 code)
# ---------------------------------------------------------------------------

def build_fused_model(
    eeg_quality: str,
    gsr_quality: str,
    video_quality: str,
    physio_checkpoint: Optional[str] = None,
) -> FusedEmotionModel:
    """
    Load Member 1's real PhysiologicalNet + Member 2's real VideoEmotionModel
    (both frozen, eval mode) and wrap them in a FusedEmotionModel.

    Reuses member1_physiological.predict.PhysiologicalPredictor purely to
    load the checkpoint + its fitted normalisation stats (no duplicate
    checkpoint-loading logic), and member2_video_fusion.inference's cached
    singleton loader for the video model.
    """
    from pathlib import Path
    from member1_physiological.predict import PhysiologicalPredictor
    from member2_video_fusion.inference import _load_model as _load_video_model

    if physio_checkpoint is None:
        physio_checkpoint = str(
            Path(__file__).parent.parent.parent
            / "member1_physiological" / "checkpoints" / "random_split_best.pt"
        )

    physio_predictor = PhysiologicalPredictor(physio_checkpoint, device="cpu")
    video_model, _device = _load_video_model()

    eeg_mean, eeg_std = physio_predictor.eeg_prep.get_stats()
    gsr_mean, gsr_std = physio_predictor.gsr_prep.get_stats()

    return FusedEmotionModel(
        physio_model=physio_predictor.model,
        video_model=video_model,
        eeg_mean=eeg_mean, eeg_std=eeg_std,
        gsr_mean=gsr_mean, gsr_std=gsr_std,
        eeg_quality=eeg_quality, gsr_quality=gsr_quality, video_quality=video_quality,
    )


# ---------------------------------------------------------------------------
# Explain -- run multi-input Integrated Gradients + aggregate into UI-ready bars
# ---------------------------------------------------------------------------

def explain_fused(
    eeg: np.ndarray,
    gsr: np.ndarray,
    video_frames: np.ndarray,
    eeg_quality: str,
    gsr_quality: str,
    video_quality: str,
    n_steps: int = 128,
) -> Dict:
    """
    Real Integrated Gradients over the FUSED prediction (both modalities,
    one gradient path, one captum call).

    Parameters
    ----------
    eeg          : (32, 512) raw EEG window
    gsr          : (512,)    raw GSR window
    video_frames : (16, 224, 224, 3) raw uint8 RGB frames (NOT pre-normalised)
    eeg_quality, gsr_quality, video_quality : "good"|"degraded"|"poor"
    n_steps      : Riemann-sum steps for IG (kept modest -- each step is a
                   full forward+backward through ResNet50+BiLSTM+PhysioNet)

    Returns
    -------
    dict with:
        target_emotion      : str
        eeg_channel_importance  : {"ch00": float, ..., "ch31": float}
        gsr_importance          : float
        video_frame_importance  : {"frame00": float, ..., "frame15": float}
        modality_totals          : {"physio": float, "video": float}
        completeness_check        : {"sum_attributions": float,
                                      "model_output_delta": float, "gap": float}
    """
    import torch
    from captum.attr import IntegratedGradients

    model = build_fused_model(eeg_quality, gsr_quality, video_quality)
    model.eval()

    eeg_t = torch.from_numpy(eeg.astype(np.float32)).unsqueeze(0)               # (1,32,512)
    gsr_t = torch.from_numpy(gsr.astype(np.float32)).unsqueeze(0)               # (1,512)
    video_t = (
        torch.from_numpy(video_frames.astype(np.float32))
        .permute(0, 3, 1, 2)                                                    # (T,3,224,224)
        .unsqueeze(0)                                                           # (1,T,3,224,224)
    )

    with torch.no_grad():
        fused_probs = model(eeg_t, gsr_t, video_t)
        target = int(fused_probs.argmax(dim=-1).item())
        baseline_probs = model(
            torch.zeros_like(eeg_t), torch.zeros_like(gsr_t), torch.zeros_like(video_t)
        )

    ig = IntegratedGradients(model)
    eeg_attr, gsr_attr, video_attr = ig.attribute(
        inputs=(eeg_t, gsr_t, video_t),
        baselines=(torch.zeros_like(eeg_t), torch.zeros_like(gsr_t), torch.zeros_like(video_t)),
        target=target,
        n_steps=n_steps,
        # Without this, captum stacks ALL n_steps interpolated inputs into one
        # giant batch and pushes it through ResNet50 in a single forward pass --
        # for 16 video frames x n_steps that's easily 1GB+ of activations and
        # reliably OOMs on a CPU-only machine. internal_batch_size tells captum
        # to loop over small chunks of steps instead (slower, bounded memory).
        internal_batch_size=2,
    )

    eeg_channel_importance = {
        f"ch{c:02d}": round(float(eeg_attr[0, c, :].sum().item()), 6) for c in range(32)
    }
    gsr_importance = round(float(gsr_attr[0, :].sum().item()), 6)
    video_frame_importance = {
        f"frame{t:02d}": round(float(video_attr[0, t].sum().item()), 6)
        for t in range(video_attr.shape[1])
    }

    physio_total = sum(eeg_channel_importance.values()) + gsr_importance
    video_total = sum(video_frame_importance.values())

    sum_attr = physio_total + video_total
    model_delta = float((fused_probs[0, target] - baseline_probs[0, target]).item())

    return {
        "target_emotion": EMOTIONS[target],
        "eeg_channel_importance": eeg_channel_importance,
        "gsr_importance": gsr_importance,
        "video_frame_importance": video_frame_importance,
        "modality_totals": {
            "physio": round(physio_total, 6),
            "video": round(video_total, 6),
        },
        "completeness_check": {
            "sum_attributions": round(sum_attr, 6),
            "model_output_delta": round(model_delta, 6),
            "gap": round(abs(sum_attr - model_delta), 6),
        },
    }
