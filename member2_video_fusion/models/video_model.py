from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when this file is run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from member2_video_fusion.models.resnet_encoder import ResNet50Encoder
from member2_video_fusion.models.bilstm import EmotionBiLSTM
from shared.data_contracts import EMOTION_CLASSES

# Reverse map: int → emotion label string
_IDX_TO_EMOTION = {v: k for k, v in EMOTION_CLASSES.items()}


# ---------------------------------------------------------------------------
# VideoEmotionModel
# ---------------------------------------------------------------------------

class VideoEmotionModel(nn.Module):
    """
    Complete video emotion recognition model.

    Parameters
    ----------
    pretrained   : bool  — load ImageNet weights for ResNet50 (default True)
    encoder_drop : float — dropout on ResNet50 output (default 0.0)
    lstm_drop    : float — dropout between BiLSTM layers (default 0.3)
    fc_drop      : float — dropout in classification head (default 0.4)
    hidden_dim   : int   — BiLSTM hidden size per direction (default 256)
    num_layers   : int   — BiLSTM stacked layers (default 2)
    num_classes  : int   — emotion classes (default 5, locked)

    Forward input  : (B, T, 3, 224, 224)
    Forward output : dict — see forward() docstring
    """

    def __init__(
        self,
        pretrained:   bool  = True,
        encoder_drop: float = 0.0,
        lstm_drop:    float = 0.3,
        fc_drop:      float = 0.4,
        hidden_dim:   int   = 256,
        num_layers:   int   = 2,
        num_classes:  int   = 5,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = ResNet50Encoder(
            pretrained=pretrained,
            dropout=encoder_drop,
        )

        self.bilstm = EmotionBiLSTM(
            input_dim=ResNet50Encoder.FEATURE_DIM,   # 2048
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            lstm_dropout=lstm_drop,
            fc_dropout=fc_drop,
        )

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, T, 3, 224, 224)
            Batch of video clips.
            B = batch size, T = number of frames (16), 224×224 RGB.

        Returns
        -------
        dict:
            "logits"             : (B, 5)       — raw scores for cross-entropy loss
            "probs"              : (B, 5)       — softmax probabilities
            "predicted_class"    : (B,)         — argmax class index
            "frame_features"     : (B, T, 2048) — per-frame ResNet50 features
            "sequence_features"  : (B, 512)     — BiLSTM output before classifier
        """
        B, T, C, H, W = x.shape

        # ── 1. Extract per-frame features ──────────────────────────────────
        # Reshape to treat every frame as an independent image
        x_flat = x.view(B * T, C, H, W)           # (B×T, 3, 224, 224)
        frame_features = self.encoder(x_flat)      # (B×T, 2048)

        # Restore temporal structure
        frame_features = frame_features.view(B, T, -1)   # (B, T, 2048)

        # ── 2. Temporal modelling ───────────────────────────────────────────
        bilstm_out = self.bilstm(frame_features)
        logits     = bilstm_out["logits"]          # (B, 5)
        probs      = bilstm_out["probs"]           # (B, 5)
        seq_feats  = bilstm_out["features"]        # (B, 512)

        predicted_class = torch.argmax(probs, dim=-1)   # (B,)

        return {
            "logits":            logits,
            "probs":             probs,
            "predicted_class":   predicted_class,
            "frame_features":    frame_features,
            "sequence_features": seq_feats,
        }

    # ── Inference helper (single clip, no gradients) ─────────────────────────

    @torch.no_grad()
    def predict_clip(
        self,
        frames_tensor: torch.Tensor,
        device: torch.device | None = None,
    ) -> dict:
        """
        Run inference on a single pre-processed clip tensor.

        Parameters
        ----------
        frames_tensor : torch.Tensor of shape (T, 3, 224, 224)
            Output of dataset.py's frames_to_tensor().
        device : optional torch.device (auto-detected if None)

        Returns
        -------
        dict matching the video portion of the M2 interface contract:
            "predicted_emotion"   : str    — e.g. "stress"
            "confidence"          : float  — entropy-based, in [0, 1]
            "class_probabilities" : dict   — {"stress": p, "calm": p, ...}
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        x = frames_tensor.unsqueeze(0).to(device)   # (1, T, 3, 224, 224)
        out = self.forward(x)

        probs_np = out["probs"][0].cpu().numpy()     # (5,)

        # Build class_probabilities dict
        class_probs = {
            emotion: float(probs_np[idx])
            for emotion, idx in EMOTION_CLASSES.items()
        }

        # Predicted emotion
        pred_idx     = int(out["predicted_class"][0].item())
        pred_emotion = _IDX_TO_EMOTION[pred_idx]

        # Entropy-based confidence: c = 1 - H(p) / log(K)
        import math
        import numpy as np
        K   = self.num_classes
        H   = -float(np.sum(probs_np * np.log(probs_np + 1e-10)))
        conf = max(0.0, 1.0 - H / math.log(K))

        return {
            "predicted_emotion":   pred_emotion,
            "confidence":          conf,
            "class_probabilities": class_probs,
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def param_summary(self) -> dict:
        enc  = self.encoder.param_summary()
        lstm = self.bilstm.param_summary()
        total     = enc["total"]     + lstm["total"]
        trainable = enc["trainable"] + lstm["trainable"]
        return {
            "total":          total,
            "trainable":      total - (enc["frozen"]),
            "frozen":         enc["frozen"],
            "encoder_total":  enc["total"],
            "bilstm_total":   lstm["total"],
            "trainable_pct":  round(trainable / total * 100, 1),
        }

    @staticmethod
    def get_device() -> torch.device:
        """Return the best available device: MPS → CUDA → CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("=== VideoEmotionModel ===\n")

    device = VideoEmotionModel.get_device()
    print(f"Device: {device}")

    model = VideoEmotionModel(pretrained=True)
    model = model.to(device)
    model.eval()

    # ── 1. Parameter summary
    summary = model.param_summary()
    print(f"\nParameters:")
    print(f"  Total     : {summary['total']:,}")
    print(f"  Trainable : {summary['trainable']:,}  ({summary['trainable_pct']}%)")
    print(f"  Frozen    : {summary['frozen']:,}")
    print(f"  Encoder   : {summary['encoder_total']:,}")
    print(f"  BiLSTM    : {summary['bilstm_total']:,}")

    # ── 2. Forward pass — batch of 2 clips, T=16 frames
    B, T = 2, 16
    dummy_clips = torch.zeros(B, T, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(dummy_clips)

    assert out["logits"].shape           == (B, 5),     f"logits: {out['logits'].shape}"
    assert out["probs"].shape            == (B, 5),     f"probs: {out['probs'].shape}"
    assert out["predicted_class"].shape  == (B,),       f"pred: {out['predicted_class'].shape}"
    assert out["frame_features"].shape   == (B, T, 2048)
    assert out["sequence_features"].shape== (B, 512)

    print(f"\nForward pass (B={B}, T={T}):")
    print(f"  logits           : {tuple(out['logits'].shape)} ✓")
    print(f"  probs            : {tuple(out['probs'].shape)}  ✓")
    print(f"  predicted_class  : {tuple(out['predicted_class'].shape)} ✓")
    print(f"  frame_features   : {tuple(out['frame_features'].shape)} ✓")
    print(f"  sequence_features: {tuple(out['sequence_features'].shape)} ✓")

    # ── 3. Probabilities sum to 1
    prob_sums = out["probs"].sum(dim=-1)
    assert all(abs(s.item() - 1.0) < 1e-4 for s in prob_sums)
    print(f"  prob sums        : {[round(s.item(),4) for s in prob_sums]} ✓")

    # ── 4. predict_clip() — single clip
    single_clip = torch.zeros(T, 3, 224, 224)
    result = model.predict_clip(single_clip, device=device)

    assert result["predicted_emotion"] in EMOTION_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["class_probabilities"].keys()) == set(EMOTION_CLASSES.keys())
    prob_total = sum(result["class_probabilities"].values())
    assert abs(prob_total - 1.0) < 1e-4

    print(f"\npredict_clip() output:")
    print(f"  predicted_emotion   : {result['predicted_emotion']}")
    print(f"  confidence          : {result['confidence']:.4f}")
    print(f"  class_probabilities : {result['class_probabilities']}")

    print("\n✓ VideoEmotionModel checks passed")
