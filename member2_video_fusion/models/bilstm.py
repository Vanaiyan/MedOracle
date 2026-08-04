from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BiLSTM Emotion Classifier
# ---------------------------------------------------------------------------

class EmotionBiLSTM(nn.Module):
    """
    Bidirectional LSTM that classifies an emotion from a sequence of
    per-frame feature vectors.

    Parameters
    ----------
    input_dim   : int   — dimensionality of each frame feature (default 2048)
    hidden_dim  : int   — hidden units per direction (default 256, × 2 = 512)
    num_layers  : int   — number of stacked LSTM layers (default 2)
    num_classes : int   — number of emotion classes (default 5)
    lstm_dropout: float — dropout between LSTM layers (default 0.3)
    fc_dropout  : float — dropout before final FC layer (default 0.4)

    Forward input  : (B, T, input_dim)   — e.g. (B, 16, 2048)
    Forward output : dict with keys
        "logits" : (B, 5)   — raw scores (use for loss)
        "probs"  : (B, 5)   — softmax probabilities (use for prediction)
    """

    def __init__(
        self,
        input_dim:    int   = 2048,
        hidden_dim:   int   = 256,
        num_layers:   int   = 2,
        num_classes:  int   = 5,
        lstm_dropout: float = 0.3,
        fc_dropout:   float = 0.4,
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # ── BiLSTM ─────────────────────────────────────────────────────────
        # bidirectional=True → effective hidden size = hidden_dim × 2
        # dropout only applies between layers (not after the last layer)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # input: (B, T, input_dim)
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        # ── Classification head ────────────────────────────────────────────
        # Input: hidden_dim × 2 (bidirectional concatenation)
        lstm_out_dim = hidden_dim * 2   # 256 × 2 = 512

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(256, num_classes),
        )

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, T, input_dim)
            Sequence of per-frame feature vectors.
            Typically (B, 16, 2048) from ResNet50Encoder.

        Returns
        -------
        dict:
            "logits"   : (B, num_classes)  — raw scores, used for cross-entropy loss
            "probs"    : (B, num_classes)  — softmax probabilities, used for prediction
            "features" : (B, 512)          — BiLSTM output before classifier (for SHAP)
        """
        # lstm_out : (B, T, hidden_dim × 2)
        lstm_out, _ = self.lstm(x)

        # Mean-pool over all timesteps instead of taking only the last one.
        # For a BiLSTM, lstm_out[:, -1, :] mixes the forward direction's full
        # context with the backward direction's *single-frame* view (it has only
        # seen the last frame), which is a weak representation. Averaging over all
        # T timesteps uses both directions across the whole clip and trains far
        # more smoothly.
        pooled = lstm_out.mean(dim=1)   # (B, 512)

        logits = self.classifier(pooled)   # (B, 5)
        probs  = F.softmax(logits, dim=-1)      # (B, 5)

        return {
            "logits":   logits,
            "probs":    probs,
            "features": pooled,   # pre-classifier, used by SHAP
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def param_summary(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":     total,
            "trainable": trainable,
            "lstm_out_dim": self.hidden_dim * 2,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== EmotionBiLSTM ===\n")

    bilstm = EmotionBiLSTM(
        input_dim=2048, hidden_dim=256, num_layers=2,
        num_classes=5, lstm_dropout=0.3, fc_dropout=0.4,
    )
    bilstm.eval()

    summary = bilstm.param_summary()
    print(f"Parameters:")
    print(f"  Total     : {summary['total']:,}")
    print(f"  Trainable : {summary['trainable']:,}")
    print(f"  LSTM out  : {summary['lstm_out_dim']}-dim")

    # ── 1. Single sample (B=1, T=16, input_dim=2048)
    dummy = torch.zeros(1, 16, 2048)
    with torch.no_grad():
        out = bilstm(dummy)

    assert out["logits"].shape   == (1, 5), f"logits: {out['logits'].shape}"
    assert out["probs"].shape    == (1, 5), f"probs: {out['probs'].shape}"
    assert out["features"].shape == (1, 512)

    prob_sum = out["probs"].sum(dim=-1).item()
    assert abs(prob_sum - 1.0) < 1e-5, f"probs don't sum to 1: {prob_sum}"

    print(f"\nSingle sample  input : (1, 16, 2048)")
    print(f"  logits   : {tuple(out['logits'].shape)} ✓")
    print(f"  probs    : {tuple(out['probs'].shape)}  (sum={prob_sum:.4f}) ✓")
    print(f"  features : {tuple(out['features'].shape)} ✓")

    # ── 2. Batch of 8
    dummy_batch = torch.zeros(8, 16, 2048)
    with torch.no_grad():
        out_batch = bilstm(dummy_batch)

    assert out_batch["logits"].shape == (8, 5)
    assert out_batch["probs"].shape  == (8, 5)
    print(f"\nBatch of 8     input : (8, 16, 2048)")
    print(f"  logits   : {tuple(out_batch['logits'].shape)} ✓")
    print(f"  probs    : {tuple(out_batch['probs'].shape)} ✓")

    # ── 3. Variable sequence length (T=8, e.g. short clip)
    dummy_short = torch.zeros(4, 8, 2048)
    with torch.no_grad():
        out_short = bilstm(dummy_short)
    assert out_short["logits"].shape == (4, 5)
    print(f"\nShort sequence input : (4, 8, 2048)")
    print(f"  logits   : {tuple(out_short['logits'].shape)} ✓")

    print("\n✓ EmotionBiLSTM checks passed")
