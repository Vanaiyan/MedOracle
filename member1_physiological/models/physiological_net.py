from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders  import EEGEncoder, GSREncoder, D_MODEL
from .attention import BidirectionalCrossModalAttention

# Locked emotion classes (NEVER change order)
EMOTION_CLASSES = {
    "stress": 0,
    "calm":   1,
    "happy":  2,
    "sad":    3,
    "angry":  4,
}
N_CLASSES = len(EMOTION_CLASSES)   # 5


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):

    def __init__(
        self,
        in_features: int = 2 * D_MODEL,    # 256
        hidden_dim : int = 128,
        n_classes  : int = N_CLASSES,      # 5
        dropout    : float = 0.4,
    ):
        super().__init__()
        self.fc1     = nn.Linear(in_features, hidden_dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_features) → logits: (B, n_classes)"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Full Network
# ---------------------------------------------------------------------------

class PhysiologicalNet(nn.Module):

    def __init__(
        self,
        d_model  : int = D_MODEL,   # 128
        n_heads  : int = 4,
        n_layers : int = 2,
        dropout  : float = 0.3,
    ):
        super().__init__()

        # Encoders: convert raw signals into sequences of feature vectors
        self.eeg_encoder = EEGEncoder(n_channels=32, d_model=d_model, dropout=dropout)
        self.gsr_encoder = GSREncoder(d_model=d_model, dropout=dropout)

        # Bidirectional cross-modal attention: EEG ↔ GSR information exchange
        self.cross_attention = BidirectionalCrossModalAttention(
            d_model  = d_model,
            n_heads  = n_heads,
            n_layers = n_layers,
            dropout  = dropout,
        )

        # Classification head: fused 2D vector → 5 emotion logits
        self.classifier = ClassificationHead(
            in_features = 2 * d_model,   # 256
            hidden_dim  = 128,
            n_classes   = N_CLASSES,
            dropout     = dropout + 0.1,  # slightly higher dropout in head
        )

        # Weight initialisation (Xavier uniform is good for classification networks)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        eeg: torch.Tensor,    # (B, 32, 512)
        gsr: torch.Tensor,    # (B, 512)
    ) -> torch.Tensor:

        # 1. Encode each modality into feature sequences
        eeg_feat = self.eeg_encoder(eeg)   # (B, T', D)
        gsr_feat = self.gsr_encoder(gsr)   # (B, T', D)

        # 2. Bidirectional cross-modal attention
        fused = self.cross_attention(eeg_feat, gsr_feat)   # (B, 2D)

        # 3. Classify
        logits = self.classifier(fused)   # (B, 5)

        return logits

    def predict_proba(
        self,
        eeg: torch.Tensor,
        gsr: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(eeg, gsr)
            return F.softmax(logits, dim=-1)

    def predict_emotion(
        self,
        eeg: torch.Tensor,
        gsr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        probs   = self.predict_proba(eeg, gsr)
        pred_idx = probs.argmax(dim=-1)
        return pred_idx, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        lines = [
            "PhysiologicalNet — Bidirectional Cross-Modal Attention",
            f"  EEGEncoder:    {sum(p.numel() for p in self.eeg_encoder.parameters()):>10,} params",
            f"  GSREncoder:    {sum(p.numel() for p in self.gsr_encoder.parameters()):>10,} params",
            f"  CrossAttention:{sum(p.numel() for p in self.cross_attention.parameters()):>10,} params",
            f"  Classifier:    {sum(p.numel() for p in self.classifier.parameters()):>10,} params",
            f"  TOTAL:         {self.count_parameters():>10,} params",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B = 4

    net = PhysiologicalNet(d_model=128, n_heads=4, n_layers=2, dropout=0.3)
    net.eval()

    fake_eeg = torch.randn(B, 32, 512)
    fake_gsr = torch.randn(B, 512)

    # Forward pass → logits
    logits = net(fake_eeg, fake_gsr)
    print(f"Input EEG: {fake_eeg.shape}")
    print(f"Input GSR: {fake_gsr.shape}")
    print(f"Logits:    {logits.shape}  expected (B={B}, 5)")

    # Probabilities
    pred_idx, probs = net.predict_emotion(fake_eeg, fake_gsr)
    print(f"Pred idx:  {pred_idx.shape}  values: {pred_idx.tolist()}")
    print(f"Probs:     {probs.shape}  row sums: {probs.sum(dim=1).tolist()}")

    assert logits.shape == (B, 5), f"Logits shape wrong: {logits.shape}"
    assert probs.shape  == (B, 5), f"Probs shape wrong: {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(B), atol=1e-5), "Probs don't sum to 1"

    print("\n" + net.model_summary())
    print("\nPhysiologicalNet shape test passed")

"""
bash to run : python -m member1_physiological.models.physiological_net
"""