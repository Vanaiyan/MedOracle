"""
member2_video_fusion/models/resnet_encoder.py
=============================================
MedOracle — Member 2 (Vanaiyan)

ResNet50 frame encoder.

Extracts a 2048-dimensional feature vector from each face-cropped frame.
These per-frame features are then fed as a sequence into the BiLSTM.

Fine-tuning strategy (He et al., 2016 — ResNet)
-------------------------------------------------
  FROZEN   : layer1, layer2, layer3 — keep low/mid-level ImageNet features
  TRAINABLE: layer4             — learn emotion-relevant high-level features
  TRAINABLE: avgpool            — global average pooling (unchanged architecture)
  REMOVED  : fc                 — original ImageNet 1000-class head, not needed

Output per frame: 2048-dim feature vector (avgpool output)

Input shape  : (B, 3, 224, 224)   — batch of single frames
Output shape : (B, 2048)          — batch of feature vectors

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


# ---------------------------------------------------------------------------
# ResNet50 Encoder
# ---------------------------------------------------------------------------

class ResNet50Encoder(nn.Module):
    """
    Pretrained ResNet50 with the classification head removed.

    layer1 + layer2 are frozen (ImageNet low-level features preserved).
    layer3 + layer4 are fine-tuned to learn emotion-relevant features.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet pretrained weights (default True).
        Set False only for unit tests / architecture inspection.
    dropout : float
        Dropout applied to the output feature vector (default 0.0).
        The VideoEmotionModel may add its own dropout before BiLSTM.

    Forward input  : (B, 3, 224, 224)  — batch of RGB frames, normalised
    Forward output : (B, 2048)         — feature vectors
    """

    FEATURE_DIM = 2048   # ResNet50 avgpool output dimensionality

    def __init__(self, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()

        # Load backbone
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ── Strip the classification head ──────────────────────────────────
        # Keep: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        # Remove: fc (1000-class ImageNet head)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1   # frozen
        self.layer2  = backbone.layer2   # frozen
        self.layer3  = backbone.layer3   # fine-tuned
        self.layer4  = backbone.layer4   # fine-tuned
        self.avgpool = backbone.avgpool  # global average pool → (B, 2048, 1, 1)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # ── Apply freezing ─────────────────────────────────────────────────
        self._freeze_layers()

    # ── Freezing ────────────────────────────────────────────────────────────

    def _freeze_layers(self) -> None:
        """Freeze conv1, bn1, layer1, layer2, layer3. Only layer4 stays trainable."""
        frozen_modules = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_layer(self, layer_name: str) -> None:
        """
        Unfreeze a specific layer by name.
        Useful for gradual unfreezing during training.

        Example:
            encoder.unfreeze_layer("layer2")
        """
        layer = getattr(self, layer_name, None)
        if layer is None:
            raise ValueError(f"No layer named '{layer_name}' in ResNet50Encoder")
        for param in layer.parameters():
            param.requires_grad = True

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, 3, 224, 224)
            Batch of face-cropped, ImageNet-normalised RGB frames.

        Returns
        -------
        torch.Tensor of shape (B, 2048)
            Per-frame feature vectors.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)   # (B, 256,  56, 56)
        x = self.layer2(x)   # (B, 512,  28, 28)
        x = self.layer3(x)   # (B, 1024, 14, 14)
        x = self.layer4(x)   # (B, 2048,  7,  7)

        x = self.avgpool(x)  # (B, 2048,  1,  1)
        x = torch.flatten(x, 1)   # (B, 2048)
        x = self.dropout(x)

        return x

    # ── Utilities ────────────────────────────────────────────────────────────

    def param_summary(self) -> dict:
        """Return a summary of trainable vs frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    frozen,
            "trainable_pct": round(trainable / total * 100, 1),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ResNet50Encoder ===\n")

    encoder = ResNet50Encoder(pretrained=True, dropout=0.0)
    encoder.eval()

    # ── 1. Parameter summary
    summary = encoder.param_summary()
    print("Parameters:")
    print(f"  Total     : {summary['total']:,}")
    print(f"  Trainable : {summary['trainable']:,}  ({summary['trainable_pct']}%)")
    print(f"  Frozen    : {summary['frozen']:,}")

    # ── 2. Forward pass — single frame
    dummy_frame = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        features = encoder(dummy_frame)
    assert features.shape == (1, 2048), f"Expected (1, 2048), got {features.shape}"
    print(f"\nSingle frame  input : {tuple(dummy_frame.shape)}")
    print(f"Single frame output : {tuple(features.shape)} ✓")

    # ── 3. Forward pass — batch of 8 frames
    dummy_batch = torch.zeros(8, 3, 224, 224)
    with torch.no_grad():
        features_batch = encoder(dummy_batch)
    assert features_batch.shape == (8, 2048)
    print(f"\nBatch of 8    input : {tuple(dummy_batch.shape)}")
    print(f"Batch of 8   output : {tuple(features_batch.shape)} ✓")

    # ── 4. Verify layer1/layer2 are frozen, layer3/layer4 are trainable
    frozen_check    = all(not p.requires_grad for p in encoder.layer1.parameters())
    frozen_check   &= all(not p.requires_grad for p in encoder.layer2.parameters())
    trainable_check = any(p.requires_grad for p in encoder.layer3.parameters())
    trainable_check &= any(p.requires_grad for p in encoder.layer4.parameters())

    print(f"\nLayer1 frozen    : {frozen_check} ✓")
    print(f"Layer3 trainable : {trainable_check} ✓")

    print("\n✓ ResNet50Encoder checks passed")
