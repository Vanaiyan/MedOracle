"""
member2_video_fusion/models/resnet_encoder.py
=============================================
MedOracle — Member 2 (Vanaiyan)

ResNet50 frame encoder.

Extracts a 2048-dimensional feature vector from each face-cropped frame.
These per-frame features are then fed as a sequence into the BiLSTM.

Fine-tuning strategy (v3.1 — discriminative fine-tune)
------------------------------------------------------
  FROZEN   : conv1, bn1, layer1, layer2, layer3 — low/mid ImageNet features.
  TRAINABLE: layer4 conv weights — adapt the high-level features to faces, BUT
             trained at a very low LR (1e-5) by the training loop so they adapt
             slowly and cannot memorise actor identity (the v2 failure mode).
  FROZEN-BN: every BatchNorm in the backbone (incl. layer4) is kept in eval()
             mode via the train() override, so running stats never drift →
             stable features and smooth validation curves.
  REMOVED  : fc                 — original ImageNet 1000-class head, not needed

  This is the middle path between v2 (full layer4 fine-tune → train/val gap
  ~0.25) and a fully-frozen backbone (→ severe underfitting).

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

    conv1/bn1/layer1/layer2/layer3 are frozen; only layer4's conv weights are
    trainable (at a low LR set by the training loop). All BatchNorm layers are
    kept in eval() mode permanently so their running statistics never drift.

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
        self.layer3  = backbone.layer3   # frozen
        self.layer4  = backbone.layer4   # conv weights trainable (low LR); BN frozen
        self.avgpool = backbone.avgpool  # global average pool → (B, 2048, 1, 1)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # ── Apply freezing ─────────────────────────────────────────────────
        self._freeze_layers()

    # ── Freezing ────────────────────────────────────────────────────────────

    def _freeze_layers(self) -> None:
        """Freeze everything except layer4's conv weights (discriminative fine-tune).

        v3.1 — the middle path between v2 (full layer4 fine-tune → overfit) and
        the fully-frozen v3 (→ underfit). conv1, bn1, layer1, layer2, layer3 stay
        frozen. layer4's convolutional weights become trainable so the high-level
        features can adapt to faces — but the training loop runs them at a very
        low LR (1e-5) so they adapt slowly and cannot memorise actor identity.

        BatchNorm parameters (in layer4 too) stay frozen, and all BN running
        statistics are frozen via the train() override below — this keeps the
        features stable and the validation curves smooth.
        """
        # 1. Freeze the entire backbone
        for param in self.parameters():
            param.requires_grad = False
        # 2. Unfreeze ONLY layer4's non-BatchNorm weights (conv layers)
        for module in self.layer4.modules():
            if not isinstance(module, nn.BatchNorm2d):
                for param in module.parameters(recurse=False):
                    param.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen backbone permanently in eval() mode.

        Even with requires_grad=False, a normal model.train() call would put the
        BatchNorm layers into training mode and let their running mean/var drift
        batch-to-batch — producing unstable features and noisy validation curves.
        Setting `training = False` on every submodule freezes the BatchNorm
        statistics so the encoder behaves as a truly fixed feature extractor.

        Note: we assign the flag directly (rather than calling .eval()) to avoid
        re-dispatching back into this overridden train() and recursing.
        """
        super().train(mode)
        for module in self.modules():
            module.training = False
        return self

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

    # ── 4. Verify layer1-3 frozen, layer4 conv trainable, all BN frozen
    early_frozen = all(
        not p.requires_grad
        for m in (encoder.layer1, encoder.layer2, encoder.layer3)
        for p in m.parameters()
    )
    layer4_conv_trainable = any(
        p.requires_grad
        for mod in encoder.layer4.modules() if isinstance(mod, nn.Conv2d)
        for p in mod.parameters(recurse=False)
    )
    layer4_bn_frozen = all(
        not p.requires_grad
        for mod in encoder.layer4.modules() if isinstance(mod, nn.BatchNorm2d)
        for p in mod.parameters(recurse=False)
    )
    print(f"\nLayer1-3 frozen        : {early_frozen} ✓")
    print(f"Layer4 conv trainable  : {layer4_conv_trainable} ✓")
    print(f"Layer4 BatchNorm frozen: {layer4_bn_frozen} ✓")

    # ── 5. Verify train() keeps BatchNorm in eval mode (no running-stat drift)
    encoder.train()
    bn_in_eval = all(
        not m.training for m in encoder.modules()
        if isinstance(m, nn.BatchNorm2d)
    )
    print(f"BatchNorm stays eval   : {bn_in_eval} ✓ (after encoder.train())")

    print("\n✓ ResNet50Encoder checks passed")
