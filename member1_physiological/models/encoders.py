from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

D_MODEL    = 128   # embedding dimension (query/key/value size for attention)
T_PRIME    = 32    # output sequence length after downsampling (512 → 32)
DROPOUT    = 0.3   # dropout rate throughout encoders


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int = D_MODEL, max_len: int = 512, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding matrix: shape (max_len, d_model)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)         # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))                        # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Shape (1, max_len, d_model) — batch dimension broadcast
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, D)
        Returns (B, T, D) with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# EEG Encoder
# ---------------------------------------------------------------------------

class EEGEncoder(nn.Module):

    def __init__(
        self,
        n_channels: int = 32,
        d_model:    int = D_MODEL,
        dropout:    float = DROPOUT,
    ):
        super().__init__()

        # --- Depthwise + Pointwise (EEGNet-inspired) ---
        # Depthwise: each of the 32 channels gets its own temporal filter
        # kernel_size=25 ≈ 200 ms at 128 Hz — captures alpha/beta oscillations
        self.depthwise = nn.Conv1d(
            in_channels  = n_channels,
            out_channels = n_channels,
            kernel_size  = 25,
            padding      = 12,      # 'same' padding: (kernel-1)//2
            groups       = n_channels,
            bias         = False,
        )
        # Pointwise: 1×1 conv mixes across 32 channels → 32 features
        self.pointwise = nn.Conv1d(
            in_channels  = n_channels,
            out_channels = 32,
            kernel_size  = 1,
            bias         = False,
        )
        self.bn1     = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        # --- Temporal downsampling block 1 ---
        # stride=4, kernel=8: 512 → 128 time steps
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(64)

        # --- Temporal downsampling block 2 ---
        # stride=4, kernel=4: 128 → 32 time steps
        self.conv3 = nn.Conv1d(64, d_model, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn3   = nn.BatchNorm1d(d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 32, 512)
        Returns (B, T', D_MODEL)
        """
        # Depthwise spatial filtering
        x = self.depthwise(x)                         # (B, 32, 512)
        x = self.pointwise(x)                         # (B, 32, 512)
        x = F.elu(self.bn1(x))                        # ELU common in EEGNet
        x = self.dropout1(x)

        # First downsampling
        x = F.elu(self.bn2(self.conv2(x)))            # (B, 64, 128)

        # Second downsampling
        x = F.elu(self.bn3(self.conv3(x)))            # (B, 128, 32)

        # Transpose to sequence format for attention: (B, T', D)
        x = x.transpose(1, 2)                         # (B, 32, 128) = (B, T', D)

        # Add positional encoding
        x = self.pos_enc(x)

        return x


# ---------------------------------------------------------------------------
# GSR Encoder
# ---------------------------------------------------------------------------

class GSREncoder(nn.Module):

    def __init__(
        self,
        d_model: int = D_MODEL,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # Initial feature extraction from raw GSR
        # kernel_size=25 ≈ 200 ms: captures phasic SCR onset
        self.conv1 = nn.Conv1d(1, 16, kernel_size=25, padding=12, bias=False)
        self.bn1   = nn.BatchNorm1d(16)

        # Temporal downsampling 1: 512 → 128
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(32)

        # Temporal downsampling 2: 128 → 32
        self.conv3 = nn.Conv1d(32, d_model, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn3   = nn.BatchNorm1d(d_model)

        self.dropout  = nn.Dropout(dropout)
        self.pos_enc  = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 512) or (B, 1, 512)
        Returns (B, T', D_MODEL)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)      # (B, 512) → (B, 1, 512)

        x = F.elu(self.bn1(self.conv1(x)))   # (B, 16, 512)
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x)))   # (B, 32, 128)
        x = F.elu(self.bn3(self.conv3(x)))   # (B, 128, 32)

        x = x.transpose(1, 2)               # (B, 32, 128) = (B, T', D)
        x = self.pos_enc(x)
        return x


# ---------------------------------------------------------------------------
# Shape test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B = 4   # batch size

    eeg_enc = EEGEncoder()
    gsr_enc = GSREncoder()

    fake_eeg = torch.randn(B, 32, 512)
    fake_gsr = torch.randn(B, 512)

    out_eeg = eeg_enc(fake_eeg)
    out_gsr = gsr_enc(fake_gsr)

    print(f"EEGEncoder: {fake_eeg.shape} → {out_eeg.shape}  (expected B, {T_PRIME}, {D_MODEL})")
    print(f"GSREncoder: {fake_gsr.shape} → {out_gsr.shape}  (expected B, {T_PRIME}, {D_MODEL})")

    assert out_eeg.shape == (B, T_PRIME, D_MODEL), f"EEGEncoder output shape wrong: {out_eeg.shape}"
    assert out_gsr.shape == (B, T_PRIME, D_MODEL), f"GSREncoder output shape wrong: {out_gsr.shape}"
    print("✓ Encoder shapes correct")

    n_params_eeg = sum(p.numel() for p in eeg_enc.parameters())
    n_params_gsr = sum(p.numel() for p in gsr_enc.parameters())
    print(f"EEGEncoder parameters: {n_params_eeg:,}")
    print(f"GSREncoder parameters: {n_params_gsr:,}")
