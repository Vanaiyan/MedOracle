from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL    = 128   
T_PRIME    = 32   
DROPOUT    = 0.3  

# Sinusoidal Positional Encoding ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int = D_MODEL, max_len: int = 512, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding matrix
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)       
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))                      
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# EEG Encoder---------------------------------------------------------------------------

class EEGEncoder(nn.Module):

    def __init__(
        self,
        n_channels: int = 32,
        d_model:    int = D_MODEL,
        dropout:    float = DROPOUT,
    ):
        super().__init__()

        # Depthwise Convolution 
        self.depthwise = nn.Conv1d(
            in_channels  = n_channels,
            out_channels = n_channels,
            kernel_size  = 25,
            padding      = 12,      
            groups       = n_channels,
            bias         = False,
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv1d(
            in_channels  = n_channels,
            out_channels = 32,
            kernel_size  = 1,
            bias         = False,
        )
        self.bn1     = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        # downsampling
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, d_model, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn3   = nn.BatchNorm1d(d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = self.depthwise(x)                        
        x = self.pointwise(x)                        
        x = F.elu(self.bn1(x))                       
        x = self.dropout1(x)

        # Downsampling
        x = F.elu(self.bn2(self.conv2(x)))  
        x = F.elu(self.bn3(self.conv3(x)))           

        x = x.transpose(1, 2)                       
        x = self.pos_enc(x)

        return x

# GSR Encoder ---------------------------------------------------------------------------

class GSREncoder(nn.Module):

    def __init__(
        self,
        d_model: int = D_MODEL,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=25, padding=12, bias=False)
        self.bn1   = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, d_model, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn3   = nn.BatchNorm1d(d_model)

        self.dropout  = nn.Dropout(dropout)
        self.pos_enc  = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.dim() == 2:
            x = x.unsqueeze(1)      

        x = F.elu(self.bn1(self.conv1(x)))  
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x)))   
        x = F.elu(self.bn3(self.conv3(x)))   

        x = x.transpose(1, 2)               
        x = self.pos_enc(x)
        return x


# Fortest ing purpose ---------------------------------------------------------------------------

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
