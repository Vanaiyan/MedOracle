from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model : int = 128,
        n_heads : int = 4,
        ffn_dim : int = 512,
        dropout : float = 0.3,
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,  
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),          
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_features: torch.Tensor,    
        context_features: torch.Tensor,  
    ) -> torch.Tensor:

        # Cross-attention
        attended, _ = self.cross_attention(
            query   = query_features,    
            key     = context_features, 
            value   = context_features, 
        )

        # Residual + LayerNorm (pre-norm style)
        x = self.norm1(query_features + attended)

        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))

        return x   


class BidirectionalCrossModalAttention(nn.Module):

    def __init__(
        self,
        d_model  : int = 128,
        n_heads  : int = 4,
        n_layers : int = 2,
        dropout  : float = 0.3,
    ):
        super().__init__()

        # Stack multiple cross-attention layers for depth
        # In each layer, EEG and GSR exchange information bidirectionally
        self.eeg_to_gsr_layers = nn.ModuleList([
            CrossModalAttentionBlock(d_model=d_model, n_heads=n_heads,
                                     ffn_dim=4*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.gsr_to_eeg_layers = nn.ModuleList([
            CrossModalAttentionBlock(d_model=d_model, n_heads=n_heads,
                                     ffn_dim=4*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output dimension: 2 × d_model (EEG attended + GSR attended)
        self.output_dim = 2 * d_model

    def forward(
        self,
        eeg_features: torch.Tensor, 
        gsr_features: torch.Tensor, 
    ) -> torch.Tensor:

        attended_eeg = eeg_features    
        attended_gsr = gsr_features    

        # Apply N layers of bidirectional cross-attention
        for eeg_to_gsr_layer, gsr_to_eeg_layer in zip(
            self.eeg_to_gsr_layers, self.gsr_to_eeg_layers
        ):
            new_eeg = eeg_to_gsr_layer(
                query_features   = attended_eeg,
                context_features = attended_gsr,
            )
            new_gsr = gsr_to_eeg_layer(
                query_features   = attended_gsr,
                context_features = attended_eeg,
            )
            attended_eeg = new_eeg
            attended_gsr = new_gsr

        # Concatenate 
        fused_sequence = torch.cat([attended_eeg, attended_gsr], dim=-1)

        # Global average pooling over time dimension: (B, 2D)
        # This collapses the T' time steps into a single vector
        # "Average" rather than "max" for smoother gradients
        fused_vector = fused_sequence.mean(dim=1)

        return fused_vector   # (B, 2D)

# Shape verification ---------------------------------------------------------------------------

if __name__ == "__main__":
    from encoders import EEGEncoder, GSREncoder, D_MODEL, T_PRIME

    B = 4

    eeg_enc  = EEGEncoder()
    gsr_enc  = GSREncoder()
    bi_attn  = BidirectionalCrossModalAttention(d_model=D_MODEL, n_heads=4, n_layers=2)

    fake_eeg = torch.randn(B, 32, 512)
    fake_gsr = torch.randn(B, 512)

    eeg_feat = eeg_enc(fake_eeg)   # (B, T', D)
    gsr_feat = gsr_enc(fake_gsr)   # (B, T', D)

    fused = bi_attn(eeg_feat, gsr_feat)   # (B, 2D)

    print(f"EEG features:   {eeg_feat.shape}   expected (B={B}, T'={T_PRIME}, D={D_MODEL})")
    print(f"GSR features:   {gsr_feat.shape}   expected (B={B}, T'={T_PRIME}, D={D_MODEL})")
    print(f"Fused vector:   {fused.shape}      expected (B={B}, 2D={2*D_MODEL})")

    assert fused.shape == (B, 2 * D_MODEL), f"Wrong fused shape: {fused.shape}"
    print("✓ BidirectionalCrossModalAttention shape test passed")

    n_params = sum(p.numel() for p in bi_attn.parameters())
    print(f"Attention module parameters: {n_params:,}")
