"""
Topology-Conditioned Flow Matching
====================================
단일 flow model이 모든 source→target chain shape 변환을 수행.
Target chain의 topology descriptor를 condition으로 받아
source latent → target latent 매핑을 학습.

Flow matching: OT-CFM (Optimal Transport Conditional Flow Matching)
  - Training: MSE on velocity field v(x_t, t, cond)
  - Inference: ODE integration (Euler or midpoint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal embedding for diffusion time t ∈ [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class ChainFlowModel(nn.Module):
    """Topology-conditioned flow matching velocity network.

    Predicts v(x_t, t, cond) where:
      x_t: noisy latent (interpolation between source and target)
      t: flow time ∈ [0, 1]
      cond: target chain descriptor + source latent

    Architecture: Transformer with conditioning via concatenation + cross-attention.

    Input space: continuous latent from VQ-VAE encoder (pre-quantization)
    or directly on chain shapes.
    """

    def __init__(
        self,
        latent_dim=128,        # VQ latent dim (or shape dim for direct flow)
        latent_seq_len=8,      # sequence length of latent (window/4)
        cond_dim=25,           # chain descriptor dim
        hidden_dim=256,
        n_layers=6,
        n_heads=4,
        dropout=0.1,
        use_source_cond=True,  # also condition on source latent
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_seq_len = latent_seq_len
        self.use_source_cond = use_source_cond

        # Time embedding
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition embedding (target chain descriptor)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: x_t (+ optional source concat)
        input_dim = latent_dim * 2 if use_source_cond else latent_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection → velocity in latent space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_t, t, cond, x_src=None):
        """Predict velocity field.

        Args:
            x_t: (batch, seq_len, latent_dim) — noisy interpolation
            t: (batch,) — flow time ∈ [0, 1]
            cond: (batch, cond_dim) — target chain descriptor
            x_src: (batch, seq_len, latent_dim) — source latent (optional)
        Returns:
            v: (batch, seq_len, latent_dim) — predicted velocity
        """
        B, S, D = x_t.shape

        # Prepare input
        if self.use_source_cond and x_src is not None:
            h = torch.cat([x_t, x_src], dim=-1)  # (B, S, 2D)
        else:
            h = x_t

        h = self.input_proj(h)  # (B, S, hidden)

        # Time conditioning: add to all positions
        t_emb = self.time_proj(self.time_emb(t))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)

        # Descriptor conditioning: add to all positions
        c_emb = self.cond_proj(cond)  # (B, hidden)
        h = h + c_emb.unsqueeze(1)

        # Transformer
        for block in self.blocks:
            h = block(h)

        # Output velocity
        v = self.output_proj(h)  # (B, S, latent_dim)
        return v


class DirectChainFlowModel(nn.Module):
    """Direct flow matching on chain shapes (no VQ-VAE).

    For simpler pipeline: operates directly on (W, K*3) chain shape sequences.
    Useful as baseline or when VQ-VAE is not yet trained.

    Input: (batch, window_size, shape_dim) normalized chain shapes
    Output: velocity field of same shape
    """

    def __init__(
        self,
        shape_dim=24,          # K*3
        window_size=32,
        cond_dim=25,           # chain descriptor
        hidden_dim=256,
        n_layers=6,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.shape_dim = shape_dim
        self.window_size = window_size

        # Time & condition embeddings
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input: x_t concat source
        self.input_proj = nn.Linear(shape_dim * 2, hidden_dim)

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, shape_dim),
        )

    def forward(self, x_t, t, cond, x_src):
        """
        Args:
            x_t: (B, W, shape_dim) — noisy target
            t: (B,)
            cond: (B, cond_dim)
            x_src: (B, W, shape_dim) — source shape
        Returns:
            v: (B, W, shape_dim) — velocity
        """
        h = torch.cat([x_t, x_src], dim=-1)
        h = self.input_proj(h)

        h = h + self.pos_emb[:, :h.shape[1]]
        h = h + self.time_proj(self.time_emb(t)).unsqueeze(1)
        h = h + self.cond_proj(cond).unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)


# =====================================================================
# Flow Matching Training & Inference Utils
# =====================================================================

def ot_cfm_sample(x0, x1, t):
    """Optimal Transport CFM: linear interpolation path.

    x_t = (1 - t) * x0 + t * x1
    target velocity = x1 - x0 (constant along path)
    """
    t = t[:, None, None]  # (B, 1, 1)
    x_t = (1 - t) * x0 + t * x1
    v_target = x1 - x0
    return x_t, v_target


def flow_matching_loss(model, x_src, x_tgt, cond):
    """Compute flow matching loss.

    Args:
        model: velocity network v(x_t, t, cond, x_src)
        x_src: (B, S, D) — source
        x_tgt: (B, S, D) — target
        cond: (B, cond_dim) — target chain descriptor
    Returns:
        loss: scalar
    """
    B = x_src.shape[0]
    t = torch.rand(B, device=x_src.device)

    x_t, v_target = ot_cfm_sample(x_src, x_tgt, t)
    v_pred = model(x_t, t, cond, x_src)

    loss = F.mse_loss(v_pred, v_target)
    return loss


@torch.no_grad()
def flow_sample(model, x_src, cond, n_steps=20):
    """Generate target by integrating the learned flow ODE.

    Uses midpoint method for better accuracy.

    Args:
        model: trained velocity network
        x_src: (B, S, D) — source latent/shape
        cond: (B, cond_dim) — target descriptor
        n_steps: integration steps
    Returns:
        x_tgt: (B, S, D) — generated target
    """
    model.eval()
    dt = 1.0 / n_steps
    x_t = x_src.clone()

    for i in range(n_steps):
        t = torch.full((x_src.shape[0],), i * dt, device=x_src.device)

        # Midpoint method
        v1 = model(x_t, t, cond, x_src)
        x_mid = x_t + 0.5 * dt * v1

        t_mid = torch.full((x_src.shape[0],), (i + 0.5) * dt, device=x_src.device)
        v2 = model(x_mid, t_mid, cond, x_src)

        x_t = x_t + dt * v2

    return x_t