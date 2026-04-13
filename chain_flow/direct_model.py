"""
Direct Chain IK Model
======================
source chain positions → target joint angles 직접 예측.
Chain LM은 학습 데이터 생성에만 사용, inference는 forward pass만.

Input:  normalized source chain shape (K*3) + target descriptor (25)
Output: target chain joint angles (n_joints, variable per chain → padded)

Architecture: Transformer with chain descriptor conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


MAX_JOINTS = 12  # 최대 chain joint 수 (padding)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, pos):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=pos.device) * -emb)
        emb = pos.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DirectChainIK(nn.Module):
    """Source chain shape → target joint angles 직접 예측.

    Input:
      src_shape: (batch, K*3) — normalized source chain shape (single frame)
      descriptor: (batch, desc_dim) — target chain topology
      n_joints_mask: (batch, MAX_JOINTS) — valid joint mask
    Output:
      angles: (batch, MAX_JOINTS) — predicted joint angles (masked)
    """

    def __init__(
        self,
        shape_dim=24,       # K_points * 3
        desc_dim=25,        # chain descriptor
        max_joints=MAX_JOINTS,
        hidden_dim=256,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.max_joints = max_joints

        # Shape encoder
        self.shape_enc = nn.Sequential(
            nn.Linear(shape_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Descriptor encoder
        self.desc_enc = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Previous angles encoder (for temporal smoothing)
        self.prev_enc = nn.Sequential(
            nn.Linear(max_joints, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fusion + prediction
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_joints),  # predict angles
            nn.Tanh(),  # output in [-1, 1], scale to joint limits
        )

    def forward(self, src_shape, descriptor, prev_angles=None):
        """
        Args:
            src_shape: (B, shape_dim)
            descriptor: (B, desc_dim)
            prev_angles: (B, max_joints) or None
        Returns:
            angles: (B, max_joints) — in [-1, 1], needs scaling to joint limits
        """
        h_shape = self.shape_enc(src_shape)
        h_desc = self.desc_enc(descriptor)

        if prev_angles is None:
            prev_angles = torch.zeros(src_shape.shape[0], self.max_joints,
                                      device=src_shape.device)
        h_prev = self.prev_enc(prev_angles)

        h = torch.cat([h_shape, h_desc, h_prev], dim=-1)
        angles = self.fusion(h)
        return angles


class TemporalDirectChainIK(nn.Module):
    """Temporal version: window of frames → window of joint angles.

    Input:
      src_shapes: (batch, W, K*3) — W frames of source chain shapes
      descriptor: (batch, desc_dim) — target chain topology
    Output:
      angles: (batch, W, MAX_JOINTS) — predicted joint angles per frame
    """

    def __init__(
        self,
        shape_dim=24,
        desc_dim=25,
        max_joints=MAX_JOINTS,
        hidden_dim=256,
        n_heads=4,
        n_layers=4,
        window_size=32,
        dropout=0.1,
    ):
        super().__init__()
        self.max_joints = max_joints
        self.window_size = window_size

        # Input projection
        self.input_proj = nn.Linear(shape_dim, hidden_dim)

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)

        # Descriptor conditioning
        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_joints),
            nn.Tanh(),
        )

    def forward(self, src_shapes, descriptor):
        """
        Args:
            src_shapes: (B, W, shape_dim)
            descriptor: (B, desc_dim)
        Returns:
            angles: (B, W, max_joints) in [-1, 1]
        """
        B, W, _ = src_shapes.shape
        h = self.input_proj(src_shapes)  # (B, W, hidden)
        h = h + self.pos_emb[:, :W]
        h = h + self.desc_proj(descriptor).unsqueeze(1)  # broadcast desc to all frames

        h = self.transformer(h)
        angles = self.output_proj(h)
        return angles


def scale_angles_to_limits(angles_normalized, joint_lo, joint_hi):
    """[-1, 1] → [lo, hi] 범위로 스케일링.

    Args:
        angles_normalized: (batch, n_joints) in [-1, 1]
        joint_lo: (n_joints,) lower limits
        joint_hi: (n_joints,) upper limits
    Returns:
        angles: (batch, n_joints) in [lo, hi]
    """
    mid = (joint_hi + joint_lo) / 2
    half_range = (joint_hi - joint_lo) / 2
    return angles_normalized * half_range + mid