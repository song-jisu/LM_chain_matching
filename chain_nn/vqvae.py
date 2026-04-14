"""
Unified Chain VQ-VAE
====================
모든 캐릭터의 chain shape을 하나의 codebook으로 압축.
Temporal window (W frames) 단위로 처리하여 시간적 일관성 학습.

Input:  (batch, W, K*3) — W frame window, K sample points × 3D
Output: (batch, W, K*3) — reconstructed chain shapes

Architecture:
  Encoder: 1D Conv (temporal) → latent
  VQ: Vector Quantization with EMA update
  Decoder: 1D ConvTranspose → reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """EMA-updated Vector Quantizer."""

    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1/n_embeddings, 1/n_embeddings)

        # EMA
        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.clone())

    def forward(self, z):
        """
        Args:
            z: (batch, seq_len, embedding_dim) — encoder output
        Returns:
            z_q: (batch, seq_len, embedding_dim) — quantized
            loss: VQ loss
            encoding_indices: (batch, seq_len)
            perplexity: codebook usage metric
        """
        # Flatten to (N, D)
        flat_z = z.reshape(-1, self.embedding_dim)

        # Distances to codebook entries
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * flat_z @ self.embedding.weight.t()
        )

        # Nearest codebook entry
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding(encoding_indices).reshape(z.shape)

        # EMA update (training only)
        if self.training:
            encodings = F.one_hot(encoding_indices, self.n_embeddings).float()
            self._ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay)
            self._ema_w.mul_(self.decay).add_(
                encodings.t() @ flat_z, alpha=1 - self.decay)

            # Laplace smoothing
            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + 1e-5)
                / (n + self.n_embeddings * 1e-5) * n
            )
            self.embedding.weight.data.copy_(
                self._ema_w / cluster_size.unsqueeze(1)
            )

        # Loss
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Perplexity (codebook usage)
        avg_probs = F.one_hot(encoding_indices, self.n_embeddings).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.reshape(z.shape[:-1])
        return z_q, loss, encoding_indices, perplexity


class ResBlock1D(nn.Module):
    """1D Residual block with temporal convolution."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ChainVQVAE(nn.Module):
    """Chain Shape VQ-VAE.

    Input/Output: (batch, window_size, shape_dim)
    where shape_dim = K_points * 3

    Encoder compresses temporally (window → window/4).
    VQ quantizes the latent.
    Decoder reconstructs.
    """

    def __init__(
        self,
        shape_dim=24,          # K_points * 3 (8 * 3 = 24)
        window_size=32,        # temporal window
        hidden_dim=256,        # conv channel dim
        latent_dim=128,        # VQ embedding dim
        n_embeddings=512,      # codebook size
        n_res_blocks=3,        # residual blocks per stage
        commitment_cost=0.25,
    ):
        super().__init__()
        self.shape_dim = shape_dim
        self.window_size = window_size
        self.latent_dim = latent_dim

        # ── Encoder ──
        # (B, W, shape_dim) → permute → (B, shape_dim, W) → conv
        encoder_layers = [
            nn.Conv1d(shape_dim, hidden_dim, 4, stride=2, padding=1),  # W → W/2
            nn.GELU(),
        ]
        for _ in range(n_res_blocks):
            encoder_layers.append(ResBlock1D(hidden_dim))
        encoder_layers += [
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # W/2 → W/4
            nn.GELU(),
        ]
        for _ in range(n_res_blocks):
            encoder_layers.append(ResBlock1D(hidden_dim))
        encoder_layers.append(nn.Conv1d(hidden_dim, latent_dim, 1))  # project to VQ dim
        self.encoder = nn.Sequential(*encoder_layers)

        # ── VQ ──
        self.vq = VectorQuantizer(n_embeddings, latent_dim, commitment_cost)

        # ── Decoder ──
        decoder_layers = [
            nn.Conv1d(latent_dim, hidden_dim, 1),  # from VQ dim
        ]
        for _ in range(n_res_blocks):
            decoder_layers.append(ResBlock1D(hidden_dim))
        decoder_layers += [
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # W/4 → W/2
            nn.GELU(),
        ]
        for _ in range(n_res_blocks):
            decoder_layers.append(ResBlock1D(hidden_dim))
        decoder_layers += [
            nn.ConvTranspose1d(hidden_dim, shape_dim, 4, stride=2, padding=1),  # W/2 → W
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode to VQ tokens.

        Args:
            x: (batch, window_size, shape_dim)
        Returns:
            z_q: (batch, latent_seq, latent_dim) — quantized latent
            vq_loss: VQ loss
            indices: (batch, latent_seq) — codebook indices
            perplexity: float
        """
        # (B, W, D) → (B, D, W)
        h = x.permute(0, 2, 1)
        h = self.encoder(h)
        # (B, latent_dim, W/4) → (B, W/4, latent_dim)
        h = h.permute(0, 2, 1)
        z_q, vq_loss, indices, perplexity = self.vq(h)
        return z_q, vq_loss, indices, perplexity

    def decode(self, z_q):
        """Decode from quantized latent.

        Args:
            z_q: (batch, latent_seq, latent_dim)
        Returns:
            x_recon: (batch, window_size, shape_dim)
        """
        # (B, S, D) → (B, D, S)
        h = z_q.permute(0, 2, 1)
        h = self.decoder(h)
        # (B, shape_dim, W) → (B, W, shape_dim)
        return h.permute(0, 2, 1)

    def forward(self, x):
        """Full forward: encode → VQ → decode.

        Args:
            x: (batch, window_size, shape_dim)
        Returns:
            x_recon, vq_loss, perplexity
        """
        z_q, vq_loss, indices, perplexity = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, perplexity

    def get_tokens(self, x):
        """Encode to discrete tokens only."""
        _, _, indices, _ = self.encode(x)
        return indices

    def decode_tokens(self, indices):
        """Decode from token indices."""
        z_q = self.vq.embedding(indices)
        return self.decode(z_q)