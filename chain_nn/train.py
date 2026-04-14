"""
Chain Flow Training
====================
Stage 1: VQ-VAE on all chain shapes (reconstruction)
Stage 2: Flow matching on paired (source, target) latents

Usage:
  # Stage 1: VQ-VAE
  python chain_nn/train.py --stage vqvae --data chain_nn/data/train_data.pkl

  # Stage 2: Flow (requires trained VQ-VAE)
  python chain_nn/train.py --stage flow --data chain_nn/data/train_data.pkl \
    --vqvae_ckpt chain_nn/checkpoints/vqvae_best.pt

  # Direct flow (no VQ-VAE, simpler pipeline)
  python chain_nn/train.py --stage direct_flow --data chain_nn/data/train_data.pkl
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from chain_nn.vqvae import ChainVQVAE
from chain_nn.flow_model import (
    ChainFlowModel, DirectChainFlowModel,
    flow_matching_loss, flow_sample, ot_cfm_sample,
)
from chain_nn.dataset import ChainFlowDataset, SHAPE_DIM, WINDOW_SIZE
from common.chain_shape import DESCRIPTOR_DIM


def train_vqvae(args):
    """Stage 1: Train VQ-VAE on all chain shapes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_ds = ChainFlowDataset(args.data, split='train')
    val_ds = ChainFlowDataset(args.data, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = ChainVQVAE(
        shape_dim=SHAPE_DIM,
        window_size=WINDOW_SIZE,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_embeddings=args.codebook_size,
        n_res_blocks=3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_losses = {'recon': [], 'vq': [], 'total': [], 'perp': []}

        for src, tgt, desc in train_loader:
            # VQ-VAE trains on ALL shapes (src + tgt concatenated)
            shapes = torch.cat([src, tgt], dim=0).to(device)  # (2B, W, D)

            x_recon, vq_loss, perplexity = model(shapes)
            recon_loss = F.mse_loss(x_recon, shapes)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses['recon'].append(recon_loss.item())
            train_losses['vq'].append(vq_loss.item())
            train_losses['total'].append(loss.item())
            train_losses['perp'].append(perplexity.item())

        scheduler.step()

        # ── Val ──
        model.eval()
        val_losses = {'recon': [], 'vq': [], 'perp': []}
        with torch.no_grad():
            for src, tgt, desc in val_loader:
                shapes = torch.cat([src, tgt], dim=0).to(device)
                x_recon, vq_loss, perplexity = model(shapes)
                recon_loss = F.mse_loss(x_recon, shapes)
                val_losses['recon'].append(recon_loss.item())
                val_losses['vq'].append(vq_loss.item())
                val_losses['perp'].append(perplexity.item())

        tr = {k: np.mean(v) for k, v in train_losses.items()}
        vr = {k: np.mean(v) for k, v in val_losses.items()}

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train: recon={tr['recon']:.6f} vq={tr['vq']:.6f} perp={tr['perp']:.1f} | "
              f"val: recon={vr['recon']:.6f} perp={vr['perp']:.1f}")

        # Save best
        val_total = vr['recon'] + vr['vq']
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_total,
                'config': {
                    'shape_dim': SHAPE_DIM,
                    'window_size': WINDOW_SIZE,
                    'hidden_dim': args.hidden_dim,
                    'latent_dim': args.latent_dim,
                    'codebook_size': args.codebook_size,
                },
            }, ckpt_dir / 'vqvae_best.pt')
            print(f"  -> Saved best (val_loss={val_total:.6f})")

        # Periodic save
        if epoch % 50 == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
            }, ckpt_dir / f'vqvae_ep{epoch}.pt')

    print(f"\nVQ-VAE training done. Best val_loss={best_val_loss:.6f}")


def train_flow(args, use_vqvae=True):
    """Stage 2: Train flow matching on paired latents.

    If use_vqvae=True: flow operates in VQ-VAE latent space.
    If use_vqvae=False: direct flow on chain shapes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_ds = ChainFlowDataset(args.data, split='train')
    val_ds = ChainFlowDataset(args.data, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # VQ-VAE (frozen, for encoding)
    vqvae = None
    if use_vqvae:
        ckpt = torch.load(args.vqvae_ckpt, map_location=device, weights_only=False)
        cfg = ckpt['config']
        vqvae = ChainVQVAE(
            shape_dim=cfg['shape_dim'],
            window_size=cfg['window_size'],
            hidden_dim=cfg['hidden_dim'],
            latent_dim=cfg['latent_dim'],
            n_embeddings=cfg['codebook_size'],
        ).to(device)
        vqvae.load_state_dict(ckpt['model'])
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
        print(f"Loaded VQ-VAE from {args.vqvae_ckpt} (epoch {ckpt['epoch']})")

        latent_dim = cfg['latent_dim']
        latent_seq = WINDOW_SIZE // 4  # encoder downsamples 4x
    else:
        latent_dim = SHAPE_DIM
        latent_seq = WINDOW_SIZE

    # Flow model
    if use_vqvae:
        flow = ChainFlowModel(
            latent_dim=latent_dim,
            latent_seq_len=latent_seq,
            cond_dim=DESCRIPTOR_DIM,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(device)
    else:
        flow = DirectChainFlowModel(
            shape_dim=SHAPE_DIM,
            window_size=WINDOW_SIZE,
            cond_dim=DESCRIPTOR_DIM,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(device)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Flow params: {sum(p.numel() for p in flow.parameters()):,}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        flow.train()
        train_losses = []

        for src, tgt, desc in train_loader:
            src, tgt, desc = src.to(device), tgt.to(device), desc.to(device)

            # Encode to latent (if using VQ-VAE)
            if vqvae is not None:
                with torch.no_grad():
                    # Pre-quantization latent (continuous)
                    src_h = src.permute(0, 2, 1)
                    src_h = vqvae.encoder(src_h).permute(0, 2, 1)
                    tgt_h = tgt.permute(0, 2, 1)
                    tgt_h = vqvae.encoder(tgt_h).permute(0, 2, 1)
                x_src, x_tgt = src_h, tgt_h
            else:
                x_src, x_tgt = src, tgt

            loss = flow_matching_loss(flow, x_src, x_tgt, desc)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Val ──
        flow.eval()
        val_losses = []
        val_shape_errors = []

        with torch.no_grad():
            for src, tgt, desc in val_loader:
                src, tgt, desc = src.to(device), tgt.to(device), desc.to(device)

                if vqvae is not None:
                    src_h = src.permute(0, 2, 1)
                    src_h = vqvae.encoder(src_h).permute(0, 2, 1)
                    tgt_h = tgt.permute(0, 2, 1)
                    tgt_h = vqvae.encoder(tgt_h).permute(0, 2, 1)
                    x_src, x_tgt = src_h, tgt_h
                else:
                    x_src, x_tgt = src, tgt

                loss = flow_matching_loss(flow, x_src, x_tgt, desc)
                val_losses.append(loss.item())

                # Shape error: generate and compare
                if len(val_shape_errors) < 10:  # only check a few batches
                    x_gen = flow_sample(flow, x_src, desc, n_steps=10)
                    if vqvae is not None:
                        # Decode generated latent
                        z_q_gen, _, _, _ = vqvae.vq(x_gen)
                        tgt_recon = vqvae.decode(z_q_gen)
                        z_q_real, _, _, _ = vqvae.vq(x_tgt)
                        tgt_real = vqvae.decode(z_q_real)
                        err = F.mse_loss(tgt_recon, tgt_real).item()
                    else:
                        err = F.mse_loss(x_gen, x_tgt).item()
                    val_shape_errors.append(err)

        tr_loss = np.mean(train_losses)
        vr_loss = np.mean(val_losses)
        vr_shape = np.mean(val_shape_errors) if val_shape_errors else 0

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train_loss={tr_loss:.6f} | val_loss={vr_loss:.6f} shape_err={vr_shape:.6f}")

        if vr_loss < best_val_loss:
            best_val_loss = vr_loss
            tag = 'flow' if use_vqvae else 'direct_flow'
            torch.save({
                'model': flow.state_dict(),
                'epoch': epoch,
                'val_loss': vr_loss,
                'use_vqvae': use_vqvae,
                'config': {
                    'latent_dim': latent_dim,
                    'latent_seq_len': latent_seq,
                    'cond_dim': DESCRIPTOR_DIM,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                    'shape_dim': SHAPE_DIM,
                    'window_size': WINDOW_SIZE,
                },
            }, ckpt_dir / f'{tag}_best.pt')
            print(f"  -> Saved best (val_loss={vr_loss:.6f})")

        if epoch % 50 == 0:
            tag = 'flow' if use_vqvae else 'direct_flow'
            torch.save({
                'model': flow.state_dict(),
                'epoch': epoch,
            }, ckpt_dir / f'{tag}_ep{epoch}.pt')

    print(f"\nFlow training done. Best val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chain flow models")
    parser.add_argument("--stage", choices=["vqvae", "flow", "direct_flow"], required=True)
    parser.add_argument("--data", required=True, help="Training data .pkl")
    parser.add_argument("--vqvae_ckpt", default=None, help="VQ-VAE checkpoint (for flow stage)")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--ckpt_dir", default="chain_nn/checkpoints")

    args = parser.parse_args()

    if args.stage == "vqvae":
        train_vqvae(args)
    elif args.stage == "flow":
        if args.vqvae_ckpt is None:
            print("ERROR: --vqvae_ckpt required for flow stage")
            sys.exit(1)
        train_flow(args, use_vqvae=True)
    elif args.stage == "direct_flow":
        train_flow(args, use_vqvae=False)