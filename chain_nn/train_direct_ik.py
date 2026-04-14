"""
Direct Chain IK 학습
=====================
source chain shape → target joint angles 직접 예측 모델 학습.
Inference에서 Chain LM 없이 forward pass만으로 joint angles 출력.

Usage:
  python chain_nn/train_direct_ik.py --data chain_nn/data/ik_data.pkl --epochs 300
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from chain_nn.direct_model import DirectChainIK, MAX_JOINTS
from common.chain_shape import DESCRIPTOR_DIM, K_POINTS

SHAPE_DIM = K_POINTS * 3


class IKDataset(Dataset):
    def __init__(self, data_path, split='train', val_ratio=0.1, seed=42):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        samples = data['samples']

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))
        n_val = max(1, int(len(samples) * val_ratio))

        if split == 'train':
            indices = indices[n_val:]
        else:
            indices = indices[:n_val]

        self.src_shapes = torch.tensor(np.stack([samples[i]['src_shape'] for i in indices]))
        self.angles = torch.tensor(np.stack([samples[i]['angles'] for i in indices]))
        self.masks = torch.tensor(np.stack([samples[i]['mask'] for i in indices]))
        self.descriptors = torch.tensor(np.stack([samples[i]['descriptor'] for i in indices]))

        print(f"[{split}] {len(self.src_shapes)} samples")

    def __len__(self):
        return len(self.src_shapes)

    def __getitem__(self, idx):
        return (self.src_shapes[idx], self.descriptors[idx],
                self.angles[idx], self.masks[idx])


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = IKDataset(args.data, 'train')
    val_ds = IKDataset(args.data, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = DirectChainIK(
        shape_dim=SHAPE_DIM,
        desc_dim=DESCRIPTOR_DIM,
        max_joints=MAX_JOINTS,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for src, desc, angles_gt, mask in train_loader:
            src, desc = src.to(device), desc.to(device)
            angles_gt, mask = angles_gt.to(device), mask.to(device)

            angles_pred = model(src, desc)

            # Masked MSE: only on valid joints
            loss = ((angles_pred - angles_gt) ** 2 * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        val_angle_errors = []  # actual angle error in degrees
        with torch.no_grad():
            for src, desc, angles_gt, mask in val_loader:
                src, desc = src.to(device), desc.to(device)
                angles_gt, mask = angles_gt.to(device), mask.to(device)

                angles_pred = model(src, desc)
                loss = ((angles_pred - angles_gt) ** 2 * mask).sum() / mask.sum()
                val_losses.append(loss.item())

                # Angle error in degrees (normalized → radians → degrees)
                # angles are in [-1, 1], typical joint range ~2rad → 1 unit ≈ 1rad
                err_rad = ((angles_pred - angles_gt).abs() * mask).sum() / mask.sum()
                val_angle_errors.append(err_rad.item() * 57.3)  # rough estimate

        tr_loss = np.mean(train_losses)
        vr_loss = np.mean(val_losses)
        vr_deg = np.mean(val_angle_errors)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"[{epoch:3d}/{args.epochs}] "
                  f"train={tr_loss:.6f} val={vr_loss:.6f} angle_err~{vr_deg:.1f}deg")

        if vr_loss < best_val:
            best_val = vr_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': vr_loss,
                'config': {
                    'shape_dim': SHAPE_DIM,
                    'desc_dim': DESCRIPTOR_DIM,
                    'max_joints': MAX_JOINTS,
                    'hidden_dim': args.hidden_dim,
                },
            }, ckpt_dir / 'direct_ik_best.pt')

        if epoch % 100 == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch},
                       ckpt_dir / f'direct_ik_ep{epoch}.pt')

    print(f"\nDone. Best val_loss={best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--ckpt_dir", default="chain_nn/checkpoints")
    args = parser.parse_args()
    train(args)