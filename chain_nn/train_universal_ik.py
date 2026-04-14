"""
Universal Chain IK 학습
========================
합성 데이터로 학습하여 임의의 chain 구조에 대해
source shape → joint angles를 직접 예측하는 모델.

Usage:
  # 합성 데이터 생성
  python chain_nn/synthetic_data.py --n_samples 200000 --n_configs 5000

  # 학습
  python chain_nn/train_universal_ik.py --data chain_nn/data/synthetic_ik_data.pkl --epochs 300
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
from chain_nn.direct_model import MAX_JOINTS
from chain_nn.synthetic_data import DESC_DIM, SHAPE_DIM


class UniversalChainIK(nn.Module):
    """Universal chain IK: 임의의 chain 구조에서 shape → angles.

    입력에 chain descriptor (길이비, 축, 한계)를 포함하여
    어떤 chain 구조에도 일반화.
    """

    def __init__(self, shape_dim=SHAPE_DIM, desc_dim=DESC_DIM,
                 max_joints=MAX_JOINTS, hidden_dim=512, n_layers=6):
        super().__init__()
        self.max_joints = max_joints

        # Shape encoder
        self.shape_enc = nn.Sequential(
            nn.Linear(shape_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Descriptor encoder (길이비 + 축 + 한계)
        self.desc_enc = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Previous angles encoder
        self.prev_enc = nn.Sequential(
            nn.Linear(max_joints, hidden_dim),
            nn.GELU(),
        )

        # Deep fusion network
        layers = []
        layers.append(nn.Linear(hidden_dim * 3, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, max_joints))
        layers.append(nn.Tanh())
        self.fusion = nn.Sequential(*layers)

    def forward(self, src_shape, descriptor, prev_angles=None):
        """
        Args:
            src_shape: (B, shape_dim) — normalized chain shape
            descriptor: (B, desc_dim) — chain structure (ratios, axes, limits)
            prev_angles: (B, max_joints) — previous frame angles (optional)
        Returns:
            angles: (B, max_joints) — in [-1, 1]
        """
        h_shape = self.shape_enc(src_shape)
        h_desc = self.desc_enc(descriptor)

        if prev_angles is None:
            prev_angles = torch.zeros(src_shape.shape[0], self.max_joints,
                                      device=src_shape.device)
        h_prev = self.prev_enc(prev_angles)

        h = torch.cat([h_shape, h_desc, h_prev], dim=-1)
        return self.fusion(h)


class SyntheticIKDataset(Dataset):
    def __init__(self, data_path, split='train', val_ratio=0.1, seed=42):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        samples = data['samples']
        self.data_config = data['config']

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))
        n_val = max(1, int(len(samples) * val_ratio))

        if split == 'train':
            indices = indices[n_val:]
        else:
            indices = indices[:n_val]

        self.src_shapes = torch.tensor(
            np.stack([samples[i]['src_shape'] for i in indices]))
        self.angles = torch.tensor(
            np.stack([samples[i]['angles'] for i in indices]))
        self.masks = torch.tensor(
            np.stack([samples[i]['mask'] for i in indices]))
        self.descriptors = torch.tensor(
            np.stack([samples[i]['descriptor'] for i in indices]))

        print(f"[{split}] {len(self.src_shapes)} samples, "
              f"desc_dim={self.data_config['desc_dim']}")

    def __len__(self):
        return len(self.src_shapes)

    def __getitem__(self, idx):
        return (self.src_shapes[idx], self.descriptors[idx],
                self.angles[idx], self.masks[idx])


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = SyntheticIKDataset(args.data, 'train')
    val_ds = SyntheticIKDataset(args.data, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    desc_dim = train_ds.data_config['desc_dim']

    model = UniversalChainIK(
        shape_dim=SHAPE_DIM,
        desc_dim=desc_dim,
        max_joints=MAX_JOINTS,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for src, desc, angles_gt, mask in train_loader:
            src = src.to(device)
            desc = desc.to(device)
            angles_gt = angles_gt.to(device)
            mask = mask.to(device)

            angles_pred = model(src, desc)
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
        with torch.no_grad():
            for src, desc, angles_gt, mask in val_loader:
                src = src.to(device)
                desc = desc.to(device)
                angles_gt = angles_gt.to(device)
                mask = mask.to(device)

                angles_pred = model(src, desc)
                loss = ((angles_pred - angles_gt) ** 2 * mask).sum() / mask.sum()
                val_losses.append(loss.item())

        tr_loss = np.mean(train_losses)
        vr_loss = np.mean(val_losses)

        # Rough angle error: [-1,1] → typical range ~2rad → 1 unit ≈ 1rad
        angle_err_deg = np.sqrt(vr_loss) * 57.3

        if epoch % 10 == 0 or epoch <= 5:
            print(f"[{epoch:3d}/{args.epochs}] "
                  f"train={tr_loss:.6f} val={vr_loss:.6f} ~{angle_err_deg:.1f}deg")

        if vr_loss < best_val:
            best_val = vr_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': vr_loss,
                'config': {
                    'shape_dim': SHAPE_DIM,
                    'desc_dim': desc_dim,
                    'max_joints': MAX_JOINTS,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                },
            }, ckpt_dir / 'universal_ik_best.pt')

        if epoch % 100 == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch},
                       ckpt_dir / f'universal_ik_ep{epoch}.pt')

    print(f"\nDone. Best val_loss={best_val:.6f} (~{np.sqrt(best_val)*57.3:.1f}deg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--ckpt_dir", default="chain_nn/checkpoints")
    args = parser.parse_args()
    train(args)