"""
Dataset Generation & Loading
==============================
1. LaFAN1 BVH + chain retargeter로 paired (source, target) chain shapes 수집
2. Temporal windowing: W=32 frame windows with sliding
3. PyTorch Dataset class for training
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Add parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from chain_flow.chain_shape import (
    normalize_chain_shape, K_POINTS, DESCRIPTOR_DIM,
    extract_chain_descriptor, descriptor_to_vector,
)


WINDOW_SIZE = 32
STRIDE = 16  # sliding window stride
SHAPE_DIM = K_POINTS * 3  # 24


def generate_dataset(
    bvh_dir,
    bvh_format,
    robots,
    output_path,
    max_files=None,
    max_frames_per_file=None,
):
    """LaFAN1 BVH로부터 chain shape 학습 데이터 생성.

    각 BVH 파일 × 각 robot에 대해:
      1. chain retarget 수행
      2. 매 프레임 source/target chain shapes 수집
      3. temporal windows로 분할

    Args:
        bvh_dir: BVH 파일 디렉토리
        bvh_format: 'lafan1', 'nokov', etc.
        robots: list of robot names (e.g., ['unitree_g1', 'fourier_n1'])
        output_path: 저장 경로 (.pkl)
        max_files: 최대 BVH 파일 수
        max_frames_per_file: 파일당 최대 프레임 수
    """
    # Lazy imports to avoid glfw
    import types
    pkg = types.ModuleType('general_motion_retargeting')
    pkg.__path__ = [str(Path(__file__).parent.parent / 'general_motion_retargeting')]
    sys.modules['general_motion_retargeting'] = pkg

    from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT
    pkg.ROBOT_XML_DICT = ROBOT_XML_DICT
    pkg.IK_CONFIG_DICT = IK_CONFIG_DICT
    pkg.ROBOT_BASE_DICT = ROBOT_BASE_DICT

    from general_motion_retargeting.chain_motion_retarget import ChainMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    bvh_files = sorted([
        os.path.join(bvh_dir, f) for f in os.listdir(bvh_dir) if f.endswith('.bvh')
    ])
    if max_files:
        bvh_files = bvh_files[:max_files]

    all_windows = []  # list of window dicts
    chain_descriptors = {}  # robot_name -> chain_name -> descriptor_vec

    for robot in robots:
        print(f"\n=== Robot: {robot} ===")

        for bvh_path in tqdm(bvh_files, desc=f"{robot}"):
            try:
                frames, human_height, bone_hierarchy = load_bvh_file(bvh_path, format=bvh_format)
                if max_frames_per_file:
                    frames = frames[:max_frames_per_file]

                retargeter = ChainMotionRetargeting(
                    src_human=f"bvh_{bvh_format}" if bvh_format != "robot" else "robot",
                    tgt_robot=robot,
                    actual_human_height=human_height,
                    verbose=False,
                )
                retargeter._bone_hierarchy = bone_hierarchy

                # Per-chain per-frame shape 수집
                chain_frames = {}  # chain_name -> list of (src_shape, tgt_shape) per frame

                for fi, human_data in enumerate(frames):
                    try:
                        qpos = retargeter.retarget(human_data)
                    except Exception:
                        continue

                    scaled = retargeter.scaled_human_data
                    if scaled is None:
                        continue

                    for ch in retargeter.chains:
                        mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                                  if hb and hb in scaled]
                        if len(mapped) < 2:
                            continue

                        cname = ch['name']
                        if cname not in chain_frames:
                            chain_frames[cname] = []

                        # Source shape
                        src_pos = np.array([np.asarray(scaled[hb][0]) for _, hb in mapped])
                        src_shape, _, _ = normalize_chain_shape(src_pos)

                        # Target shape
                        tgt_bids = [ch['body_ids'][bi] for bi, _ in mapped]
                        tgt_pos = np.array([retargeter.data.xpos[bid].copy() for bid in tgt_bids])
                        tgt_shape, _, _ = normalize_chain_shape(tgt_pos)

                        chain_frames[cname].append((
                            src_shape.astype(np.float32),
                            tgt_shape.astype(np.float32),
                        ))

                        # Descriptor (once per chain per robot)
                        if robot not in chain_descriptors:
                            chain_descriptors[robot] = {}
                        if cname not in chain_descriptors[robot]:
                            desc = extract_chain_descriptor(ch, retargeter.model)
                            chain_descriptors[robot][cname] = descriptor_to_vector(desc)

                # Windowing
                for cname, frame_list in chain_frames.items():
                    n_frames = len(frame_list)
                    if n_frames < WINDOW_SIZE:
                        continue

                    desc_vec = chain_descriptors[robot][cname]

                    for start in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
                        window_src = np.stack([f[0].reshape(-1) for f in frame_list[start:start+WINDOW_SIZE]])
                        window_tgt = np.stack([f[1].reshape(-1) for f in frame_list[start:start+WINDOW_SIZE]])

                        all_windows.append({
                            'src': window_src.astype(np.float32),     # (W, K*3)
                            'tgt': window_tgt.astype(np.float32),     # (W, K*3)
                            'desc': desc_vec.astype(np.float32),      # (desc_dim,)
                            'robot': robot,
                            'chain': cname,
                            'bvh': os.path.basename(bvh_path),
                        })

            except Exception as e:
                print(f"  Error {os.path.basename(bvh_path)}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\nTotal windows: {len(all_windows)}")
    print(f"Robots: {list(chain_descriptors.keys())}")
    for r in chain_descriptors:
        print(f"  {r}: {list(chain_descriptors[r].keys())}")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'windows': all_windows,
            'descriptors': chain_descriptors,
            'config': {
                'window_size': WINDOW_SIZE,
                'stride': STRIDE,
                'k_points': K_POINTS,
                'shape_dim': SHAPE_DIM,
                'desc_dim': DESCRIPTOR_DIM,
            },
        }, f)
    print(f"Saved to {output_path}")


class ChainFlowDataset:
    """PyTorch-compatible dataset for chain flow training."""

    def __init__(self, data_path, split='train', val_ratio=0.1, seed=42):
        import torch

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        windows = data['windows']
        config = data['config']
        self.window_size = config['window_size']
        self.shape_dim = config['shape_dim']
        self.desc_dim = config['desc_dim']

        # Train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(windows))
        n_val = max(1, int(len(windows) * val_ratio))

        if split == 'train':
            indices = indices[n_val:]
        else:
            indices = indices[:n_val]

        self.src = torch.tensor(np.stack([windows[i]['src'] for i in indices]))
        self.tgt = torch.tensor(np.stack([windows[i]['tgt'] for i in indices]))
        self.desc = torch.tensor(np.stack([windows[i]['desc'] for i in indices]))

        print(f"[{split}] {len(self.src)} windows, shape_dim={self.shape_dim}")

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.desc[idx]


# =====================================================================
# CLI: generate dataset
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate chain flow training data")
    parser.add_argument("--bvh_dir", required=True)
    parser.add_argument("--format", default="lafan1")
    parser.add_argument("--robots", nargs='+',
                        default=["unitree_g1", "fourier_n1", "booster_t1"])
    parser.add_argument("--output", default="chain_flow/data/train_data.pkl")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    generate_dataset(
        bvh_dir=args.bvh_dir,
        bvh_format=args.format,
        robots=args.robots,
        output_path=args.output,
        max_files=args.max_files,
        max_frames_per_file=args.max_frames,
    )