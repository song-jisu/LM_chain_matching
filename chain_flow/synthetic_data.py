"""
Synthetic Chain IK Data Generator
===================================
임의의 chain 구조를 랜덤 생성하고 FK/bounded-LM으로 학습 데이터 생성.
특정 로봇에 의존하지 않는 universal chain IK solver 학습용.

생성 과정:
  1. 랜덤 chain 구조: N joints, link 길이 비, joint 축(x/y/z), joint limits
  2. 랜덤 joint angles → FK → chain shape (reachable target)
  3. 랜덤 perturbation → unreachable target → bounded LM → 최선 해 (closest feasible)
  4. (source shape, chain descriptor, target angles) 쌍으로 저장
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from chain_flow.chain_shape import normalize_chain_shape, K_POINTS
import pickle
import os
from tqdm import tqdm


# Joint 축 one-hot
AXIS_X = np.array([1, 0, 0], dtype=np.float32)
AXIS_Y = np.array([0, 1, 0], dtype=np.float32)
AXIS_Z = np.array([0, 0, 1], dtype=np.float32)
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

MAX_JOINTS = 12
SHAPE_DIM = K_POINTS * 3  # 24
# descriptor: n_joints(1) + link_ratios(MAX_JOINTS) + axes(MAX_JOINTS*3) + limits(MAX_JOINTS*2)
DESC_DIM = 1 + MAX_JOINTS + MAX_JOINTS * 3 + MAX_JOINTS * 2  # 73


def random_chain_config(n_joints_range=(2, 8)):
    """랜덤 chain 구조 생성."""
    n = np.random.randint(n_joints_range[0], n_joints_range[1] + 1)

    # Link 길이 비: Dirichlet 분포로 자연스러운 비율
    link_ratios = np.random.dirichlet(np.ones(n) * 2)  # sum = 1

    # Joint 축: 각 joint별 x/y/z 중 하나
    axes = [AXES[np.random.randint(3)] for _ in range(n)]

    # Joint 한계: 다양한 범위
    limits = []
    for _ in range(n):
        # 로봇 관절의 실제적인 범위
        range_type = np.random.choice(['narrow', 'medium', 'wide', 'full'])
        if range_type == 'narrow':
            span = np.random.uniform(0.3, 1.0)
        elif range_type == 'medium':
            span = np.random.uniform(1.0, 2.0)
        elif range_type == 'wide':
            span = np.random.uniform(2.0, 3.0)
        else:
            span = np.random.uniform(3.0, 2 * np.pi)
        center = np.random.uniform(-0.5, 0.5)
        lo = center - span / 2
        hi = center + span / 2
        limits.append((lo, hi))

    return {
        'n_joints': n,
        'link_ratios': np.array(link_ratios, dtype=np.float32),
        'axes': axes,
        'limits': limits,
    }


def chain_fk(angles, config):
    """순방향 운동학: joint angles → body positions.

    Args:
        angles: (n_joints,) — joint angles
        config: chain config dict
    Returns:
        positions: (n_joints+1, 3) — body positions (시작점 포함)
    """
    n = config['n_joints']
    ratios = config['link_ratios']
    axes = config['axes']

    positions = [np.zeros(3)]
    cumul_rot = R.identity()

    for i in range(n):
        # Joint rotation
        local_rot = R.from_rotvec(angles[i] * axes[i])
        cumul_rot = cumul_rot * local_rot
        # Link direction (rest direction = +X, scaled by link ratio)
        link_vec = cumul_rot.apply(np.array([ratios[i], 0, 0]))
        positions.append(positions[-1] + link_vec)

    return np.array(positions)


def closest_feasible_angles(target_shape, config, max_nfev=100):
    """도달 불가능한 target에 대해 bounded LM으로 최선 해 찾기.

    Args:
        target_shape: (K, 3) — normalized target chain shape
        config: chain config
    Returns:
        best_angles: (n_joints,) — joint limit 내 최선 해
        achieved_shape: (K, 3) — 실제 달성된 shape
    """
    n = config['n_joints']
    lo = np.array([l[0] for l in config['limits']])
    hi = np.array([l[1] for l in config['limits']])

    def residual(angles):
        pos = chain_fk(angles, config)
        shape, _, _ = normalize_chain_shape(pos)
        return (shape - target_shape).reshape(-1)

    # 여러 초기값에서 시도, 최선 선택
    best_cost = float('inf')
    best_x = np.zeros(n)

    for _ in range(3):
        x0 = np.random.uniform(lo, hi)
        x0 = np.clip(x0, lo + 1e-4, hi - 1e-4)
        try:
            result = least_squares(residual, x0, bounds=(lo, hi),
                                   method='trf', max_nfev=max_nfev)
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x
        except Exception:
            pass

    achieved_pos = chain_fk(best_x, config)
    achieved_shape, _, _ = normalize_chain_shape(achieved_pos)
    return best_x.astype(np.float32), achieved_shape


def config_to_descriptor(config):
    """Chain config → 고정 크기 descriptor vector.

    Format: [n_joints_norm, link_ratios(MAX), axes_onehot(MAX*3), limits_norm(MAX*2)]
    """
    n = config['n_joints']
    desc = np.zeros(DESC_DIM, dtype=np.float32)

    # n_joints normalized
    desc[0] = n / MAX_JOINTS

    # link ratios (padded)
    offset = 1
    desc[offset:offset + n] = config['link_ratios']

    # axes one-hot (padded)
    offset = 1 + MAX_JOINTS
    for i in range(n):
        axis_idx = np.argmax(config['axes'][i])
        desc[offset + i * 3 + axis_idx] = 1.0

    # limits normalized to [-1, 1] range / pi
    offset = 1 + MAX_JOINTS + MAX_JOINTS * 3
    for i in range(n):
        desc[offset + i * 2] = config['limits'][i][0] / np.pi
        desc[offset + i * 2 + 1] = config['limits'][i][1] / np.pi

    return desc


def generate_sample(config, unreachable_prob=0.3):
    """단일 학습 샘플 생성.

    Args:
        config: chain config
        unreachable_prob: 도달 불가능한 target 생성 확률
    Returns:
        sample dict or None
    """
    n = config['n_joints']
    lo = np.array([l[0] for l in config['limits']])
    hi = np.array([l[1] for l in config['limits']])

    # Random angles within limits
    angles = np.random.uniform(lo, hi).astype(np.float32)

    # FK → shape
    positions = chain_fk(angles, config)
    shape, _, _ = normalize_chain_shape(positions)

    is_unreachable = False

    if np.random.random() < unreachable_prob:
        # Perturb shape to make it unreachable
        noise_scale = np.random.uniform(0.05, 0.3)
        shape_perturbed = shape + np.random.randn(*shape.shape).astype(np.float32) * noise_scale
        # Find closest feasible
        angles, shape = closest_feasible_angles(shape_perturbed, config)
        is_unreachable = True
    else:
        # Reachable: use the FK-generated shape directly
        pass

    # Normalize angles to [-1, 1]
    mid = (hi + lo) / 2
    half_range = np.maximum((hi - lo) / 2, 1e-6)
    angles_norm = np.clip((angles - mid) / half_range, -1, 1).astype(np.float32)

    # Pad
    angles_padded = np.zeros(MAX_JOINTS, dtype=np.float32)
    angles_padded[:n] = angles_norm
    mask = np.zeros(MAX_JOINTS, dtype=np.float32)
    mask[:n] = 1.0

    return {
        'src_shape': shape.reshape(-1).astype(np.float32),
        'angles': angles_padded,
        'mask': mask,
        'descriptor': config_to_descriptor(config),
        'n_joints': n,
        'is_unreachable': is_unreachable,
    }


def generate_dataset(n_samples, output_path, n_joints_range=(2, 8),
                     unreachable_prob=0.3, n_configs=1000):
    """대규모 합성 데이터셋 생성.

    Args:
        n_samples: 총 샘플 수
        output_path: 저장 경로
        n_joints_range: joint 수 범위
        unreachable_prob: 도달 불가능 target 비율
        n_configs: 사용할 고유 chain 구조 수
    """
    print(f"Generating {n_samples} samples from {n_configs} chain configs...")

    # 다양한 chain 구조 미리 생성
    configs = [random_chain_config(n_joints_range) for _ in range(n_configs)]

    samples = []
    for i in tqdm(range(n_samples), desc="Generating"):
        config = configs[i % n_configs]
        sample = generate_sample(config, unreachable_prob)
        if sample is not None:
            samples.append(sample)

    # 통계
    n_unreach = sum(1 for s in samples if s['is_unreachable'])
    joint_counts = {}
    for s in samples:
        nj = s['n_joints']
        joint_counts[nj] = joint_counts.get(nj, 0) + 1

    print(f"\nTotal: {len(samples)} samples")
    print(f"Unreachable: {n_unreach} ({n_unreach/len(samples)*100:.1f}%)")
    print(f"Joint count distribution: {dict(sorted(joint_counts.items()))}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'samples': samples,
            'config': {
                'shape_dim': SHAPE_DIM,
                'desc_dim': DESC_DIM,
                'max_joints': MAX_JOINTS,
                'k_points': K_POINTS,
                'n_configs': n_configs,
                'n_joints_range': n_joints_range,
            },
        }, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200000)
    parser.add_argument("--n_configs", type=int, default=5000)
    parser.add_argument("--n_joints_min", type=int, default=2)
    parser.add_argument("--n_joints_max", type=int, default=8)
    parser.add_argument("--unreachable_prob", type=float, default=0.3)
    parser.add_argument("--output", default="chain_flow/data/synthetic_ik_data.pkl")
    args = parser.parse_args()

    generate_dataset(
        args.n_samples, args.output,
        n_joints_range=(args.n_joints_min, args.n_joints_max),
        unreachable_prob=args.unreachable_prob,
        n_configs=args.n_configs,
    )