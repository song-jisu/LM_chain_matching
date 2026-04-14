"""
Chain Shape: Skeleton-agnostic motion representation
=====================================================
어떤 skeleton이든 serial chain 집합으로 분해하고,
각 chain의 shape을 고정 크기 표현으로 정규화.

Chain shape representation:
  - K개의 등간격 sample points (chain 길이 기준)
  - chain start를 원점으로, chain 길이로 정규화
  - 방향은 유지 (orientation-dependent — forward alignment으로 처리)

Shape: (K, 3) per chain per frame
  K=8 (default): 8개 sample points × 3D = 24 dims
"""

import numpy as np
from scipy.interpolate import interp1d


# Default number of sample points per chain
K_POINTS = 8


def resample_chain(positions, k=K_POINTS):
    """Chain body positions를 k개 등간격 점으로 리샘플링.

    Args:
        positions: (n_bodies, 3) — chain body의 3D positions
        k: output sample count
    Returns:
        (k, 3) — 등간격 리샘플된 positions
    """
    n = len(positions)
    if n < 2:
        return np.tile(positions[0], (k, 1))

    # 각 body 사이 누적 거리 계산
    dists = np.zeros(n)
    for i in range(1, n):
        dists[i] = dists[i-1] + np.linalg.norm(positions[i] - positions[i-1])

    total_len = dists[-1]
    if total_len < 1e-8:
        return np.tile(positions[0], (k, 1))

    # 0~1 정규화
    t_orig = dists / total_len
    t_new = np.linspace(0, 1, k)

    # 각 축별 보간
    resampled = np.zeros((k, 3))
    for axis in range(3):
        f = interp1d(t_orig, positions[:, axis], kind='linear')
        resampled[:, axis] = f(t_new)

    return resampled


def normalize_chain_shape(positions, k=K_POINTS):
    """Chain positions를 정규화된 shape으로 변환.

    1. k개 점으로 리샘플
    2. chain start를 원점으로 이동
    3. chain 길이로 나누어 scale normalization

    Args:
        positions: (n_bodies, 3) — raw body positions
        k: sample count
    Returns:
        shape: (k, 3) — normalized chain shape
        chain_start: (3,) — original chain start position
        chain_length: float — original chain length
    """
    positions = np.asarray(positions, dtype=np.float32)

    # 리샘플
    resampled = resample_chain(positions, k)

    chain_start = resampled[0].copy()

    # chain 길이 계산
    chain_length = 0.0
    for i in range(1, k):
        chain_length += np.linalg.norm(resampled[i] - resampled[i-1])
    chain_length = max(chain_length, 1e-6)

    # 정규화: 시작점 원점, 길이 1
    shape = (resampled - chain_start) / chain_length

    return shape, chain_start, chain_length


def denormalize_chain_shape(shape, chain_start, chain_length):
    """정규화된 shape을 실제 positions로 복원.

    Args:
        shape: (k, 3) — normalized
        chain_start: (3,)
        chain_length: float
    Returns:
        positions: (k, 3) — world positions
    """
    return shape * chain_length + chain_start


def extract_chain_descriptor(chain_info, model, data=None):
    """Chain의 topology descriptor 추출 (flow conditioning용).

    Args:
        chain_info: dict with 'body_ids', 'name'
        model: MuJoCo model
        data: MuJoCo data (optional, for rest pose)
    Returns:
        descriptor: dict with chain properties
    """
    import mujoco as mj

    body_ids = chain_info['body_ids']
    n_bodies = len(body_ids)

    # Rest pose에서 link lengths 계산
    if data is None:
        data = mj.MjData(model)
        data.qpos[:] = model.qpos0
        if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            data.qpos[2] = 0.793
            data.qpos[3] = 1.0
        mj.mj_forward(model, data)

    positions = np.array([data.xpos[bid].copy() for bid in body_ids])
    link_lengths = []
    for i in range(1, n_bodies):
        link_lengths.append(np.linalg.norm(positions[i] - positions[i-1]))

    total_length = sum(link_lengths) if link_lengths else 0.01
    link_ratios = [l / total_length for l in link_lengths] if link_lengths else []

    # Joint 정보
    n_joints = 0
    joint_ranges = []
    start_j = 1 if (model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE) else 0
    body_set = set(body_ids)
    for i in range(start_j, model.njnt):
        if model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE:
            continue
        if int(model.jnt_bodyid[i]) in body_set:
            n_joints += 1
            if model.jnt_limited[i]:
                joint_ranges.append(model.jnt_range[i].tolist())
            else:
                joint_ranges.append([-3.14, 3.14])

    return {
        'n_bodies': n_bodies,
        'n_joints': n_joints,
        'total_length': total_length,
        'link_ratios': link_ratios,
        'joint_ranges': joint_ranges,
    }


def descriptor_to_vector(desc, max_links=10, max_joints=12):
    """Chain descriptor를 고정 크기 벡터로 변환 (NN input용).

    Returns:
        (dim,) vector:
          [n_bodies(1), n_joints(1), total_length(1),
           link_ratios(max_links), joint_range_spans(max_joints)]
    """
    vec = np.zeros(3 + max_links + max_joints, dtype=np.float32)
    vec[0] = desc['n_bodies'] / 10.0  # normalize
    vec[1] = desc['n_joints'] / 12.0
    vec[2] = desc['total_length']

    for i, r in enumerate(desc['link_ratios'][:max_links]):
        vec[3 + i] = r

    for i, (lo, hi) in enumerate(desc['joint_ranges'][:max_joints]):
        vec[3 + max_links + i] = (hi - lo) / (2 * 3.14)  # normalized range

    return vec


DESCRIPTOR_DIM = 3 + 10 + 12  # 25
SHAPE_DIM = K_POINTS * 3      # 24


def extract_all_chain_shapes(retargeter, human_data, k=K_POINTS):
    """retarget 후 source/target 양쪽 chain shapes 추출.

    Args:
        retargeter: ChainMotionRetargeting (retarget() 호출 후)
        human_data: source BVH frame data
        k: sample points per chain
    Returns:
        pairs: list of {
            'chain_name': str,
            'src_shape': (k, 3),
            'tgt_shape': (k, 3),
            'src_start': (3,), 'src_length': float,
            'tgt_start': (3,), 'tgt_length': float,
            'tgt_descriptor': (desc_dim,),
        }
    """
    scaled = retargeter.scaled_human_data
    if scaled is None:
        return []

    pairs = []
    for ch in retargeter.chains:
        # Source: mapped human body positions
        mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                  if hb and hb in scaled]
        if len(mapped) < 2:
            continue

        src_positions = np.array([np.asarray(scaled[hb][0]) for _, hb in mapped])
        src_shape, src_start, src_length = normalize_chain_shape(src_positions, k)

        # Target: robot body positions (from retarget result)
        tgt_body_ids = [ch['body_ids'][bi] for bi, _ in mapped]
        tgt_positions = np.array([retargeter.data.xpos[bid].copy() for bid in tgt_body_ids])
        tgt_shape, tgt_start, tgt_length = normalize_chain_shape(tgt_positions, k)

        # Chain descriptor
        desc = extract_chain_descriptor(ch, retargeter.model)
        desc_vec = descriptor_to_vector(desc)

        pairs.append({
            'chain_name': ch['name'],
            'src_shape': src_shape.astype(np.float32),
            'tgt_shape': tgt_shape.astype(np.float32),
            'src_start': src_start.astype(np.float32),
            'src_length': float(src_length),
            'tgt_start': tgt_start.astype(np.float32),
            'tgt_length': float(tgt_length),
            'tgt_descriptor': desc_vec,
        })

    return pairs