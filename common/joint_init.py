"""
Data-driven Joint Initialization
==================================
데이터상의 joint 각도 분포 + 로봇 가동범위를 분석하여
가동범위가 최대화되는 초기값을 자동 계산.

qpos0 (전부 0)으로 시작하면 joint limit 경계에 붙어서
LM이 한쪽으로만 움직일 수 있는 문제를 해결.

Usage:
  # 1회: 분포 분석 + 초기값 저장
  python chain_flow/joint_init.py \
    --data chain_flow/data/ik_data.pkl --robot unitree_g1

  # chain_motion_retarget.py에서 자동 로드
"""

import sys
import os
import json
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


INIT_DIR = Path(__file__).parent / "data"


def analyze_joint_distribution(data_path, robot):
    """ik_data.pkl에서 특정 로봇의 per-joint 각도 분포 분석.

    Returns:
        joint_stats: dict {qposadr: {mean, median, std, p5, p95, lo, hi, init}}
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    chain_meta = data['chain_meta']
    if robot not in chain_meta:
        print(f"Robot '{robot}' not in data. Available: {list(chain_meta.keys())}")
        return None

    # Per-chain: collect angle distributions
    samples = data['samples']
    max_joints = data['config']['max_joints']

    # Group by chain
    chain_angles = {}  # chain_name -> list of (n_joints,) angle arrays
    for s in samples:
        if s['robot'] != robot:
            continue
        cname = s['chain']
        if cname not in chain_angles:
            chain_angles[cname] = []
        n = int(s['mask'].sum())
        # Denormalize from [-1,1] to actual angles
        meta = chain_meta[robot][cname]
        lo, hi = meta['lo'][:n], meta['hi'][:n]
        mid = (hi + lo) / 2
        hr = np.maximum((hi - lo) / 2, 1e-6)
        angles = s['angles'][:n] * hr + mid
        chain_angles[cname] = chain_angles.get(cname, [])
        chain_angles[cname].append(angles)

    # Analyze per-joint distribution
    print(f"\n{'='*70}")
    print(f"  Joint Distribution Analysis: {robot}")
    print(f"{'='*70}")

    joint_stats = {}  # will map to qposadr later

    for cname, angle_list in sorted(chain_angles.items()):
        meta = chain_meta[robot][cname]
        n = meta['n_joints']
        all_angles = np.array(angle_list)  # (N_samples, n_joints)

        print(f"\n  Chain: {cname} ({n} joints, {len(angle_list)} samples)")
        print(f"  {'Joint':>5s}  {'Limit_Lo':>9s} {'Limit_Hi':>9s} | "
              f"{'Data_P5':>9s} {'Data_Med':>9s} {'Data_P95':>9s} | "
              f"{'qpos0':>6s} {'Init':>9s}")
        print(f"  {'-'*80}")

        for ji in range(n):
            lo_j = float(meta['lo'][ji])
            hi_j = float(meta['hi'][ji])
            angles_j = all_angles[:, ji]

            median_j = float(np.median(angles_j))
            mean_j = float(np.mean(angles_j))
            std_j = float(np.std(angles_j))
            p5 = float(np.percentile(angles_j, 5))
            p95 = float(np.percentile(angles_j, 95))

            # 초기값: 데이터 분포 중심 (median), limit 안쪽 margin 유지
            margin = 0.05  # ~3 degrees
            init_j = float(np.clip(median_j, lo_j + margin, hi_j - margin))

            # 가동범위: init에서 양쪽 limit까지 거리
            room_lo = init_j - lo_j
            room_hi = hi_j - init_j

            print(f"  {ji:5d}  {lo_j:9.3f} {hi_j:9.3f} | "
                  f"{p5:9.3f} {median_j:9.3f} {p95:9.3f} | "
                  f"{'0.000':>6s} {init_j:9.3f}")

            joint_stats[f"{cname}_j{ji}"] = {
                'chain': cname,
                'joint_idx': ji,
                'lo': lo_j,
                'hi': hi_j,
                'mean': mean_j,
                'median': median_j,
                'std': std_j,
                'p5': p5,
                'p95': p95,
                'init': init_j,
                'room_lo': room_lo,
                'room_hi': room_hi,
            }

    return joint_stats, chain_meta[robot]


def compute_init_qpos(data_path, robot, model=None):
    """초기 qpos 벡터 계산.

    Args:
        data_path: ik_data.pkl 경로
        robot: 로봇 이름
        model: MuJoCo model (optional, for qposadr mapping)
    Returns:
        init_qpos: (nq,) numpy array
    """
    result = analyze_joint_distribution(data_path, robot)
    if result is None:
        return None

    joint_stats, chain_meta_robot = result

    if model is None:
        # Load model
        import mujoco as mj
        from general_motion_retargeting.params import ROBOT_XML_DICT
        xml_path = str(ROBOT_XML_DICT[robot])
        model = mj.MjModel.from_xml_path(xml_path)

    import mujoco as mj

    init_qpos = model.qpos0.copy()
    has_free = model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
    if has_free:
        init_qpos[2] = 0.793  # default height
        init_qpos[3] = 1.0    # quaternion w

    # chain_meta에서 chain → joint mapping 복원
    # chain별로 body_ids를 모르니까 model에서 직접 추출
    for cname, meta in chain_meta_robot.items():
        n = meta['n_joints']
        # Find joints for this chain by matching count
        # (chain_meta doesn't store qposadr, so we use the stats)
        for ji in range(n):
            key = f"{cname}_j{ji}"
            if key in joint_stats:
                init_val = joint_stats[key]['init']
                # Find the actual qposadr — need to match chain joints to model joints
                # For now, store by chain_name + index

    # Better approach: iterate model joints and match to chain stats
    # Build chain→body mapping from model
    children = {i: [] for i in range(model.nbody)}
    for i in range(1, model.nbody):
        children[int(model.body_parentid[i])].append(i)

    def body_name(bid):
        return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)

    # Extract chains (same logic as chain_motion_retarget._extract_chains)
    body_joints = {}
    start_j = 1 if has_free else 0
    for i in range(start_j, model.njnt):
        if model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE:
            continue
        bid = int(model.jnt_bodyid[i])
        body_joints.setdefault(bid, []).append(i)

    def has_jd(bid):
        if bid in body_joints:
            return True
        return any(has_jd(c) for c in children[bid])

    chains_extracted = []
    def trace(start_bid):
        body_ids = [start_bid]
        bid = start_bid
        while True:
            kids = children[bid]
            jkids = [c for c in kids if has_jd(c)]
            if len(jkids) == 0:
                break
            elif len(jkids) == 1:
                bid = jkids[0]
                body_ids.append(bid)
            else:
                break
        if len(body_ids) >= 2:
            name = f"{body_name(body_ids[0])}->{body_name(body_ids[-1])}"
            chains_extracted.append({'name': name, 'body_ids': body_ids})
        jkids = [c for c in children[bid] if has_jd(c)]
        if len(jkids) >= 2:
            for kid in jkids:
                trace(kid)

    root_bid = int(model.jnt_bodyid[0]) if has_free else 1
    jkids = [c for c in children[root_bid] if has_jd(c)]
    for kid in (jkids if len(jkids) >= 2 else jkids[:1]):
        trace(kid)

    # Map chain joints to qposadr and apply init values
    applied = 0
    for ch in chains_extracted:
        cname = ch['name']
        if cname not in chain_meta_robot:
            continue
        body_set = set(ch['body_ids'])
        ji_counter = 0
        for i in range(start_j, model.njnt):
            if model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE:
                continue
            if int(model.jnt_bodyid[i]) in body_set:
                key = f"{cname}_j{ji_counter}"
                if key in joint_stats:
                    qpa = int(model.jnt_qposadr[i])
                    init_qpos[qpa] = joint_stats[key]['init']
                    applied += 1
                ji_counter += 1

    print(f"\n  Applied {applied} joint initializations")
    return init_qpos


def save_init_qpos(data_path, robot, output_path=None):
    """초기값 계산하고 JSON으로 저장."""
    init_qpos = compute_init_qpos(data_path, robot)
    if init_qpos is None:
        return

    if output_path is None:
        INIT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = INIT_DIR / f"init_qpos_{robot}.json"

    with open(output_path, 'w') as f:
        json.dump({
            'robot': robot,
            'qpos': init_qpos.tolist(),
        }, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return init_qpos


def load_init_qpos(robot, model=None):
    """저장된 초기값 로드. 없으면 None 반환."""
    path = INIT_DIR / f"init_qpos_{robot}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    qpos = np.array(data['qpos'], dtype=np.float64)

    # Validate against model
    if model is not None and len(qpos) != model.nq:
        print(f"[JointInit] Warning: saved qpos length {len(qpos)} != model.nq {model.nq}")
        return None

    return qpos


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute data-driven joint initialization")
    parser.add_argument("--data", default="chain_flow/data/ik_data.pkl")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    save_init_qpos(args.data, args.robot, args.output)