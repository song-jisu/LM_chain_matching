"""
Direct IK 학습 데이터 생성
===========================
Chain LM으로 retarget → (source shape, target joint angles) 쌍 수집.
Inference에서는 LM 없이 모델 forward pass만으로 joint angles 예측.
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import mujoco as mj

sys.path.insert(0, str(Path(__file__).parent.parent))

from chain_flow.chain_shape import (
    normalize_chain_shape, K_POINTS, DESCRIPTOR_DIM,
    extract_chain_descriptor, descriptor_to_vector,
)
from chain_flow.direct_model import MAX_JOINTS

SHAPE_DIM = K_POINTS * 3  # 24


def generate_ik_dataset(bvh_dir, bvh_format, robots, output_path,
                        max_files=None, max_frames_per_file=None):
    """Chain LM retarget 결과에서 (src_shape, tgt_joint_angles) 수집."""

    # Lazy imports
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

    all_samples = []
    chain_meta = {}  # robot -> chain_name -> {descriptor, joint_lo, joint_hi, n_joints}

    for robot in robots:
        print(f"\n=== Robot: {robot} ===")

        for bvh_path in tqdm(bvh_files, desc=f"{robot}"):
            try:
                frames, human_height, bone_hierarchy = load_bvh_file(
                    bvh_path, format=bvh_format)
                if max_frames_per_file:
                    frames = frames[:max_frames_per_file]

                retargeter = ChainMotionRetargeting(
                    src_human=f"bvh_{bvh_format}" if bvh_format != "robot" else "robot",
                    tgt_robot=robot,
                    actual_human_height=human_height,
                    verbose=False,
                )
                retargeter._bone_hierarchy = bone_hierarchy

                # Chain별 joint indices 미리 수집
                chain_joint_info = {}
                for ch in retargeter.chains:
                    body_set = set(ch['body_ids'])
                    qpa_list = []
                    lo_list = []
                    hi_list = []
                    start_j = 1 if retargeter.has_freejoint else 0
                    for ji in range(start_j, retargeter.model.njnt):
                        if retargeter.model.jnt_type[ji] != mj.mjtJoint.mjJNT_HINGE:
                            continue
                        if int(retargeter.model.jnt_bodyid[ji]) in body_set:
                            qpa_list.append(int(retargeter.model.jnt_qposadr[ji]))
                            lo, hi = retargeter.model.jnt_range[ji] if retargeter.model.jnt_limited[ji] else (-np.pi, np.pi)
                            lo_list.append(float(lo))
                            hi_list.append(float(hi))
                    chain_joint_info[ch['name']] = {
                        'qpa': qpa_list,
                        'lo': np.array(lo_list, dtype=np.float32),
                        'hi': np.array(hi_list, dtype=np.float32),
                        'n_joints': len(qpa_list),
                    }

                    # Meta (once per robot per chain)
                    if robot not in chain_meta:
                        chain_meta[robot] = {}
                    if ch['name'] not in chain_meta[robot]:
                        desc = extract_chain_descriptor(ch, retargeter.model)
                        chain_meta[robot][ch['name']] = {
                            'descriptor': descriptor_to_vector(desc),
                            'lo': np.array(lo_list, dtype=np.float32),
                            'hi': np.array(hi_list, dtype=np.float32),
                            'n_joints': len(qpa_list),
                        }

                # Per-frame retarget + 수집
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
                        jinfo = chain_joint_info[cname]
                        if jinfo['n_joints'] == 0:
                            continue

                        # Source: normalized chain shape
                        src_pos = np.array([np.asarray(scaled[hb][0]) for _, hb in mapped])
                        src_shape, _, _ = normalize_chain_shape(src_pos)

                        # Target: actual joint angles (from LM result)
                        angles = np.array([qpos[a] for a in jinfo['qpa']], dtype=np.float32)

                        # Normalize angles to [-1, 1]
                        mid = (jinfo['hi'] + jinfo['lo']) / 2
                        half_range = (jinfo['hi'] - jinfo['lo']) / 2
                        half_range = np.maximum(half_range, 1e-6)
                        angles_norm = (angles - mid) / half_range
                        angles_norm = np.clip(angles_norm, -1, 1)

                        # Pad to MAX_JOINTS
                        angles_padded = np.zeros(MAX_JOINTS, dtype=np.float32)
                        angles_padded[:jinfo['n_joints']] = angles_norm
                        mask = np.zeros(MAX_JOINTS, dtype=np.float32)
                        mask[:jinfo['n_joints']] = 1.0

                        all_samples.append({
                            'src_shape': src_shape.reshape(-1).astype(np.float32),
                            'angles': angles_padded,
                            'mask': mask,
                            'descriptor': chain_meta[robot][cname]['descriptor'],
                            'robot': robot,
                            'chain': cname,
                        })

            except Exception as e:
                print(f"  Error {os.path.basename(bvh_path)}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nTotal samples: {len(all_samples)}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'samples': all_samples,
            'chain_meta': chain_meta,
            'config': {
                'shape_dim': SHAPE_DIM,
                'desc_dim': DESCRIPTOR_DIM,
                'max_joints': MAX_JOINTS,
                'k_points': K_POINTS,
            },
        }, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_dir", required=True)
    parser.add_argument("--format", default="lafan1")
    parser.add_argument("--robots", nargs='+', default=["unitree_g1", "fourier_n1"])
    parser.add_argument("--output", default="chain_flow/data/ik_data.pkl")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    generate_ik_dataset(args.bvh_dir, args.format, args.robots, args.output,
                        args.max_files, args.max_frames)