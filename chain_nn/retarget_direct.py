"""
Direct IK Retargeting: 실시간 추론
====================================
학습된 모델로 source BVH → target joint angles를 forward pass만으로 출력.
Chain LM 최적화 없이 실시간 retarget.

Usage:
  python chain_nn/retarget_direct.py \
    --bvh_file <input.bvh> --format lafan1 --robot unitree_g1 \
    --ckpt chain_nn/checkpoints/direct_ik_best.pt \
    --visualize
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.chain_shape import (
    normalize_chain_shape, K_POINTS, DESCRIPTOR_DIM,
    extract_chain_descriptor, descriptor_to_vector,
)
from chain_nn.direct_model import DirectChainIK, MAX_JOINTS


def main():
    parser = argparse.ArgumentParser(description="Direct IK retargeting (real-time)")
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--format", default="lafan1")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--ckpt", required=True, help="direct_ik_best.pt")
    parser.add_argument("--meta", default="chain_nn/data/ik_data.pkl",
                        help="Training data (for chain_meta)")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--lm_steps", type=int, default=0,
                        help="LM refinement steps after NN prediction (0=NN only, 3=hybrid)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Setup package ──
    import types
    pkg = types.ModuleType('general_motion_retargeting')
    pkg.__path__ = [str(Path(__file__).parent.parent / 'general_motion_retargeting')]
    sys.modules['general_motion_retargeting'] = pkg
    from general_motion_retargeting.params import (
        ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT,
    )
    pkg.ROBOT_XML_DICT = ROBOT_XML_DICT
    pkg.IK_CONFIG_DICT = IK_CONFIG_DICT
    pkg.ROBOT_BASE_DICT = ROBOT_BASE_DICT
    pkg.VIEWER_CAM_DISTANCE_DICT = VIEWER_CAM_DISTANCE_DICT

    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    import mujoco as mj

    # ── Load model ──
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = DirectChainIK(
        shape_dim=cfg['shape_dim'],
        desc_dim=cfg['desc_dim'],
        max_joints=cfg['max_joints'],
        hidden_dim=cfg['hidden_dim'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

    # ── Load chain meta (joint limits, descriptors) ──
    with open(args.meta, 'rb') as f:
        meta_data = pickle.load(f)
    chain_meta = meta_data['chain_meta']
    if args.robot not in chain_meta:
        print(f"ERROR: robot '{args.robot}' not in training data. Available: {list(chain_meta.keys())}")
        sys.exit(1)

    # ── Load BVH ──
    frames, height, hierarchy = load_bvh_file(args.bvh_file, format=args.format)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames")

    # ── Robot model (for FK visualization only) ──
    xml_path = str(ROBOT_XML_DICT[args.robot])
    mj_model = mj.MjModel.from_xml_path(xml_path)
    mj_data = mj.MjData(mj_model)

    has_freejoint = (mj_model.njnt > 0 and mj_model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE)

    # ── Chain setup: IK config에서 body mapping 로드 ──
    src_human = f"bvh_{args.format}" if args.format != "robot" else "robot"
    from general_motion_retargeting.chain_motion_retarget import ChainMotionRetargeting
    setup_retargeter = ChainMotionRetargeting(
        src_human=src_human, tgt_robot=args.robot,
        actual_human_height=height, verbose=False,
    )
    setup_retargeter._bone_hierarchy = hierarchy
    # Do one retarget to initialize mapping
    setup_retargeter.retarget(frames[0])

    chains = setup_retargeter.chains
    human_root_name = setup_retargeter.human_root_name

    # Chain별 joint qposadr + limits + axes + link ratios (MuJoCo에서 직접 추출)
    # Rest pose FK for link lengths
    rest_data = mj.MjData(mj_model)
    rest_data.qpos[:] = mj_model.qpos0
    if has_freejoint:
        rest_data.qpos[2] = 0.793
        rest_data.qpos[3] = 1.0
    mj.mj_forward(mj_model, rest_data)

    chain_joint_map = {}
    for ch in chains:
        cname = ch['name']
        if cname not in chain_meta[args.robot]:
            continue
        meta = chain_meta[args.robot][cname]
        body_set = set(ch['body_ids'])
        qpa_list = []
        dof_list = []
        joint_axes = []
        lo_list = []
        hi_list = []
        start_j = 1 if has_freejoint else 0
        for ji in range(start_j, mj_model.njnt):
            if mj_model.jnt_type[ji] != mj.mjtJoint.mjJNT_HINGE:
                continue
            if int(mj_model.jnt_bodyid[ji]) in body_set:
                qpa_list.append(int(mj_model.jnt_qposadr[ji]))
                dof_list.append(int(mj_model.jnt_dofadr[ji]))
                joint_axes.append(mj_model.jnt_axis[ji].copy())
                lo, hi = mj_model.jnt_range[ji] if mj_model.jnt_limited[ji] else (-np.pi, np.pi)
                lo_list.append(float(lo))
                hi_list.append(float(hi))

        # Link ratios from rest pose body positions
        bids = ch['body_ids']
        link_lengths = []
        for bi in range(1, len(bids)):
            d = np.linalg.norm(rest_data.xpos[bids[bi]] - rest_data.xpos[bids[bi-1]])
            link_lengths.append(d)
        total_len = sum(link_lengths) if link_lengths else 1.0
        link_ratios = np.array([l / total_len for l in link_lengths], dtype=np.float32)

        chain_joint_map[cname] = {
            'qpa': qpa_list,
            'lo': np.array(lo_list, dtype=np.float32),
            'hi': np.array(hi_list, dtype=np.float32),
            'descriptor': torch.tensor(meta['descriptor'], device=device).unsqueeze(0),
            'n_joints': len(qpa_list),
            'dofadr': dof_list,
            'joint_axes': joint_axes,
            'link_ratios': link_ratios,
        }

    # ── Forward alignment (from setup retargeter) ──
    fwd_rot = setup_retargeter._fwd_rotation if hasattr(setup_retargeter, '_fwd_rotation') else None
    scale_table = setup_retargeter.human_scale_table

    # ── Viewer ──
    viewer = None
    if args.visualize:
        from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.motion_fps,
            transparent_robot=0,
        )

    # ── Real-time retarget loop ──
    print("Retargeting...")
    qpos = mj_model.qpos0.copy()
    prev_angles = {}  # chain_name → previous predicted angles tensor

    t0 = time.time()
    i = 0
    frame_count = 0

    while True:
        human_data = {k: [np.asarray(v[0]), np.asarray(v[1])]
                      for k, v in frames[i].items()}

        # Scale + ground (reuse setup_retargeter's logic)
        scaled = setup_retargeter._scale(human_data)
        scaled = setup_retargeter._ground(scaled)

        # Root position
        if has_freejoint and human_root_name in scaled:
            qpos[0:3] = scaled[human_root_name][0]
            if fwd_rot is not None:
                qpos[3:7] = fwd_rot.inv().as_quat(scalar_first=True)

        # Per-chain: model forward pass → joint angles
        with torch.no_grad():
            for ch in chains:
                cname = ch['name']
                if cname not in chain_joint_map:
                    continue

                mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                          if hb and hb in scaled]
                if len(mapped) < 2:
                    continue

                # Source shape
                src_pos = np.array([np.asarray(scaled[hb][0]) for _, hb in mapped])
                src_shape, _, _ = normalize_chain_shape(src_pos)
                src_tensor = torch.tensor(
                    src_shape.reshape(-1), device=device, dtype=torch.float32).unsqueeze(0)

                jm = chain_joint_map[cname]

                # Previous angles for temporal smoothing
                prev = prev_angles.get(cname)

                # NN forward pass
                angles_norm = model(src_tensor, jm['descriptor'], prev)

                n = jm['n_joints']
                lo_np = jm['lo']
                hi_np = jm['hi']
                mid_np = (hi_np + lo_np) / 2
                half_range_np = np.maximum((hi_np - lo_np) / 2, 1e-6)

                angles_np = angles_norm[0, :n].cpu().numpy() * half_range_np + mid_np
                angles_np = np.clip(angles_np, lo_np, hi_np).astype(np.float32)

                # LM refinement using MuJoCo FK (hybrid mode)
                if args.lm_steps > 0:
                    # NN 출력을 qpos에 세팅
                    for k, qpa in enumerate(jm['qpa']):
                        qpos[qpa] = angles_np[k]
                    # source body positions를 target으로 LM refine
                    target_bids = [ch['body_ids'][bi] for bi, _ in mapped]
                    target_pos = np.array([np.asarray(scaled[hb][0]) for _, hb in mapped])
                    chain_dof = jm['dofadr']

                    for _step in range(args.lm_steps):
                        mj_data.qpos[:] = qpos
                        mj.mj_forward(mj_model, mj_data)
                        # residual: FK position - target position
                        res = []
                        for ti, bid in enumerate(target_bids):
                            res.extend((mj_data.xpos[bid] - target_pos[ti]).tolist())
                        res = np.array(res)
                        if np.linalg.norm(res) < 1e-4:
                            break
                        # Jacobian
                        n_res = len(res)
                        J = np.zeros((n_res, n))
                        jacp_buf = np.zeros((3, mj_model.nv))
                        for ti, bid in enumerate(target_bids):
                            jacp_buf[:] = 0
                            mj.mj_jacBody(mj_model, mj_data, jacp_buf, None, bid)
                            for k, da in enumerate(chain_dof):
                                if 0 <= da < mj_model.nv:
                                    J[ti*3:(ti+1)*3, k] = jacp_buf[:, da]
                        # LM step: dx = -(J^T J + lambda I)^-1 J^T r
                        lam = 1e-3
                        JtJ = J.T @ J + lam * np.eye(n)
                        dx = -np.linalg.solve(JtJ, J.T @ res)
                        # Update and clamp
                        for k, qpa in enumerate(jm['qpa']):
                            qpos[qpa] = np.clip(qpos[qpa] + dx[k], lo_np[k], hi_np[k])
                    # Read back refined angles
                    angles_np = np.array([qpos[a] for a in jm['qpa']], dtype=np.float32)

                # Store for next frame
                a_norm = np.clip((angles_np - mid_np) / half_range_np, -1, 1)
                padded = np.zeros(MAX_JOINTS, dtype=np.float32)
                padded[:n] = a_norm
                prev_angles[cname] = torch.tensor(padded, device=device).unsqueeze(0)

                # Apply to qpos
                for k, qpa in enumerate(jm['qpa']):
                    qpos[qpa] = np.clip(angles_np[k], lo_np[k], hi_np[k])

        # FK
        mj_data.qpos[:] = qpos
        mj.mj_forward(mj_model, mj_data)

        frame_count += 1

        # Visualize
        if viewer is not None:
            viewer.step(
                root_pos=qpos[:3] if has_freejoint else np.zeros(3),
                root_rot=qpos[3:7] if has_freejoint else np.array([1, 0, 0, 0]),
                dof_pos=qpos[7:] if has_freejoint else qpos,
                human_motion_data=scaled,
                rate_limit=True,
                follow_camera=True,
            )

        # Next frame
        if args.loop:
            i = (i + 1) % len(frames)
        else:
            i += 1
            if i >= len(frames):
                break

    elapsed = time.time() - t0
    print(f"\n{frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.0f} fps)")

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()