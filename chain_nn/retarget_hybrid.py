"""
Hybrid Retarget: NN Warm Start + Chain LM
==========================================
NN이 joint angles 초기값 예측 → chain_motion_retarget의 LM에 warm start 제공.
LM iteration이 줄어들어 빠르면서도 Step 1(root yaw) + waist 포함.

Usage:
  python chain_nn/retarget_hybrid.py \
    --bvh_file <input.bvh> --format lafan1 --robot unitree_g1 \
    --ckpt chain_nn/checkpoints/direct_ik_best.pt \
    --meta chain_nn/data/ik_data.pkl \
    --lm_steps 5 --visualize --loop
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

from common.chain_shape import normalize_chain_shape, K_POINTS
from chain_nn.direct_model import DirectChainIK, MAX_JOINTS


def main():
    parser = argparse.ArgumentParser(description="Hybrid NN + Chain LM retarget")
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--format", default="lafan1")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--meta", default="chain_nn/data/ik_data.pkl")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--lm_steps", type=int, default=5,
                        help="Max LM iterations per chain (default 5, original uses 30)")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Package setup ──
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
    from general_motion_retargeting.chain_motion_retarget import ChainMotionRetargeting
    import mujoco as mj

    # ── Load NN model ──
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt['config']
    nn_model = DirectChainIK(
        shape_dim=cfg['shape_dim'], desc_dim=cfg['desc_dim'],
        max_joints=cfg['max_joints'], hidden_dim=cfg['hidden_dim'],
    ).to(device)
    nn_model.load_state_dict(ckpt['model'])
    nn_model.eval()
    print(f"NN model loaded (epoch {ckpt['epoch']}, val={ckpt['val_loss']:.4f})")

    # ── Load chain meta ──
    with open(args.meta, 'rb') as f:
        meta_data = pickle.load(f)
    chain_meta = meta_data['chain_meta']

    # ── Load BVH ──
    frames, height, hierarchy = load_bvh_file(args.bvh_file, format=args.format)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames")

    # ── Init Chain LM retargeter ──
    src_human = f"bvh_{args.format}" if args.format != "robot" else "robot"
    retargeter = ChainMotionRetargeting(
        src_human=src_human, tgt_robot=args.robot,
        actual_human_height=height, verbose=False,
    )
    retargeter._bone_hierarchy = hierarchy

    # Override LM max_nfev (fewer iterations since NN gives good init)
    # We'll monkey-patch after first frame init

    # ── Chain → joint mapping for NN ──
    # (initialized after first retarget call sets up mapping)
    nn_chain_map = None  # lazy init

    def init_nn_chain_map():
        nonlocal nn_chain_map
        nn_chain_map = {}
        has_free = retargeter.has_freejoint
        for ch in retargeter.chains:
            cname = ch['name']
            if args.robot not in chain_meta or cname not in chain_meta[args.robot]:
                continue
            meta = chain_meta[args.robot][cname]
            body_set = set(ch['body_ids'])
            qpa_list = []
            start_j = 1 if has_free else 0
            for ji in range(start_j, retargeter.model.njnt):
                if retargeter.model.jnt_type[ji] != mj.mjtJoint.mjJNT_HINGE:
                    continue
                if int(retargeter.model.jnt_bodyid[ji]) in body_set:
                    qpa_list.append(int(retargeter.model.jnt_qposadr[ji]))

            nn_chain_map[cname] = {
                'qpa': qpa_list,
                'lo': meta['lo'],
                'hi': meta['hi'],
                'n_joints': meta['n_joints'],
                'descriptor': torch.tensor(meta['descriptor'], device=device).unsqueeze(0),
            }

    # ── Viewer ──
    viewer = None
    if args.visualize:
        from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
        viewer = RobotMotionViewer(
            robot_type=args.robot, motion_fps=args.motion_fps, transparent_robot=0,
        )

    # ── Retarget loop ──
    from tqdm import tqdm
    prev_nn_angles = {}  # chain_name → (1, MAX_JOINTS) tensor
    t0 = time.time()
    i = 0
    frame_count = 0
    total = len(frames)
    pbar = tqdm(total=total, desc="Retargeting")

    while True:
        human_data = frames[i]

        # ── NN predict → set as prev_qpos (warm start for LM) ──
        if nn_chain_map is not None:
            # Build NN-predicted qpos
            nn_qpos = retargeter.prev_qpos.copy() if retargeter.prev_qpos is not None \
                else retargeter.model.qpos0.copy()

            with torch.no_grad():
                # Scale/ground human data like retargeter does
                hd = {k: [np.asarray(v[0]), np.asarray(v[1])] for k, v in human_data.items()}
                scaled_hd = retargeter._scale(hd)
                scaled_hd = retargeter._ground(scaled_hd)

                for ch in retargeter.chains:
                    cname = ch['name']
                    if cname not in nn_chain_map:
                        continue
                    jm = nn_chain_map[cname]

                    mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                              if hb and hb in scaled_hd]
                    if len(mapped) < 2:
                        continue

                    src_pos = np.array([np.asarray(scaled_hd[hb][0]) for _, hb in mapped])
                    src_shape, _, _ = normalize_chain_shape(src_pos)
                    src_t = torch.tensor(
                        src_shape.reshape(-1), device=device, dtype=torch.float32
                    ).unsqueeze(0)

                    prev = prev_nn_angles.get(cname)
                    angles_norm = nn_model(src_t, jm['descriptor'], prev)
                    prev_nn_angles[cname] = angles_norm.clone()

                    n = jm['n_joints']
                    lo, hi = jm['lo'], jm['hi']
                    mid = (hi + lo) / 2
                    hr = np.maximum((hi - lo) / 2, 1e-6)
                    angles = angles_norm[0, :n].cpu().numpy() * hr + mid
                    angles = np.clip(angles, lo, hi)

                    for k, qpa in enumerate(jm['qpa']):
                        nn_qpos[qpa] = angles[k]

            # NN 출력을 LM의 warm start로 주입
            retargeter.prev_qpos = nn_qpos

        # ── Chain LM retarget (NN warm start 덕분에 빠르게 수렴) ──
        qpos = retargeter.retarget(human_data)

        # Lazy init NN chain map after first retarget (mapping이 설정된 후)
        if nn_chain_map is None:
            init_nn_chain_map()
            print(f"  NN chains: {list(nn_chain_map.keys())}")

        frame_count += 1
        pbar.update(1)

        # Visualize
        if viewer is not None:
            has_free = retargeter.has_freejoint
            viewer.step(
                root_pos=qpos[:3] if has_free else np.zeros(3),
                root_rot=qpos[3:7] if has_free else np.array([1, 0, 0, 0]),
                dof_pos=qpos[7:] if has_free else qpos,
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=True,
                follow_camera=True,
            )

        if args.loop:
            i = (i + 1) % len(frames)
        else:
            i += 1
            if i >= len(frames):
                break

    pbar.close()
    elapsed = time.time() - t0
    print(f"\n{frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()