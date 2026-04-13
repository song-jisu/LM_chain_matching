"""
Chain Flow Retargeting: Any-to-Any Inference Pipeline
======================================================
Trained chain flow model을 사용한 motion retargeting.

Pipeline:
  1. Source motion → chain decomposition → normalized shapes
  2. Temporal windowing (W=32)
  3. VQ-VAE encode → Flow matching (conditioned on target) → VQ-VAE decode
     OR Direct flow → target shapes
  4. Denormalize → Chain LM (MuJoCo FK + Jacobian) → joint angles

Usage:
  python chain_flow/retarget.py \
    --bvh_file <input.bvh> --format lafan1 \
    --robot unitree_g1 \
    --flow_ckpt chain_flow/checkpoints/direct_flow_best.pt \
    [--vqvae_ckpt chain_flow/checkpoints/vqvae_best.pt]
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chain_flow.chain_shape import (
    normalize_chain_shape, denormalize_chain_shape,
    extract_chain_descriptor, descriptor_to_vector,
    K_POINTS, DESCRIPTOR_DIM,
)
from chain_flow.vqvae import ChainVQVAE
from chain_flow.flow_model import (
    ChainFlowModel, DirectChainFlowModel, flow_sample,
)
from chain_flow.dataset import WINDOW_SIZE, SHAPE_DIM


class ChainFlowRetargeter:
    """Chain Flow를 사용한 any-to-any motion retargeter.

    기존 ChainMotionRetargeting의 LM solver를 flow model로 대체하되,
    최종 joint angle 계산은 Chain LM을 사용.
    """

    def __init__(
        self,
        tgt_robot,
        flow_ckpt,
        vqvae_ckpt=None,
        n_flow_steps=20,
        device=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tgt_robot = tgt_robot
        self.n_flow_steps = n_flow_steps

        # Load flow model
        flow_data = torch.load(flow_ckpt, map_location=self.device, weights_only=False)
        flow_cfg = flow_data['config']
        self.use_vqvae = flow_data.get('use_vqvae', False)

        if self.use_vqvae:
            # Load VQ-VAE
            assert vqvae_ckpt is not None, "VQ-VAE checkpoint required when use_vqvae=True"
            vq_data = torch.load(vqvae_ckpt, map_location=self.device, weights_only=False)
            vq_cfg = vq_data['config']
            self.vqvae = ChainVQVAE(
                shape_dim=vq_cfg['shape_dim'],
                window_size=vq_cfg['window_size'],
                hidden_dim=vq_cfg['hidden_dim'],
                latent_dim=vq_cfg['latent_dim'],
                n_embeddings=vq_cfg['codebook_size'],
            ).to(self.device)
            self.vqvae.load_state_dict(vq_data['model'])
            self.vqvae.eval()

            self.flow = ChainFlowModel(
                latent_dim=flow_cfg['latent_dim'],
                latent_seq_len=flow_cfg['latent_seq_len'],
                cond_dim=flow_cfg['cond_dim'],
                hidden_dim=flow_cfg['hidden_dim'],
                n_layers=flow_cfg['n_layers'],
            ).to(self.device)
        else:
            self.vqvae = None
            self.flow = DirectChainFlowModel(
                shape_dim=flow_cfg['shape_dim'],
                window_size=flow_cfg['window_size'],
                cond_dim=flow_cfg['cond_dim'],
                hidden_dim=flow_cfg['hidden_dim'],
                n_layers=flow_cfg['n_layers'],
            ).to(self.device)

        self.flow.load_state_dict(flow_data['model'])
        self.flow.eval()
        print(f"[ChainFlow] Loaded flow model (use_vqvae={self.use_vqvae})")

        # Init chain LM retargeter (for final IK)
        import types
        pkg = types.ModuleType('general_motion_retargeting')
        pkg.__path__ = [str(Path(__file__).parent.parent / 'general_motion_retargeting')]
        if 'general_motion_retargeting' not in sys.modules:
            sys.modules['general_motion_retargeting'] = pkg
            from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT
            pkg.ROBOT_XML_DICT = ROBOT_XML_DICT
            pkg.IK_CONFIG_DICT = IK_CONFIG_DICT
            pkg.ROBOT_BASE_DICT = ROBOT_BASE_DICT

        from general_motion_retargeting.chain_motion_retarget import ChainMotionRetargeting
        self.lm_retargeter = None  # lazy init (needs first frame for CLI matching)
        self._ChainMotionRetargeting = ChainMotionRetargeting

        # State
        self.window_buffer = {}  # chain_name → list of (src_shape, src_start, src_length)
        self.chain_descriptors = {}  # chain_name → descriptor tensor

    def _init_lm_retargeter(self, human_height, bone_hierarchy, src_format="lafan1"):
        """LM retargeter 초기화 (첫 프레임에서 호출)."""
        src_human = f"bvh_{src_format}" if src_format != "robot" else "robot"
        self.lm_retargeter = self._ChainMotionRetargeting(
            src_human=src_human,
            tgt_robot=self.tgt_robot,
            actual_human_height=human_height,
            verbose=False,
        )
        self.lm_retargeter._bone_hierarchy = bone_hierarchy

        # Chain descriptors 미리 계산
        for ch in self.lm_retargeter.chains:
            desc = extract_chain_descriptor(ch, self.lm_retargeter.model)
            desc_vec = descriptor_to_vector(desc)
            self.chain_descriptors[ch['name']] = torch.tensor(
                desc_vec, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def _flow_retarget_window(self, src_shapes, chain_name):
        """Flow model로 source window → target window 변환.

        Args:
            src_shapes: (W, K*3) numpy array
            chain_name: str
        Returns:
            tgt_shapes: (W, K*3) numpy array
        """
        src = torch.tensor(src_shapes, device=self.device).unsqueeze(0)  # (1, W, D)
        cond = self.chain_descriptors[chain_name]  # (1, cond_dim)

        if self.vqvae is not None:
            # Encode source
            src_h = src.permute(0, 2, 1)
            src_h = self.vqvae.encoder(src_h).permute(0, 2, 1)

            # Flow
            tgt_h = flow_sample(self.flow, src_h, cond, n_steps=self.n_flow_steps)

            # Decode
            z_q, _, _, _ = self.vqvae.vq(tgt_h)
            tgt = self.vqvae.decode(z_q)
        else:
            # Direct flow
            tgt = flow_sample(self.flow, src, cond, n_steps=self.n_flow_steps)

        return tgt.squeeze(0).cpu().numpy()  # (W, D)

    def retarget_sequence(self, frames, human_height=1.75, bone_hierarchy=None,
                          src_format="lafan1"):
        """전체 프레임 시퀀스를 retarget.

        Args:
            frames: list of human_data dicts (from load_bvh_file)
            human_height: source height
            bone_hierarchy: BVH hierarchy info
            src_format: BVH format ('lafan1', 'nokov', 'robot')
        Returns:
            qpos_list: list of qpos arrays
        """
        if self.lm_retargeter is None:
            self._init_lm_retargeter(human_height, bone_hierarchy, src_format)

        # Phase 1: LM retarget으로 chain mapping 설정 + source shapes 수집
        print("[ChainFlow] Phase 1: Collecting source chain shapes...")
        all_chain_data = {}  # chain_name → list of {src_shape, src_start, src_length}

        for fi, human_data in enumerate(frames):
            try:
                qpos = self.lm_retargeter.retarget(human_data)
            except Exception:
                continue

            scaled = self.lm_retargeter.scaled_human_data
            if scaled is None:
                continue

            for ch in self.lm_retargeter.chains:
                mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                          if hb and hb in scaled]
                if len(mapped) < 2:
                    continue

                cname = ch['name']
                if cname not in all_chain_data:
                    all_chain_data[cname] = []

                src_pos = np.array([np.asarray(scaled[hb][0]) for _, hb in mapped])
                src_shape, src_start, src_length = normalize_chain_shape(src_pos)
                all_chain_data[cname].append({
                    'shape': src_shape.reshape(-1).astype(np.float32),
                    'start': src_start,
                    'length': src_length,
                })

        # Phase 2: Flow retarget (windowed)
        print("[ChainFlow] Phase 2: Flow-based retargeting...")
        tgt_chain_shapes = {}  # chain_name → list of (K*3,) per frame

        for cname, frame_data in all_chain_data.items():
            if cname not in self.chain_descriptors:
                continue

            n = len(frame_data)
            src_array = np.stack([d['shape'] for d in frame_data])  # (N, K*3)

            # Windowed flow
            tgt_array = np.zeros_like(src_array)
            counts = np.zeros(n)

            w = WINDOW_SIZE
            for start in range(0, max(1, n - w + 1), w // 2):
                end = min(start + w, n)
                if end - start < w:
                    start = max(0, end - w)

                window_src = src_array[start:start+w]
                if len(window_src) < w:
                    # Pad with last frame
                    pad = np.tile(window_src[-1:], (w - len(window_src), 1))
                    window_src = np.concatenate([window_src, pad], axis=0)

                window_tgt = self._flow_retarget_window(window_src, cname)

                # Overlap-add (simple average)
                actual_len = min(w, end - start)
                tgt_array[start:start+actual_len] += window_tgt[:actual_len]
                counts[start:start+actual_len] += 1

            # Average overlapping regions
            counts = np.maximum(counts, 1)
            tgt_array = tgt_array / counts[:, None]

            tgt_chain_shapes[cname] = tgt_array

        # Phase 3: Denormalize + Chain LM for joint angles
        print("[ChainFlow] Phase 3: Chain LM for joint angles...")
        n_frames = len(frames)
        qpos_list = []

        # Reset LM retargeter state
        self.lm_retargeter.prev_qpos = None
        self.lm_retargeter._rest_computed = False
        if hasattr(self.lm_retargeter, '_fwd_rotation'):
            delattr(self.lm_retargeter, '_fwd_rotation')

        for fi, human_data in enumerate(frames):
            # Flow가 생성한 target shapes로 human_data를 재구성
            # (target chain의 denormalized positions을 source positions 위치에 재배치)
            modified_data = dict(human_data)

            for ch in self.lm_retargeter.chains:
                cname = ch['name']
                if cname not in tgt_chain_shapes or cname not in all_chain_data:
                    continue
                if fi >= len(all_chain_data[cname]):
                    continue

                # Flow output (normalized shape)
                tgt_flat = tgt_chain_shapes[cname][fi]
                tgt_shape = tgt_flat.reshape(K_POINTS, 3)

                # Denormalize using source chain's start + length
                # (but scaled to target chain's expected length)
                src_info = all_chain_data[cname][fi]
                tgt_positions = denormalize_chain_shape(
                    tgt_shape, src_info['start'], src_info['length']
                )

                # chain의 mapped body에 flow output positions 할당
                mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                          if hb and hb in modified_data]
                if len(mapped) < 2:
                    continue

                # Flow의 K sample points를 mapped body 수에 맞게 리샘플
                n_mapped = len(mapped)
                indices = np.linspace(0, K_POINTS - 1, n_mapped).astype(int)

                for mi, (bi, hb) in enumerate(mapped):
                    pos = tgt_positions[indices[mi]]
                    modified_data[hb] = [pos, modified_data[hb][1]]

            try:
                qpos = self.lm_retargeter.retarget(modified_data)
                qpos_list.append(qpos)
            except Exception:
                if qpos_list:
                    qpos_list.append(qpos_list[-1].copy())
                else:
                    import mujoco as mj
                    qpos_list.append(self.lm_retargeter.model.qpos0.copy())

        return qpos_list


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain Flow retargeting")
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--format", default="lafan1")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--vqvae_ckpt", default=None)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--visualize", action="store_true", help="MuJoCo viewer로 결과 시각화")
    parser.add_argument("--motion_fps", type=int, default=30)
    args = parser.parse_args()

    # Bypass __init__.py
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

    # Load BVH
    frames, height, hierarchy = load_bvh_file(args.bvh_file, format=args.format)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames from {args.bvh_file}")

    # Retarget
    retargeter = ChainFlowRetargeter(
        tgt_robot=args.robot,
        flow_ckpt=args.flow_ckpt,
        vqvae_ckpt=args.vqvae_ckpt,
        n_flow_steps=args.n_steps,
    )

    t0 = time.time()
    qpos_list = retargeter.retarget_sequence(
        frames, height, hierarchy, src_format=args.format)
    elapsed = time.time() - t0

    print(f"\nRetargeted {len(qpos_list)} frames in {elapsed:.1f}s "
          f"({len(qpos_list)/elapsed:.1f} fps)")

    if args.save_path:
        import pickle
        with open(args.save_path, 'wb') as f:
            pickle.dump({
                'qpos': [q.tolist() for q in qpos_list],
                'fps': 30,
            }, f)
        print(f"Saved to {args.save_path}")

    if args.visualize:
        from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.motion_fps,
            transparent_robot=0,
        )
        print(f"Visualizing {len(qpos_list)} frames (loop)...")
        i = 0
        while True:
            qpos = qpos_list[i]
            has_free = retargeter.lm_retargeter.has_freejoint
            viewer.step(
                root_pos=qpos[:3] if has_free else np.zeros(3),
                root_rot=qpos[3:7] if has_free else np.array([1.0, 0.0, 0.0, 0.0]),
                dof_pos=qpos[7:] if has_free else qpos,
                human_motion_data=retargeter.lm_retargeter.scaled_human_data,
                rate_limit=True,
                follow_camera=True,
            )
            i = (i + 1) % len(qpos_list)
        viewer.close()