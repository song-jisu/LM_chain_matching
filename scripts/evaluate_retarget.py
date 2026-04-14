"""
GMR 논문 메트릭 기반 Motion Retarget 성능 평가
================================================
Metrics (from "Retargeting Matters", ICRA 2026, arXiv 2510.02252):
  E_g-mpbpe: Global Mean Position Body Part Error
  E_mpbpe:   Root-relative Mean Position Body Part Error
"""

import argparse
import pathlib
import os
import sys
import csv
import time
import numpy as np
from tqdm import tqdm

# Add parent to path — bypass __init__.py to avoid glfw/torch deps
HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

# Prevent __init__.py from running (it imports viewer/torch)
import types
_pkg = types.ModuleType('general_motion_retargeting')
_pkg.__path__ = [str(HERE / 'general_motion_retargeting')]
sys.modules['general_motion_retargeting'] = _pkg

# Import params first (needed by motion_retarget and chain_motion_retarget)
from general_motion_retargeting.params import IK_CONFIG_ROOT, ASSET_ROOT, ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT
_pkg.IK_CONFIG_ROOT = IK_CONFIG_ROOT
_pkg.ASSET_ROOT = ASSET_ROOT
_pkg.ROBOT_XML_DICT = ROBOT_XML_DICT
_pkg.IK_CONFIG_DICT = IK_CONFIG_DICT
_pkg.ROBOT_BASE_DICT = ROBOT_BASE_DICT

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.chain_motion_retarget import ChainMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file
import mujoco as mj


def get_body_mapping(retargeter, method):
    """retargeter에서 (source_body_name, target_body_id) 매핑 리스트 추출."""
    pairs = []  # [(src_body_name, tgt_body_id, chain_name)]

    if method == "chain":
        for ch in retargeter.chains:
            for bi, hb in enumerate(ch['human_bodies']):
                if hb:
                    pairs.append((hb, ch['body_ids'][bi], ch['name']))
    else:
        # GMR IK: ik_match_table1/2에서 매핑 추출
        # ik_match_table: {robot_frame_name: [human_body_name, pos_w, rot_w, ...]}
        model = retargeter.configuration.model
        for table in [retargeter.ik_match_table1, retargeter.ik_match_table2]:
            for frame_name, entry in table.items():
                human_body = entry[0]
                pos_weight = entry[1]
                if pos_weight > 0:
                    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, frame_name)
                    if bid >= 0:
                        pairs.append((human_body, bid, frame_name))
    return pairs


def get_root_info(retargeter, method):
    """root body name과 target root body id 반환."""
    if method == "chain":
        root_name = retargeter.human_root_name
        root_bid = retargeter.body_name2id.get('pelvis', 1)
        return root_name, root_bid
    else:
        root_name = retargeter.human_root_name
        model = retargeter.configuration.model
        robot_root = retargeter.robot_root_name
        root_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot_root)
        return root_name, root_bid


def get_target_xpos(retargeter, method):
    """retarget 후 target body positions 반환."""
    if method == "chain":
        return retargeter.data.xpos
    else:
        return retargeter.configuration.data.xpos


def compute_robot_height(retargeter, method):
    """rest pose에서 로봇 높이 계산."""
    if method == "chain":
        model = retargeter.model
        data = mj.MjData(model)
        data.qpos[:] = model.qpos0
        if retargeter.has_freejoint:
            data.qpos[2] = 0.793
            data.qpos[3] = 1.0
        mj.mj_forward(model, data)
        all_z = [data.xpos[i][2] for i in range(model.nbody)]
    else:
        model = retargeter.configuration.model
        data = mj.MjData(model)
        data.qpos[:] = model.qpos0
        mj.mj_forward(model, data)
        all_z = [data.xpos[i][2] for i in range(model.nbody)]
    return max(all_z) - min(all_z)


def evaluate_single(bvh_file, format_type, robot, method, max_frames=None):
    """단일 BVH 파일에 대해 retarget 수행 및 메트릭 수집."""

    # Load BVH
    frames, human_height, bone_hierarchy = load_bvh_file(bvh_file, format=format_type)
    if max_frames:
        frames = frames[:max_frames]

    # Init retargeter
    if method == "chain":
        retargeter = ChainMotionRetargeting(
            src_human=f"bvh_{format_type}" if format_type != "robot" else "robot",
            tgt_robot=robot,
            actual_human_height=human_height,
            verbose=False,
        )
        retargeter._bone_hierarchy = bone_hierarchy
    else:
        retargeter = GMR(
            src_human=f"bvh_{format_type}",
            tgt_robot=robot,
            actual_human_height=human_height,
            solver="daqp",
            verbose=False,
        )

    # Metrics storage
    frame_g_mpbpe = []   # per-frame global error
    frame_mpbpe = []     # per-frame root-relative error
    chain_errors = {}    # chain_name -> {'global': [], 'relative': []}

    mapping = None
    root_name = None
    root_bid = None

    n_errors = 0
    for fi, human_data in enumerate(tqdm(frames, desc=os.path.basename(bvh_file), leave=False)):
        try:
            qpos = retargeter.retarget(human_data)
        except Exception:
            n_errors += 1
            continue

        # Lazy init mapping (after first retarget sets up CLI matching etc.)
        if mapping is None:
            mapping = get_body_mapping(retargeter, method)
            root_name, root_bid = get_root_info(retargeter, method)
            for _, _, cname in mapping:
                if cname not in chain_errors:
                    chain_errors[cname] = {'global': [], 'relative': []}

        scaled = retargeter.scaled_human_data
        xpos = get_target_xpos(retargeter, method)

        if root_name not in scaled:
            continue

        src_root = np.asarray(scaled[root_name][0])
        tgt_root = xpos[root_bid].copy()

        g_errors = []
        r_errors = []

        for src_body, tgt_bid, cname in mapping:
            if src_body not in scaled:
                continue

            src_pos = np.asarray(scaled[src_body][0])
            tgt_pos = xpos[tgt_bid].copy()

            # E_g-mpbpe: global position error
            g_err = np.linalg.norm(src_pos - tgt_pos)
            g_errors.append(g_err)

            # E_mpbpe: root-relative position error
            src_rel = src_pos - src_root
            tgt_rel = tgt_pos - tgt_root
            r_err = np.linalg.norm(src_rel - tgt_rel)
            r_errors.append(r_err)

            chain_errors[cname]['global'].append(g_err)
            chain_errors[cname]['relative'].append(r_err)

        if g_errors:
            frame_g_mpbpe.append(np.mean(g_errors))
            frame_mpbpe.append(np.mean(r_errors))

    # Aggregate
    result = {
        'bvh_file': os.path.basename(bvh_file),
        'robot': robot,
        'method': method,
        'n_frames': len(frame_g_mpbpe),
        'n_mapped_bodies': len(mapping) if mapping else 0,
        'n_errors': n_errors,
    }

    if frame_g_mpbpe:
        g = np.array(frame_g_mpbpe)
        r = np.array(frame_mpbpe)
        result['E_g_mpbpe_mean'] = float(np.mean(g))
        result['E_g_mpbpe_std'] = float(np.std(g))
        result['E_g_mpbpe_median'] = float(np.median(g))
        result['E_g_mpbpe_max'] = float(np.max(g))
        result['E_mpbpe_mean'] = float(np.mean(r))
        result['E_mpbpe_std'] = float(np.std(r))
        result['E_mpbpe_median'] = float(np.median(r))
        result['E_mpbpe_max'] = float(np.max(r))

        # Per-chain
        result['per_chain'] = {}
        for cname, errs in chain_errors.items():
            if errs['global']:
                result['per_chain'][cname] = {
                    'E_g_mean': float(np.mean(errs['global'])),
                    'E_r_mean': float(np.mean(errs['relative'])),
                }
    else:
        result['E_g_mpbpe_mean'] = float('nan')
        result['E_mpbpe_mean'] = float('nan')

    return result


def print_results(results, robot_height=None):
    """결과 출력."""
    if not results:
        print("No results.")
        return

    print(f"\n{'='*70}")
    print(f"  Evaluation: {results[0]['robot']}, method={results[0]['method']}")
    print(f"{'='*70}")

    total_frames = sum(r['n_frames'] for r in results)
    print(f"  BVH files: {len(results)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Mapped bodies per frame: {results[0]['n_mapped_bodies']}")

    # Aggregate across files
    all_g = [r['E_g_mpbpe_mean'] for r in results if not np.isnan(r.get('E_g_mpbpe_mean', float('nan')))]
    all_r = [r['E_mpbpe_mean'] for r in results if not np.isnan(r.get('E_mpbpe_mean', float('nan')))]

    print(f"\n  {'Metric':<20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-'*60}")

    if all_g:
        g = np.array(all_g)
        print(f"  {'E_g-mpbpe (m)':<20s} {np.mean(g):10.4f} {np.std(g):10.4f} {np.min(g):10.4f} {np.max(g):10.4f}")

    if all_r:
        r = np.array(all_r)
        print(f"  {'E_mpbpe (m)':<20s} {np.mean(r):10.4f} {np.std(r):10.4f} {np.min(r):10.4f} {np.max(r):10.4f}")

    if robot_height and all_r:
        r_norm = np.array(all_r) / robot_height
        print(f"  {'E_mpbpe (norm)':<20s} {np.mean(r_norm):10.4f} {np.std(r_norm):10.4f} {np.min(r_norm):10.4f} {np.max(r_norm):10.4f}")

    # Per-chain breakdown (from first result that has it)
    for res in results:
        if 'per_chain' in res and res['per_chain']:
            print(f"\n  Per-chain breakdown ({res['bvh_file']}):")
            print(f"  {'Chain':<30s} {'E_mpbpe':>10s} {'E_g-mpbpe':>10s}")
            print(f"  {'-'*50}")
            for cname, errs in sorted(res['per_chain'].items()):
                print(f"  {cname:<30s} {errs['E_r_mean']:10.4f} {errs['E_g_mean']:10.4f}")
            break

    # Per-file table
    if len(results) > 1:
        print(f"\n  Per-file results:")
        print(f"  {'File':<35s} {'Frames':>7s} {'E_g-mpbpe':>10s} {'E_mpbpe':>10s}")
        print(f"  {'-'*62}")
        for r in results:
            print(f"  {r['bvh_file']:<35s} {r['n_frames']:7d} {r.get('E_g_mpbpe_mean', float('nan')):10.4f} {r.get('E_mpbpe_mean', float('nan')):10.4f}")

    print(f"{'='*70}\n")


def save_csv(results, output_path):
    """결과를 CSV로 저장."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fields = ['bvh_file', 'robot', 'method', 'n_frames', 'n_mapped_bodies',
              'E_g_mpbpe_mean', 'E_g_mpbpe_std', 'E_g_mpbpe_median', 'E_g_mpbpe_max',
              'E_mpbpe_mean', 'E_mpbpe_std', 'E_mpbpe_median', 'E_mpbpe_max']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved CSV to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMR metric evaluation for motion retargeting")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bvh_file", type=str, help="Single BVH file to evaluate")
    group.add_argument("--bvh_dir", type=str, help="Directory of BVH files to evaluate")

    parser.add_argument("--format", choices=["lafan1", "nokov", "robot"], default="lafan1")
    parser.add_argument("--robot", choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1",
                                            "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos"],
                        default="unitree_g1")
    parser.add_argument("--method", choices=["ik", "chain"], default="chain")
    parser.add_argument("--max_files", type=int, default=None, help="Max BVH files to evaluate")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per file")
    parser.add_argument("--output_csv", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--normalize", action="store_true", help="Normalize errors by robot height")

    args = parser.parse_args()

    # Collect BVH files
    if args.bvh_file:
        bvh_files = [args.bvh_file]
    else:
        bvh_files = sorted([
            os.path.join(args.bvh_dir, f)
            for f in os.listdir(args.bvh_dir)
            if f.endswith('.bvh')
        ])
        if args.max_files:
            bvh_files = bvh_files[:args.max_files]

    print(f"\nEvaluating {len(bvh_files)} BVH file(s)")
    print(f"  Robot: {args.robot}")
    print(f"  Method: {args.method}")
    print(f"  Format: {args.format}")
    if args.max_frames:
        print(f"  Max frames: {args.max_frames}")

    # Evaluate
    results = []
    t0 = time.time()

    for bvh_path in tqdm(bvh_files, desc="Files"):
        try:
            res = evaluate_single(bvh_path, args.format, args.robot, args.method, args.max_frames)
            results.append(res)
        except Exception as e:
            print(f"\n  ERROR on {os.path.basename(bvh_path)}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Robot height for normalization
    robot_height = None
    if args.normalize and results:
        # Quick height computation
        frames, h, bh = load_bvh_file(bvh_files[0], format=args.format)
        if args.method == "chain":
            ret = ChainMotionRetargeting(
                src_human=f"bvh_{args.format}" if args.format != "robot" else "robot",
                tgt_robot=args.robot, actual_human_height=h, verbose=False)
            robot_height = compute_robot_height(ret, args.method)
        else:
            ret = GMR(src_human=f"bvh_{args.format}", tgt_robot=args.robot, actual_human_height=h)
            robot_height = compute_robot_height(ret, args.method)
        print(f"  Robot height: {robot_height:.3f}m")

    # Output
    print_results(results, robot_height)

    if args.output_csv:
        save_csv(results, args.output_csv)