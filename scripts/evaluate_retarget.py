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
from general_motion_retargeting.auto_chain_retarget import AutoChainRetargeting
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
    if method in ("chain", "auto"):
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


class CommonMetrics:
    """모든 방법에 같은 기준을 적용하는 평가 지표.

    정답 대응은 GMR IK 설정(bvh_lafan1_to_<robot>.json)의 위치 가중치>0 바디 대응을 쓴다.
    이 대응은 평가에만 쓰이며 auto 방법의 리타게팅에는 쓰이지 않는다.
      E_dir : 사지 분절(허벅지·정강이·위팔·아래팔) 방향 각도 오차 [deg]
      E_cmp : 공통 균일 스케일(사지 길이 비) 기준 루트 상대 위치 오차 [m]
      flip  : 무릎·팔꿈치 굽힘 법선이 소스와 반대(dot<0)인 프레임 비율
    """

    def __init__(self, robot, src_format, model, human_rest):
        cfg_path = IK_CONFIG_DICT.get(f"bvh_{src_format}", {}).get(robot)
        self.ok = cfg_path is not None and os.path.exists(cfg_path)
        if not self.ok:
            return
        import json
        cfg = json.load(open(cfg_path))
        lm = {}
        for t in ("ik_match_table1", "ik_match_table2"):
            for rname, e in cfg.get(t, {}).items():
                if e[1] > 0 or rname == cfg["robot_root_name"]:
                    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, rname)
                    if bid >= 0:
                        lm[bid] = e[0]
        self.lm = lm
        self.root_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg["robot_root_name"])
        self.root_h = cfg["human_root_name"]

        def anc(b):
            b = int(model.body_parentid[b])
            while b > 0 and b not in lm:
                b = int(model.body_parentid[b])
            return b if b in lm else None
        parent = {b: anc(b) for b in lm}
        kids = {b: [c for c in lm if parent[c] == b] for b in lm}
        # 사지 분절: 부모 landmark가 허브(자식 landmark 2개 이상)가 아닌 분절
        self.segs = [(parent[b], b) for b in lm if parent[b] is not None and len(kids[parent[b]]) == 1]
        self.trip = [(parent[b], b, kids[b][0]) for b in lm
                     if parent[b] is not None and len(kids[parent[b]]) == 1 and len(kids[b]) == 1]

        d = mj.MjData(model)
        mj.mj_forward(model, d)
        rl = sum(np.linalg.norm(d.xpos[b] - d.xpos[a]) for a, b in self.segs)
        hl = sum(np.linalg.norm(np.asarray(human_rest[lm[b]][0]) - np.asarray(human_rest[lm[a]][0]))
                 for a, b in self.segs)
        self.scale = rl / max(hl, 1e-9)
        self.dvecs = []

    def dynamic_error(self):
        """랜드마크별 시간 평균 오프셋(바디 원점 위치 관례 차이)을 뺀 위치 오차."""
        D = np.array(self.dvecs)
        return float(np.mean(np.linalg.norm(D - D.mean(axis=0), axis=2)))

    def frame(self, xpos, hd):
        lm = self.lm
        ang = []
        for a, b in self.segs:
            vr = xpos[b] - xpos[a]
            vh = np.asarray(hd[lm[b]][0]) - np.asarray(hd[lm[a]][0])
            nr, nh = np.linalg.norm(vr), np.linalg.norm(vh)
            if nr > 1e-3 and nh > 1e-3:
                ang.append(np.degrees(np.arccos(np.clip(vr @ vh / (nr * nh), -1, 1))))
        r0, h0 = xpos[self.root_bid], np.asarray(hd[self.root_h][0])
        dvec = np.array([(xpos[b] - r0) - self.scale * (np.asarray(hd[lm[b]][0]) - h0)
                         for b in lm if b != self.root_bid])
        self.dvecs.append(dvec)
        pos = np.linalg.norm(dvec, axis=1)
        flips, nflip = 0, 0
        for a, b, c in self.trip:
            def bendn(pa, pb, pc):
                v1, v2 = pa - pb, pc - pb
                cr = np.cross(v1, v2)
                return cr / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            br = bendn(xpos[a], xpos[b], xpos[c])
            bh = bendn(*(np.asarray(hd[lm[x]][0]) for x in (a, b, c)))
            if np.linalg.norm(bh) > 0.17 and np.linalg.norm(br) > 0.17:  # 둘 다 10° 이상 굽음
                nflip += 1
                flips += float(br @ bh < 0)
        return np.mean(ang) if ang else np.nan, np.mean(pos), flips, nflip


def make_retargeter(method, format_type, robot, human_height, frames, bone_hierarchy, **kw):
    if method == "chain":
        r = ChainMotionRetargeting(
            src_human=f"bvh_{format_type}" if format_type != "robot" else "robot",
            tgt_robot=robot, actual_human_height=human_height, verbose=False)
        r._bone_hierarchy = bone_hierarchy
    elif method == "ik":
        r = GMR(src_human=f"bvh_{format_type}", tgt_robot=robot,
                actual_human_height=human_height, solver="daqp", verbose=False)
    elif method == "auto":
        r = AutoChainRetargeting(robot, **kw)
        r.prepare(frames, bone_hierarchy)
    else:
        raise ValueError(method)
    return r


def evaluate_single(bvh_file, format_type, robot, method, max_frames=None, fps=30.0, **kw):
    """단일 BVH 파일에 대해 retarget 수행 및 메트릭 수집."""

    frames, human_height, bone_hierarchy = load_bvh_file(bvh_file, format=format_type)
    if max_frames:
        frames = frames[:max_frames]

    retargeter = make_retargeter(method, format_type, robot, human_height, frames, bone_hierarchy, **kw)
    model = retargeter.model if method != "ik" else retargeter.configuration.model
    stand = int(np.argmax([f[bone_hierarchy['bones'][0]][0][2] - min(v[0][2] for v in f.values())
                           for f in frames]))
    common = CommonMetrics(robot, format_type, model, frames[stand])

    frame_g_mpbpe, frame_mpbpe = [], []
    chain_errors = {}
    c_dir, c_pos, flips, nflip = [], [], 0, 0
    qs = []
    mapping = root_name = root_bid = None
    n_errors = 0
    t0 = time.time()

    for fi, human_data in enumerate(tqdm(frames, desc=os.path.basename(bvh_file), leave=False,
                                         disable=kw.get('quiet', True))):
        try:
            qpos = retargeter.retarget(human_data)
        except Exception:
            n_errors += 1
            continue
        xpos = get_target_xpos(retargeter, method)
        qs.append(np.array(qpos, dtype=float).copy())

        g_errors, r_errors = [], []
        if method == "auto":
            pts = retargeter.own_reference_points(human_data)
            rp, sp = retargeter.pelvis_points(human_data)
            for p_r, p_s, j in pts:
                cname = retargeter.chains[j]['name']
                g_err = np.linalg.norm(p_r - p_s)
                r_err = np.linalg.norm((p_r - rp) - (p_s - sp))
                g_errors.append(g_err)
                r_errors.append(r_err)
                chain_errors.setdefault(cname, {'global': [], 'relative': []})
                chain_errors[cname]['global'].append(g_err)
                chain_errors[cname]['relative'].append(r_err)
            mapping = pts
        else:
            if mapping is None:
                mapping = get_body_mapping(retargeter, method)
                root_name, root_bid = get_root_info(retargeter, method)
                for _, _, cname in mapping:
                    chain_errors.setdefault(cname, {'global': [], 'relative': []})
            scaled = retargeter.scaled_human_data
            if root_name not in scaled:
                continue
            src_root = np.asarray(scaled[root_name][0])
            tgt_root = xpos[root_bid].copy()
            for src_body, tgt_bid, cname in mapping:
                if src_body not in scaled:
                    continue
                src_pos = np.asarray(scaled[src_body][0])
                tgt_pos = xpos[tgt_bid].copy()
                g_err = np.linalg.norm(src_pos - tgt_pos)
                r_err = np.linalg.norm((src_pos - src_root) - (tgt_pos - tgt_root))
                g_errors.append(g_err)
                r_errors.append(r_err)
                chain_errors[cname]['global'].append(g_err)
                chain_errors[cname]['relative'].append(r_err)
        if g_errors:
            frame_g_mpbpe.append(np.mean(g_errors))
            frame_mpbpe.append(np.mean(r_errors))
        if common.ok:
            a, p, fl, nf = common.frame(xpos, human_data)
            c_dir.append(a)
            c_pos.append(p)
            flips += fl
            nflip += nf

    elapsed = time.time() - t0
    result = {
        'bvh_file': os.path.basename(bvh_file), 'robot': robot, 'method': method,
        'n_frames': len(frame_g_mpbpe), 'n_mapped_bodies': len(mapping) if mapping else 0,
        'n_errors': n_errors, 'sec_per_frame': elapsed / max(len(frames), 1),
    }
    if frame_g_mpbpe:
        g, r = np.array(frame_g_mpbpe), np.array(frame_mpbpe)
        result.update({
            'E_g_mpbpe_mean': float(np.mean(g)), 'E_g_mpbpe_std': float(np.std(g)),
            'E_g_mpbpe_median': float(np.median(g)), 'E_g_mpbpe_max': float(np.max(g)),
            'E_mpbpe_mean': float(np.mean(r)), 'E_mpbpe_std': float(np.std(r)),
            'E_mpbpe_median': float(np.median(r)), 'E_mpbpe_max': float(np.max(r)),
        })
        result['per_chain'] = {c: {'E_g_mean': float(np.mean(e['global'])),
                                   'E_r_mean': float(np.mean(e['relative']))}
                               for c, e in chain_errors.items() if e['global']}
    else:
        result['E_g_mpbpe_mean'] = result['E_mpbpe_mean'] = float('nan')
    if c_dir:
        result['E_dir_deg'] = float(np.nanmean(c_dir))
        result['E_cmp'] = float(np.mean(c_pos))
        result['flip_rate'] = flips / max(nflip, 1)
        result['E_cmp_dyn'] = common.dynamic_error()
    if len(qs) > 2:
        Q = np.array(qs)
        j0 = 7 if model.njnt > 0 and model.jnt_type[0] == 0 else 0
        acc = np.diff(Q[:, j0:], n=2, axis=0) * fps ** 2
        result['joint_acc'] = float(np.mean(np.abs(acc)))
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
    parser.add_argument("--method", choices=["ik", "chain", "auto"], default="chain")
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