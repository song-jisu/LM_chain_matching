"""
로봇→로봇 리타게팅 평가 (심사위원 2: any-to-any, 심사위원 3: 동형 로봇 검증).

1) LaFAN1 → 소스 로봇(기본 G1)을 auto로 리타게팅해 로봇 모션을 만든다.
2) 그 로봇 모션을 '소스 스켈레톤'으로 변환한다: 바디 원점 = bone, body_parentid = 계층.
   (로봇 MJCF와 qpos만 있으면 되고, 이름·설정 정보는 쓰지 않는다)
3) 소스 로봇 → 타겟 로봇들을 auto로 리타게팅한다.
   - 자기 리타게팅(G1→G1): 소스 qpos와의 관절각 오차 → 0에 가까워야 한다.
   - 경로 독립성: LaFAN1→X 직접 결과와 LaFAN1→G1→X 결과의 관절각 차이.

  python scripts/eval_robot_to_robot.py --source_robot unitree_g1 \
      --targets unitree_g1 unitree_h1 booster_t1_29dof fourier_n1 \
      --max_frames 2000 --output_csv results/revision/robot2robot.csv
"""

import argparse
import csv
import os
import sys
import time
import types
from multiprocessing import Pool

import mujoco as mj
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
_pkg = types.ModuleType('general_motion_retargeting')
_pkg.__path__ = [os.path.join(HERE, 'general_motion_retargeting')]
sys.modules['general_motion_retargeting'] = _pkg
from general_motion_retargeting.auto_chain_retarget import AutoChainRetargeting  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402


def robot_motion_to_frames(retargeter, qpos_seq):
    """로봇 qpos 시퀀스 → 관절 중심 스켈레톤 (frames, bone_hierarchy).

    bone = 루트 바디 원점 + 각 체인의 key point(인접 hinge 축 교점 = 관절 중심, tip 포함).
    바디 원점은 관절 중심이 아니므로(예: G1 hip_yaw_link 원점은 허벅지 중간) 그대로 쓰면
    다른 로봇의 관절과 어긋난다. 모션캡처가 관절 중심을 기록하는 것과 같은 형식으로 내보낸다.
    계층: 체인 내부는 key 순서, 체인 첫 key의 부모는 부모 체인의 마지막 key(없으면 루트).
    """
    m, chains = retargeter.model, retargeter.chains
    root = retargeter.root_bid
    names, parents, pts = ['root'], [-1], [(root, np.zeros(3))]
    last_idx = {}
    order = []

    def visit(ci):
        order.append(ci)
        for c in chains[ci]['children']:
            visit(c)
    for ci, ch in enumerate(chains):
        if ch['parent'] is None:
            visit(ci)
    for ci in order:
        ch = chains[ci]
        par = last_idx.get(ch['parent'], 0)
        for k, (bid, local, _) in enumerate(ch['keys']):
            names.append(f"c{ci}_k{k}")
            parents.append(par)
            pts.append((bid, local))
            par = len(names) - 1
        last_idx[ci] = par
    d = mj.MjData(m)
    frames = []
    q_id = np.array([1.0, 0, 0, 0])
    for q in qpos_seq:
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        frames.append({n: [d.xpos[b] + d.xmat[b].reshape(3, 3) @ l, q_id]
                       for n, (b, l) in zip(names, pts)})
    return frames, {'bones': names, 'parents': parents}


def hinge_slice(model):
    return slice(7, None) if int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE) else slice(0, None)


def retarget_all(r, frames):
    t0 = time.time()
    qs, errs = [], []
    for f in frames:
        qs.append(r.retarget(f).copy())
        pts = r.own_reference_points(f)
        errs.append(np.mean([np.linalg.norm(a - b) for a, b, _ in pts]))
    return np.array(qs), float(np.mean(errs)), (time.time() - t0) / len(frames)


def job(args):
    bvh, src_robot, targets, max_frames = args
    frames, _, bh = load_bvh_file(bvh)
    frames = frames[:max_frames] if max_frames else frames
    rows = []

    rs = AutoChainRetargeting(src_robot)
    rs.prepare(frames, bh)
    q_src, e_src, _ = retarget_all(rs, frames)
    rframes, rbh = robot_motion_to_frames(rs, q_src)

    for tgt in targets:
        rt = AutoChainRetargeting(tgt)
        rt.prepare(rframes, rbh)
        q_rr, e_rr, spf = retarget_all(rt, rframes)
        sl = hinge_slice(rt.model)
        row = {'bvh_file': os.path.basename(bvh), 'source': src_robot, 'target': tgt,
               'n_frames': len(rframes), 'E_g_own': e_rr, 'sec_per_frame': spf,
               'n_matched': len(rt.match), 'n_chains': len(rt.chains)}
        if tgt == src_robot:
            row['self_joint_err_deg'] = float(np.degrees(np.mean(np.abs(q_rr[:, sl] - q_src[:, sl]))))
            row['self_root_err_m'] = float(np.mean(np.linalg.norm(q_rr[:, :3] - q_src[:, :3], axis=1)))
        # 경로 독립성: LaFAN1 → tgt 직접
        rd = AutoChainRetargeting(tgt)
        rd.prepare(frames, bh)
        q_dir, _, _ = retarget_all(rd, frames)
        row['path_joint_diff_deg'] = float(np.degrees(np.mean(np.abs(q_rr[:, sl] - q_dir[:, sl]))))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bvh_dir', default='/SHARED/dataset/LaFAN1')
    p.add_argument('--file_list', default=os.path.join(HERE, 'results', 'lafan1_29motions.txt'))
    p.add_argument('--source_robot', default='unitree_g1')
    p.add_argument('--targets', nargs='+', default=['unitree_g1', 'unitree_h1', 'booster_t1_29dof',
                                                     'fourier_n1', 'engineai_pm01'])
    p.add_argument('--max_frames', type=int, default=2000)
    p.add_argument('--workers', type=int, default=29)
    p.add_argument('--output_csv', required=True)
    a = p.parse_args()
    names = [l.strip() for l in open(a.file_list) if l.strip()]
    jobs = [(os.path.join(a.bvh_dir, n), a.source_robot, a.targets, a.max_frames) for n in names]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(job, jobs, chunksize=1) for r in rs]
    fields = ['bvh_file', 'source', 'target', 'n_frames', 'n_matched', 'n_chains', 'E_g_own',
              'self_joint_err_deg', 'self_root_err_m', 'path_joint_diff_deg', 'sec_per_frame']
    os.makedirs(os.path.dirname(a.output_csv) or '.', exist_ok=True)
    with open(a.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    for t in a.targets:
        rr = [r for r in rows if r['target'] == t]
        msg = f"{a.source_robot}->{t}: E_g_own {np.mean([r['E_g_own'] for r in rr]) * 1000:.1f} mm, " \
              f"path diff {np.mean([r['path_joint_diff_deg'] for r in rr]):.2f} deg"
        if t == a.source_robot:
            msg += f", self joint err {np.mean([r['self_joint_err_deg'] for r in rr]):.2f} deg"
        print(msg)
    print('saved', a.output_csv)


if __name__ == '__main__':
    main()
