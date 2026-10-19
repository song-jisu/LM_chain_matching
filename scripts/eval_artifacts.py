"""
물리적 아티팩트 평가 (심사위원 3-3/3-4: 실제 로봇에 쓸 수 있는 수준인가).

GMR 논문이 리타게팅 품질 문제로 지목한 아티팩트를 리타게팅 결과 qpos에서 직접 측정한다.
  - 지면 관통: 로봇 geom과 바닥의 접촉 거리(음수 = 관통)의 최대 깊이, 1 cm 이상 관통 프레임 비율
  - 자기 충돌: 바닥이 아닌 geom 쌍이 5 mm 이상 겹친 프레임 비율
  - 발 미끄러짐: 발 바디의 최저 geom 점이 지면 2 cm 이내인 연속 프레임에서의 수평 속도 평균
발 바디는 Auto의 다리 체인 마지막 관절 중심을 가진 바디로 정한다(평가에만 사용, 두 방법에 같게 적용).

  python scripts/eval_artifacts.py --robot unitree_g1 --output_csv results/revision/g1_artifacts.csv
"""

import argparse
import csv
import os
import sys
from multiprocessing import Pool

import mujoco as mj
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_retarget as ev  # noqa: E402

FPS = 30.0


def foot_bodies(robot, frames, bh):
    r = ev.AutoChainRetargeting(robot)
    r.prepare(frames[:300], bh)
    return [r.chains[j]['keys'][-1][0] for j in r.tgt_legs], r.model


def artifacts(model, qs, feet):
    d = mj.MjData(model)
    floor = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0]
    feet_geoms = {b: [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b] for b in feet}
    pen, selfc, slide = [], [], []
    prev = None
    for q in qs:
        d.qpos[:] = q
        mj.mj_forward(model, d)
        p_floor, p_self = 0.0, 0.0
        for c in d.contact[:d.ncon]:
            if c.geom1 in floor or c.geom2 in floor:
                p_floor = max(p_floor, -c.dist)
            else:
                p_self = max(p_self, -c.dist)
        pen.append(p_floor)
        selfc.append(p_self)
        cur = {}
        for b, gs in feet_geoms.items():
            zmin = min((d.geom_xpos[g][2] + d.geom_xmat[g].reshape(3, 3)[2] @ model.geom_aabb[g][:3]
                        - np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_aabb[g][3:]) for g in gs) \
                if gs else d.xpos[b][2]
            cur[b] = (d.xpos[b].copy(), zmin)
        if prev is not None:
            for b in feet:
                (p0, z0), (p1, z1) = prev[b], cur[b]
                if z0 < 0.02 and z1 < 0.02:
                    slide.append(np.linalg.norm((p1 - p0)[:2]) * FPS)
        prev = cur
    pen, selfc = np.array(pen), np.array(selfc)
    return {'pen_max_mm': float(pen.max() * 1000), 'pen_mean_mm': float(pen.mean() * 1000),
            'pen_frac_1cm': float((pen > 0.01).mean()), 'self_frac_5mm': float((selfc > 0.005).mean()),
            'slide_cm_s': float(np.mean(slide) * 100) if slide else float('nan'),
            'n_contact_steps': len(slide)}


def job(args):
    path, robot, method, max_frames = args
    frames, human_height, bh = ev.load_bvh_file(path, format='lafan1')
    frames = frames[:max_frames] if max_frames else frames
    feet, _ = foot_bodies(robot, frames, bh)
    r = ev.make_retargeter(method, 'lafan1', robot, human_height, frames, bh)
    model = r.model if method != 'ik' else r.configuration.model
    qs = [np.array(r.retarget(f), dtype=float).copy() for f in frames]
    if method == 'ik':  # 같은 MJCF로 측정(바디 id 일치 확인)
        assert model.nbody == ev.AutoChainRetargeting(robot).model.nbody
    out = artifacts(model, qs, feet)
    out.update({'bvh_file': os.path.basename(path), 'robot': robot, 'method': method, 'n_frames': len(qs)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bvh_dir', default='/SHARED/dataset/LaFAN1')
    p.add_argument('--file_list', default=os.path.join(ev.HERE, 'results', 'lafan1_29motions.txt'))
    p.add_argument('--robot', default='unitree_g1')
    p.add_argument('--methods', nargs='+', default=['ik', 'auto'])
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--workers', type=int, default=29)
    p.add_argument('--output_csv', required=True)
    a = p.parse_args()
    names = [l.strip() for l in open(a.file_list) if l.strip()]
    jobs = [(os.path.join(a.bvh_dir, n), a.robot, m, a.max_frames) for n in names for m in a.methods]
    with Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    fields = ['bvh_file', 'robot', 'method', 'n_frames', 'pen_max_mm', 'pen_mean_mm', 'pen_frac_1cm',
              'self_frac_5mm', 'slide_cm_s', 'n_contact_steps']
    with open(a.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for m in a.methods:
        rr = [r for r in rows if r['method'] == m]
        n = np.array([r['n_frames'] for r in rr], dtype=float)
        avg = lambda k: float(np.nansum([r[k] * w for r, w in zip(rr, n)]) / n.sum())
        print(m, {k: round(avg(k), 4) for k in fields[4:9]})
    print('saved', a.output_csv)


if __name__ == '__main__':
    main()
