"""
evaluate_retarget.evaluate_single을 파일 단위로 병렬 실행하고 CSV로 저장.

  python scripts/run_eval_parallel.py --bvh_dir /SHARED/dataset/LaFAN1 \
      --file_list results/lafan1_29motions.txt --robot unitree_g1 --method chain \
      --output_csv results/eval.csv --workers 29
"""

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_retarget as ev  # noqa: E402

FIELDS = ['bvh_file', 'robot', 'method', 'n_frames', 'n_mapped_bodies', 'n_errors',
          'E_g_mpbpe_mean', 'E_g_mpbpe_std', 'E_g_mpbpe_median', 'E_g_mpbpe_max',
          'E_mpbpe_mean', 'E_mpbpe_std', 'E_mpbpe_median', 'E_mpbpe_max',
          'E_dir_deg', 'E_cmp', 'E_cmp_dyn', 'flip_rate', 'joint_acc', 'sec_per_frame', 'per_chain']


def _run(job):
    path, fmt, robot, method, max_frames, kwargs = job
    t0 = time.time()
    try:
        res = ev.evaluate_single(path, fmt, robot, method, max_frames, **kwargs)
    except Exception as e:  # 파일 단위 실패도 기록한다
        return {'bvh_file': os.path.basename(path), 'robot': robot, 'method': method,
                'n_frames': 0, 'error': repr(e)}
    res['sec_per_frame'] = (time.time() - t0) / max(res['n_frames'], 1)
    res['per_chain'] = json.dumps(res.get('per_chain', {}))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bvh_dir', required=True)
    p.add_argument('--file_list', default=None, help='한 줄에 BVH 파일명 하나')
    p.add_argument('--format', default='lafan1')
    p.add_argument('--robot', default='unitree_g1')
    p.add_argument('--method', default='chain')
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--output_csv', required=True)
    p.add_argument('--kwargs', default='{}', help='evaluate_single에 넘길 JSON (예: retargeter 옵션)')
    a = p.parse_args()

    if a.file_list:
        names = [l.strip() for l in open(a.file_list) if l.strip()]
    else:
        names = sorted(f for f in os.listdir(a.bvh_dir) if f.endswith('.bvh'))
    kwargs = json.loads(a.kwargs)
    jobs = [(os.path.join(a.bvh_dir, n), a.format, a.robot, a.method, a.max_frames, kwargs)
            for n in names]

    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(_run, jobs, chunksize=1)
    print(f'{len(results)} files in {time.time() - t0:.0f}s')

    os.makedirs(os.path.dirname(a.output_csv) or '.', exist_ok=True)
    with open(a.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ['error'], extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    for r in results:
        if 'error' in r:
            print('ERROR', r['bvh_file'], r['error'])
    print('saved', a.output_csv)


if __name__ == '__main__':
    main()
