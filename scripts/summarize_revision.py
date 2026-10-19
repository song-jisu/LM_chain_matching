"""results/revision/*.csv → 논문 표용 요약(markdown)."""

import csv
import glob
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'results', 'revision')


def load(name):
    p = os.path.join(OUT, f'{name}.csv')
    if not os.path.exists(p):
        return None
    return [r for r in csv.DictReader(open(p)) if not r.get('error')]


def col(rows, k, scale=1.0):
    v = [float(r[k]) * scale for r in rows if r.get(k) not in (None, '', 'nan')]
    return np.array(v)


def stats(rows, k, scale=1.0):
    v = col(rows, k, scale)
    if len(v) == 0:
        return '-'
    return f"{v.mean():.1f}"


def frames(rows):
    return int(sum(int(r['n_frames']) for r in rows))


def table(title, names, labels=None):
    print(f"\n### {title}\n")
    print("| Method | files | frames | E_g (mm) | E_mpbpe (mm) | E_dir (deg) | E_cmp (mm) | E_cmp_dyn (mm) "
          "| flip (%) | joint acc (rad/s²) | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, lab in zip(names, labels or names):
        r = load(n)
        if r is None:
            print(f"| {lab} | (missing) |")
            continue
        print(f"| {lab} | {len(r)} | {frames(r)} | {stats(r, 'E_g_mpbpe_mean', 1000)} | "
              f"{stats(r, 'E_mpbpe_mean', 1000)} | {stats(r, 'E_dir_deg')} | {stats(r, 'E_cmp', 1000)} | "
              f"{stats(r, 'E_cmp_dyn', 1000)} | {stats(r, 'flip_rate', 100)} | {stats(r, 'joint_acc')} | "
              f"{int(col(r, 'n_errors').sum())} |")


def dist(names, labels, key='E_dir_deg', scale=1.0, unit='deg'):
    print(f"\n### Per-motion distribution of {key} ({unit})\n")
    print("| Method | Min | Median | Mean | Max |")
    print("|---|---|---|---|---|")
    for n, lab in zip(names, labels):
        r = load(n)
        if r is None:
            continue
        v = col(r, key, scale)
        print(f"| {lab} | {v.min():.1f} | {np.median(v):.1f} | {v.mean():.1f} | {v.max():.1f} |")


table("LaFAN1 → Unitree G1 (29 motions)",
      ['g1_ik', 'g1_legacy', 'g1_auto'],
      ['GMR IK (per-pair JSON)', 'Chain LM (submitted, GMR mapping)', 'Auto Chain LM (config-free)'])
dist(['g1_ik', 'g1_legacy', 'g1_auto'], ['GMR IK', 'Chain LM (submitted)', 'Auto Chain LM'])
dist(['g1_ik', 'g1_legacy', 'g1_auto'], ['GMR IK', 'Chain LM (submitted)', 'Auto Chain LM'],
     key='E_g_mpbpe_mean', scale=1000, unit='mm, own reference')

table("Ablation (Auto Chain LM, G1)",
      ['g1_auto', 'g1_auto_yaw', 'g1_auto_bendcross', 'g1_auto_bendnone',
       'g1_auto_wbend0.1', 'g1_auto_wbend1.0', 'g1_auto_wbend3.0',
       'g1_auto_wreg0.0', 'g1_auto_wreg0.01', 'g1_auto_wreg0.2'],
      ['full (default)', 'root yaw only', 'bend = cross (submitted form)', 'no bend term',
       'w_bend=0.1', 'w_bend=1.0', 'w_bend=3.0', 'w_reg=0', 'w_reg=0.01', 'w_reg=0.2'])

robots = sorted({os.path.basename(p)[:-len('_auto.csv')] for p in glob.glob(os.path.join(OUT, '*_auto.csv'))
                 if not os.path.basename(p).startswith('g1_')})
print("\n### LaFAN1 → other robots\n")
print("| Robot | method | frames | E_g own (mm) | E_dir (deg) | E_cmp_dyn (mm) | flip (%) | errors |")
print("|---|---|---|---|---|---|---|---|")
for rb in robots:
    for m in ('ik', 'auto'):
        r = load(f'{rb}_{m}')
        if r is None:
            continue
        print(f"| {rb} | {'GMR IK' if m == 'ik' else 'Auto'} | {frames(r)} | {stats(r, 'E_g_mpbpe_mean', 1000)} | "
              f"{stats(r, 'E_dir_deg')} | {stats(r, 'E_cmp_dyn', 1000)} | {stats(r, 'flip_rate', 100)} | "
              f"{int(col(r, 'n_errors').sum())} |")

r = load('robot2robot_g1')
if r:
    print("\n### Robot → robot (source: G1 motion retargeted from LaFAN1)\n")
    print("| Target | frames | E_g own (mm) | self joint err (deg) | path diff vs direct (deg) |")
    print("|---|---|---|---|---|")
    for t in dict.fromkeys(x['target'] for x in r):
        rr = [x for x in r if x['target'] == t]
        print(f"| {t} | {frames(rr)} | {stats(rr, 'E_g_own', 1000)} | {stats(rr, 'self_joint_err_deg')} | "
              f"{stats(rr, 'path_joint_diff_deg')} |")
