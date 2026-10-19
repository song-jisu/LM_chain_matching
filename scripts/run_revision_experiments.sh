#!/usr/bin/env bash
# IeMeK 재심 대응 실험 일괄 실행.
#   LAFAN=/SHARED/dataset/LaFAN1 bash scripts/run_revision_experiments.sh [g1|ablation|robots|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LAFAN=${LAFAN:-/SHARED/dataset/LaFAN1}
OUT=${OUT:-results/revision}
LIST=results/lafan1_29motions.txt
W=${W:-29}
export OMP_NUM_THREADS=1
mkdir -p "$OUT"

run() {  # run <name> <robot> <method> [kwargs-json]
  local name=$1 robot=$2 method=$3 kw=${4:-'{}'}
  [ -f "$OUT/$name.csv" ] && { echo "skip $name"; return; }
  echo "== $name ($robot, $method, $kw)"
  $PY scripts/run_eval_parallel.py --bvh_dir "$LAFAN" --file_list $LIST --robot "$robot" \
      --method "$method" --workers "$W" --kwargs "$kw" --output_csv "$OUT/$name.csv"
}

what=${1:-all}

if [[ $what == g1 || $what == all ]]; then
  run g1_ik     unitree_g1 ik
  run g1_auto   unitree_g1 auto
  run g1_legacy unitree_g1 chain
fi

if [[ $what == ablation || $what == all ]]; then
  run g1_auto_yaw       unitree_g1 auto '{"root_rot": "yaw"}'
  run g1_auto_bendcross unitree_g1 auto '{"bend": "cross"}'
  run g1_auto_bendnone  unitree_g1 auto '{"bend": "none"}'
  for wb in 0.1 1.0 3.0; do run g1_auto_wbend$wb unitree_g1 auto "{\"w_bend\": $wb}"; done
  for wr in 0.0 0.01 0.2; do run g1_auto_wreg$wr  unitree_g1 auto "{\"w_reg\": $wr}"; done
fi

if [[ $what == robots || $what == all ]]; then
  # GMR LaFAN1 설정이 있는 로봇: GMR과 직접 비교
  for r in booster_t1_29dof fourier_n1 engineai_pm01 stanford_toddy; do
    run ${r}_ik   $r ik
    run ${r}_auto $r auto
  done
  # GMR LaFAN1 설정이 없는 로봇: auto만 (자기 기준 지표)
  for r in unitree_h1 unitree_h1_2 booster_k1 kuavo_s45 hightorque_hi pal_talos fourier_gr3 \
           pnd_adam_lite tienkung berkeley_humanoid_lite galaxea_r1pro; do
    run ${r}_auto $r auto
  done
fi

if [[ $what == r2r || $what == all ]]; then
  [ -f "$OUT/robot2robot_g1.csv" ] || $PY scripts/eval_robot_to_robot.py --bvh_dir "$LAFAN" --file_list $LIST \
      --source_robot unitree_g1 --workers "$W" --max_frames 2000 \
      --targets unitree_g1 unitree_h1 booster_t1_29dof fourier_n1 engineai_pm01 \
      --output_csv "$OUT/robot2robot_g1.csv"
fi
