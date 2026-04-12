# Motion Retarget 성능 평가 결과

## 평가 기준

GMR 논문 ("Retargeting Matters", ICRA 2026, arXiv 2510.02252) 메트릭 사용.

| Metric | 정의 |
|--------|------|
| **E_g-mpbpe** | Global Mean Position Body Part Error: `mean(‖src_pos - tgt_pos‖)` |
| **E_mpbpe** | Root-relative Mean Position Body Part Error: `mean(‖(src-src_root) - (tgt-tgt_root)‖)` |

## 실험 조건

- **Target robot**: Unitree G1 (29 DOF)
- **Source**: LaFAN1 dataset (5 BVH files, 200 frames/file)
- **Format**: lafan1
- **Date**: 2026-04-13

## 전체 비교: Chain LM vs GMR IK

| Metric | **Chain LM** | **GMR IK** |
|--------|:-----------:|:---------:|
| **E_g-mpbpe (m)** | 0.117 | **0.085** |
| **E_mpbpe (m)** | 0.117 | **0.092** |
| Mapped bodies/frame | 13 | 15 |
| 성공 프레임 | **1000/1000** (100%) | 505/1000 (50.5%) |
| 처리 시간 (1000f) | 71.4s | ~15s |

## Per-file 결과

### Chain LM

| File | Frames | E_g-mpbpe | E_mpbpe |
|------|-------:|----------:|--------:|
| aiming1_subject1.bvh | 200 | 0.1059 | 0.1059 |
| aiming1_subject4.bvh | 200 | 0.1130 | 0.1130 |
| aiming2_subject2.bvh | 200 | 0.1237 | 0.1237 |
| aiming2_subject3.bvh | 200 | 0.1225 | 0.1225 |
| aiming2_subject5.bvh | 200 | 0.1187 | 0.1187 |

### GMR IK

| File | Frames | E_g-mpbpe | E_mpbpe |
|------|-------:|----------:|--------:|
| aiming1_subject1.bvh | 174 | 0.0841 | 0.0907 |
| aiming1_subject4.bvh | 118 | 0.0867 | 0.0938 |
| aiming2_subject2.bvh | 0 | - | - |
| aiming2_subject3.bvh | 110 | 0.0856 | 0.0931 |
| aiming2_subject5.bvh | 103 | 0.0838 | 0.0912 |

## Per-chain 분석

### Chain LM (chain 단위)

| Chain | E_mpbpe | E_g-mpbpe |
|-------|--------:|----------:|
| left_hip→left_ankle | 0.107 | 0.107 |
| right_hip→right_ankle | 0.119 | 0.119 |
| waist→torso | 0.231 | 0.231 |
| left_shoulder→left_wrist | 0.073 | 0.073 |
| right_shoulder→right_wrist | 0.083 | 0.083 |

### GMR IK (body 단위)

| Body | E_mpbpe | E_g-mpbpe |
|------|--------:|----------:|
| pelvis | 0.000 | 0.010 |
| left_hip_yaw | 0.258 | 0.249 |
| left_knee | 0.068 | 0.060 |
| left_ankle_roll | 0.011 | 0.003 |
| left_elbow | 0.071 | 0.065 |
| left_shoulder_yaw | 0.174 | 0.166 |
| left_wrist_yaw | 0.084 | 0.075 |
| right_hip_yaw | 0.256 | 0.246 |
| right_knee | 0.068 | 0.060 |
| right_ankle_roll | 0.011 | 0.003 |
| right_elbow | 0.078 | 0.073 |
| right_shoulder_yaw | 0.166 | 0.159 |
| right_wrist_yaw | 0.095 | 0.087 |

## 분석

### GMR IK 장점
- 전체 오차 약 20-25% 낮음
- Body 단위 세밀한 매핑 (15개 body)
- 처리 속도 ~5x 빠름

### GMR IK 단점
- **Joint limit crash**: 약 50% 프레임에서 mink solver가 joint limit violation으로 실패
- **hip_yaw 오차 25cm**: 가장 큰 오차 지점
- Per-robot IK config JSON 수동 튜닝 필요

### Chain LM 장점
- **100% 안정성**: 모든 프레임에서 crash 없이 동작
- IK config 없이 자동 chain 매칭 가능
- Robot→robot retarget 지원 (BVH 출력/입력)

### Chain LM 단점
- 전체 오차 약 30% 높음 (11.7cm vs 8.5cm)
- **Waist chain 오차 23cm**: 가장 큰 병목
- Chain 단위 매핑으로 body 수가 적음 (13개)

### E_g-mpbpe = E_mpbpe인 이유 (Chain LM)
Chain LM은 root position을 source BVH에서 직접 복사하므로 `src_root ≈ tgt_root`. 따라서 global 오차와 root-relative 오차가 거의 동일.

## 재현 방법

```bash
# Chain LM
python scripts/evaluate_retarget.py \
  --bvh_dir C:/Datasets/ubisoft-laforge-animation-dataset/lafan1/lafan1/ \
  --format lafan1 --robot unitree_g1 --method chain \
  --max_files 5 --max_frames 200 \
  --output_csv results/eval_chain_g1.csv

# GMR IK
python scripts/evaluate_retarget.py \
  --bvh_dir C:/Datasets/ubisoft-laforge-animation-dataset/lafan1/lafan1/ \
  --format lafan1 --robot unitree_g1 --method ik \
  --max_files 5 --max_frames 200 \
  --output_csv results/eval_ik_g1.csv
```