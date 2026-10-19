# IeMeK 재심 대응: 설정 없는(Auto) Chain LM

제출본의 Chain LM은 G1 실험에서 GMR의 `bvh_lafan1_to_g1.json` 바디 매핑을 읽어 썼고,
설정이 없으면 CLI로 사람이 체인 쌍을 입력해야 했다. 이 디렉터리는 그 부분을 자동화한
`general_motion_retargeting/auto_chain_retarget.py`(`AutoChainRetargeting`)의 평가 결과다.

## 무엇이 바뀌었나

| 항목 | 제출본 (`chain_motion_retarget.py`) | Auto (`auto_chain_retarget.py`) |
|---|---|---|
| 바디 대응 | GMR JSON `ik_match_table` 또는 CLI 입력 | 자동 (설정·이름 규칙 없음) |
| 체인 매칭 | 없음 (논문의 Hungarian은 미구현) | 정규화 시작점 + 길이 + 좌우·내부/말단·다리 일치 비용 → Hungarian |
| 좌/우 판정 | 이름(설정) | 무릎이 전방으로 굽는다는 사실(소스) / MJCF +X 전방 관례(로봇) |
| 로봇 관절 위치 | 바디 원점 | 인접 hinge 축 교점(key point). 원점의 축 방향 위치는 임의이기 때문 |
| 체인 내부 대응 | 인덱스 비례(CLI 경로만) | 시작 노드 + 분율 보간, 다리 끝은 분절 방향+분율로 발목/발끝 판별 |
| G1 하드코딩 | `pelvis`, 0.793 m, `waist/torso/hip` 이름 | 없음 (freejoint 바디, geom으로 키·지면 계산) |
| 루트 | 위치 복사 + yaw | 위치·회전 6-DOF 최적화 (옵션 `root_rot="yaw"`) |
| 트리 처리 | waist 이름 체인만 Step 1 | 자식 체인이 있는 체인 = 내부 체인 → Step 1, 말단 체인 → Step 2 |
| 굽힘 잔차 | `n̂_r × n̂_s` (정반대도 0 → 반전 미방지) | `sinθ·n̂` 벡터 차 (반전 시 2sinθ, 펴짐에서 연속) |
| 추가 잔차 | – | 골반 방향 앵커(내부 체인 시작점), 발 방향(발목→발끝), 중립 자세 정규화 |
| Jacobian | 위치 해석적, 굽힘 수치, Step 1 수치 | 모두 해석적 |
| 고정 베이스 | 미지원 | 지원 (소스를 골반 heading 좌표계로) |

## 평가 지표

- `E_g`, `E_mpbpe`: 각 방법의 자기 기준(제출본과 동일한 정의). 방법마다 스케일·바디 집합이 달라 직접 비교용이 아니다.
  제출본은 출력 루트 위치를 기준 루트로 덮어써서 두 값이 구조적으로 같다(심사위원 2-2).
- 공통 지표(모든 방법에 동일, 정답 대응은 GMR JSON의 위치 가중치>0 바디 — 평가에만 사용):
  - `E_dir`: 허벅지·정강이·위팔·아래팔 방향 각도 오차
  - `E_cmp`: 사지 길이 비 균일 스케일 기준 루트 상대 위치 오차
  - `E_cmp_dyn`: `E_cmp`에서 랜드마크별 시간 평균 오프셋(바디 원점 관례 차이)을 뺀 값
  - `flip`: 무릎·팔꿈치가 둘 다 10° 이상 굽은 프레임 중 굽힘 법선이 반대인 비율
  - `joint acc`: 관절 가속도 평균 크기(30 fps)

## 재현

```bash
uv venv -p 3.10 .venv && uv pip install -p .venv/bin/python mink mujoco numpy scipy "qpsolvers[daqp]" daqp rich tqdm natsort
LAFAN=/SHARED/dataset/LaFAN1 bash scripts/run_revision_experiments.sh all
.venv/bin/python scripts/summarize_revision.py > results/revision/summary.md
```

LaFAN1 원본: https://github.com/ubisoft/ubisoft-laforge-animation-dataset (`lafan1/lafan1.zip`), `/SHARED/dataset/LaFAN1`.
