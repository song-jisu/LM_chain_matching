# MOREFLOW vs Chain LM: One-to-One 한계 분석 및 Any-to-Any 가능성

## Context

MOREFLOW ("Motion Retargeting Learning Through Unsupervised Flow Matching", arXiv 2509.25600)는
unsupervised flow matching 기반 motion retargeting으로 높은 품질을 보여주지만,
**one-to-one (단일 source-target 쌍) 모델만 학습 가능**하다는 근본적 한계가 있음.
Chain LM 접근과 비교하여 이 한계의 원인과 해결 가능성을 분석한다.

## MOREFLOW 구조 요약

```
Stage 1: Per-character VQ-VAE
  Character A motion → VQ-VAE_A → discrete tokens_A
  Character B motion → VQ-VAE_B → discrete tokens_B
  (각 캐릭터별 독립 학습, 쌍 데이터 불필요)

Stage 2: Pairwise Flow Matching
  tokens_A → Flow Model_{A→B} → tokens_B
  (A→B 쌍에 대해 별도 모델 학습)
  
  Conditioning: local-frame / world-frame / joint-space
  Classifier-free guidance로 조건 강도 조절
```

## One-to-One 한계의 3가지 근본 원인

### 1. Per-character VQ-VAE (가장 핵심)

각 캐릭터마다 **독립적인 VQ-VAE encoder**를 학습:
- Character A의 codebook은 A의 모션 분포만 반영
- Character B의 codebook은 B의 모션 분포만 반영
- **두 codebook 사이에 의미적 대응 관계가 없음**

새 캐릭터 C가 추가되면:
- VQ-VAE_C를 새로 학습해야 하고
- Flow Model_{A→C}, Flow Model_{B→C}도 새로 학습해야 함
- N개 캐릭터 → N개 VQ-VAE + N(N-1)개 flow model

**Chain LM 대비**: Chain LM은 캐릭터별 학습이 없음. 
MuJoCo FK + analytical Jacobian으로 실시간 최적화하므로 새 로봇을 추가해도 학습 불필요.

### 2. Skeleton-specific 입력 표현

VQ-VAE 입력이 **캐릭터 고유의 joint 구조**에 종속:
- Humanoid (22 joints) → 22×3 = 66차원 입력
- Quadruped (16 joints) → 16×3 = 48차원 입력
- 입력 차원이 다르면 같은 네트워크를 공유할 수 없음

**Chain LM 대비**: Chain LM은 "serial chain" 단위로 표현을 정규화.
- 어떤 skeleton이든 serial chain들의 집합으로 분해
- chain 내 body positions의 shape = (n_bodies, 3)
- n_bodies가 다르면 LM이 자동으로 맞춤 (position interpolation)

### 3. Pairwise Flow Model

Flow matching 모델이 **특정 source→target 쌍**에 대해서만 학습:
- `p(tokens_B | tokens_A)` 분포를 학습
- A→B 모델은 C→B에 사용할 수 없음 (tokens_A와 tokens_C의 분포가 다름)

논문의 우회법: **Chain-of-retargeting** (A→B→C)
→ 중간 캐릭터를 거쳐 간접 변환, 하지만 오차 누적 + 2배 비용

## Chain LM이 Any-to-Any를 가능하게 하는 핵심 요소

### 1. Skeleton-agnostic 표현: Serial Chain

```
어떤 skeleton이든:
  Body tree → branch point에서 분할 → serial chains
  
  Human:  [Hips→LeftToe, Hips→RightToe, Spine→Head, ...]
  G1:     [pelvis→left_ankle, pelvis→right_ankle, waist→torso, ...]
  Dog:    [torso→left_front_paw, torso→left_rear_paw, ...]
```

Chain은 **topology-independent**한 단위:
- 시작점, 끝점, 중간 body positions로 정의
- joint 수가 달라도 position sequence로 표현 가능
- 굽힘 방향(bend direction)으로 chain "shape"을 기술

### 2. Position 기반 매칭 (rotation 불필요)

MOREFLOW는 joint angles/rotations을 다루므로 좌표계 변환 문제가 큼.
Chain LM은 **3D position만 사용**:
- Source chain positions → target chain positions (LM 최적화)
- 좌표계 변환은 forward alignment로 자동 처리
- 회전 표현의 ambiguity (gimbal lock, quaternion 부호 등)를 회피

### 3. 학습 없이 즉시 사용

| | MOREFLOW | Chain LM |
|---|---------|---------|
| 새 로봇 추가 | VQ-VAE + Flow 학습 필요 | XML/URDF만 있으면 즉시 사용 |
| 학습 데이터 | 캐릭터별 모션 데이터 필요 | 불필요 |
| 쌍 추가 | 쌍별 Flow 모델 학습 | CLI에서 chain 매칭만 |

## MOREFLOW + Chain LM 통합 가능성

### 아이디어: Chain-normalized Latent Space

MOREFLOW의 one-to-one 한계를 Chain LM으로 해결하는 방법:

```
기존 MOREFLOW:
  Motion_A → VQ-VAE_A → tokens_A → Flow_{A→B} → tokens_B → VQ-VAE_B → Motion_B

제안:
  Motion_any → Chain Decomposition → [chain shapes] → Unified VQ-VAE → tokens
                                                              ↓
  tokens → Flow Model (single, conditioned on target topology) → tokens_target
                                                              ↓
  tokens_target → Chain LM (target FK + Jacobian) → Motion_target
```

핵심 변경:
1. **Per-character VQ-VAE → Unified chain-level VQ-VAE**
   - 입력: chain shape (position sequence, normalized by chain length)
   - 모든 캐릭터의 모든 chain이 같은 VQ-VAE를 공유
   - codebook이 "chain shape"의 universal vocabulary가 됨

2. **Pairwise Flow → Topology-conditioned Flow**
   - Flow model에 target chain 구조 (길이, joint 수, limits)를 condition으로 제공
   - 하나의 flow model이 모든 source→target 변환을 수행

3. **출력: Flow tokens → Chain LM**
   - Flow가 출력한 target chain shape를 Chain LM의 target positions로 사용
   - Chain LM이 실제 joint angles를 FK + Jacobian으로 계산
   - Joint limits, physical constraints를 자연스럽게 반영

### 예상 장점
- **Any-to-any**: N개 캐릭터에 대해 1개 VQ-VAE + 1개 Flow model로 충분
- **Zero-shot 새 캐릭터**: chain 구조만 있으면 학습 없이 retargeting 가능
- **물리적 타당성**: Chain LM의 joint limit + FK constraint 보장

### 예상 과제
- Chain shape normalization의 정보 손실 (chain 간 coupling 무시)
- Unified VQ-VAE의 codebook이 다양한 morphology를 커버해야 함
- Flow conditioning의 복잡도 증가

## 결론

| 측면 | MOREFLOW | Chain LM | 통합 시 |
|------|---------|---------|--------|
| 품질 | 높음 (학습 기반) | 중간 (최적화 기반) | 높음 (기대) |
| 범용성 | one-to-one | **any-to-any** | **any-to-any** |
| 새 캐릭터 | 재학습 필요 | 즉시 사용 | 즉시 사용 (기대) |
| 물리 제약 | 보장 안 됨 | joint limits 반영 | 반영 |
| 속도 | 빠름 (inference) | 중간 (LM per frame) | 중간 |

**Chain LM의 serial chain 분해가 MOREFLOW의 one-to-one 한계를 해결하는 열쇠가 될 수 있음.**
Per-character 표현을 chain-level universal 표현으로 대체하면,
하나의 모델로 임의의 source→target 변환이 가능해짐.
