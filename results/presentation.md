# Chain-based Any-to-Any Motion Retargeting via Levenberg-Marquardt Optimization with Neural Warm Start

## 1. Introduction

Motion retargeting — transferring motion from one skeleton (source) to another (target) with different morphology — is fundamental to humanoid robot control, animation, and sim-to-real transfer. Existing approaches fall into two categories:

**Per-pair methods** (e.g., IK config tuning, MOREFLOW) achieve high quality but require manual configuration or retraining for each source-target pair. Given $N$ characters, this scales as $O(N^2)$.

**Universal methods** aim for any-to-any transfer but often sacrifice quality due to over-simplified skeleton representations.

We propose a **chain-based retargeting framework** with three progressively refined methods:

1. **Chain LM**: Decomposes any kinematic tree into serial chains and optimizes joint angles per-chain using Levenberg-Marquardt with MuJoCo analytical Jacobians. No per-pair configuration required.
2. **Chain LM-NN**: Trains an MLP to predict joint angles from normalized chain shapes, providing real-time inference without iterative optimization.
3. **Hybrid (Chain LM-NN + LM)**: Uses NN predictions as warm start for LM, combining NN speed with LM precision.

## 2. Related Works

### 2.1 IK-based Motion Retargeting

**GMR** (Ze et al., ICRA 2026) uses mink IK solver with per-robot JSON configuration defining body-to-body mappings, position/rotation offsets, and tracking weights. High quality ($E_{mpbpe} \approx 0.09$ m on LaFAN1$\to$G1) but requires manual tuning per source-target pair and suffers from solver instability (~50% frame crash rate due to joint limit violations with proxqp solver).

### 2.2 Learning-based Retargeting

**MOREFLOW** (arXiv 2509.25600) uses per-character VQ-VAE tokenizers and pairwise flow matching. Achieves natural motion transfer including cross-morphology (humanoid$\leftrightarrow$quadruped), but is inherently one-to-one: $N$ characters require $N$ VQ-VAEs and $N(N-1)$ flow models.

**SAN** (Skeleton-Agnostic Network) and **HuMoT** learn skeleton-independent embeddings but require large-scale training data and cannot guarantee physical constraints.

### 2.3 Optimization-based Retargeting

Classical IK approaches (CCD, FABRIK, Jacobian-based) solve per-frame but typically operate on individual end-effectors without considering chain shape consistency or temporal coherence. Our work extends this with chain-level shape matching and bend direction constraints.

## 3. Methodology

### 3.1 Serial Chain Decomposition

Given a kinematic tree $\mathcal{T} = (\mathcal{B}, \mathcal{E})$ with body set $\mathcal{B}$ and parent edges $\mathcal{E}$, we decompose it into a set of serial chains $\{C_1, C_2, \ldots, C_M\}$.

**Definition.** A *serial chain* $C = (b_1, b_2, \ldots, b_L)$ is a maximal path in $\mathcal{T}$ where each intermediate body $b_i$ ($1 < i < L$) has exactly one child with actuated joints.

The decomposition algorithm:

1. Identify *branch points*: bodies with $\geq 2$ children having actuated descendants.
2. Trace from each branch point's children along single-child paths until the next branch point or leaf.
3. Each traced path forms a serial chain.

For a humanoid robot, this typically yields $M = 4\text{--}6$ chains: two legs, waist/torso, and two arms.

### 3.2 Source-Target Chain Matching

Source chains $\{C^s_i\}$ and target chains $\{C^t_j\}$ are matched via CLI:

$$\texttt{Chain matching: 0->0 0->2 1->1 1->3 4->4}$$

Each entry `S->T` assigns source chain $S$ to target chain $T$. Multiple targets per source are allowed (e.g., for 2-leg$\to$4-leg mapping). Chains sharing the same source are automatically grouped for topology-aware processing.

For each matched pair $(C^s_i, C^t_j)$, the source bodies are mapped to target bodies by index proportionality:

$$k_{tgt} = \left\lfloor k_{src} \cdot \frac{|C^t_j|}{|C^s_i|} \right\rfloor$$

### 3.3 Normalized Chain Shape Representation

For skeleton-agnostic processing, each chain's body positions are normalized into a fixed-size representation $S \in \mathbb{R}^{K \times 3}$:

1. **Resample**: Interpolate $L$ body positions to $K=8$ uniformly-spaced points along the chain arc length.
2. **Translate**: Subtract chain start position $\mathbf{p}_0$.
3. **Scale**: Divide by total chain arc length $\ell$.

$$S_k = \frac{\text{interp}(\mathbf{p}_{1:L},\; k/K) - \mathbf{p}_0}{\ell}, \quad k = 0, \ldots, K-1$$

### 3.4 Forward Direction Alignment

We estimate the forward direction $\hat{f}$ from root chain anchor positions per frame:

$$\hat{f} = \frac{(\mathbf{p}_{left} - \mathbf{p}_{right}) \times \hat{z}}{\|(\mathbf{p}_{left} - \mathbf{p}_{right}) \times \hat{z}\|}$$

The yaw angle $\psi = \text{atan2}(\hat{f}_y, \hat{f}_x)$ defines $R_{fwd} = R_z(-\psi)$, updated every frame to track large turns.

### 3.5 Joint Initialization

Each joint is initialized to the weighted midpoint of its limits:

$$\theta^{init}_j = \alpha \cdot \frac{\theta^{lo}_j + \theta^{hi}_j}{2}, \quad \alpha = 0.2$$

This shifts joints away from limit boundaries (where $\theta^{init} = 0$ may be) while staying near the natural rest pose. No per-robot tuning or data dependency — only joint limits from the URDF/MJCF.

Empirically, $\alpha = 0.2$ yields 5.4% improvement in $E_{mpbpe}$ over $\alpha = 0$ (pure zero init).

### 3.6 Chain-level Length Scaling

For cross-morphology retargeting, each chain's target positions are scaled by the ratio of target-to-source chain lengths:

$$s_c = \frac{\sum_k \|\mathbf{r}^{tgt}_{k+1} - \mathbf{r}^{tgt}_k\|}{\sum_k \|\mathbf{p}^{src}_{k+1} - \mathbf{p}^{src}_k\|}$$

Target positions are constructed relative to the chain start:

$$\mathbf{p}^{tgt}_i = \mathbf{a}^{tgt} + s_c \cdot (\mathbf{p}^{src}_i - \mathbf{p}^{src}_0)$$

where $\mathbf{a}^{tgt}$ is the target chain's anchor position from Step 1, preserving the target robot's structural layout from its rest pose.

### 3.7 Method A: Chain LM (Optimization-only)

#### 3.7.1 Step 1: Root Orientation + Anchor Alignment

**Decision variables:** root yaw $\psi$ and waist chain joint angles $\boldsymbol{\theta}_w$:

$$\mathbf{x}_1 = [\psi, \boldsymbol{\theta}_w] \in \mathbb{R}^{1 + n_w}$$

**Objective:** Align chain anchor points to their **rest-pose relative positions** (preserving target robot structure):

$$\min_{\mathbf{x}_1} \sum_{i=1}^{M} \left\| \text{FK}(\mathbf{x}_1)_{a_i} - (\mathbf{p}^{src}_{root} + \boldsymbol{\delta}^{rest}_i) \right\|^2$$

where $\boldsymbol{\delta}^{rest}_i$ is the $i$-th chain's anchor offset from root in the target robot's rest pose. Solved with trust-region reflective LM (`max_nfev=80`).

#### 3.7.2 Step 2: Per-Chain Shape Matching

For each chain $C^t_j$ (in tree order), with joint angles $\boldsymbol{\theta}_j \in \mathbb{R}^{n_j}$:

**Residual vector:**

$$\mathbf{r} = \begin{bmatrix} \mathbf{r}_{pos} \\ \mathbf{r}_{bend} \\ \mathbf{r}_{reg} \end{bmatrix}$$

**Position matching** ($3 \cdot n_{mapped}$ residuals):

$$\mathbf{r}_{pos,i} = \text{FK}(\boldsymbol{\theta}_j)_{b_i} - \mathbf{p}^{tgt}_i$$

**Bend direction constraint** ($3 \cdot n_{bend}$ residuals):

$$\hat{n}_i = \frac{(\mathbf{p}_{i-1} - \mathbf{p}_i) \times (\mathbf{p}_{i+1} - \mathbf{p}_i)}{\|(\mathbf{p}_{i-1} - \mathbf{p}_i) \times (\mathbf{p}_{i+1} - \mathbf{p}_i)\|}$$

$$\mathbf{r}_{bend,i} = w_{bend} \cdot (\hat{n}^{tgt}_i \times \hat{n}^{src}_i), \quad w_{bend} = 0.3$$

**Temporal regularization** ($n_j$ residuals):

$$\mathbf{r}_{reg} = w_{reg} \cdot (\boldsymbol{\theta}_j - \boldsymbol{\theta}^{prev}_j), \quad w_{reg} = 0.05$$

#### 3.7.3 Analytical Jacobian

**Position block:** $\mathbf{J}_{pos}[3i:3(i+1), k] = \texttt{mj\_jacBody}(\text{model}, \text{data}, b_i)[:, \text{dofadr}(k)]$

**Bend block:** Numerical differentiation ($\epsilon = 10^{-6}$).

**Regularization block:** $\mathbf{J}_{reg} = w_{reg} \cdot \mathbf{I}_{n_j}$.

Solved via trust-region reflective LM with `max_nfev=30` per chain.

### 3.8 Method B: Chain LM-NN (Learned IK)

An MLP is trained to directly predict joint angles from normalized chain shapes, replacing iterative LM at inference.

#### 3.8.1 Training Data Generation

Chain LM (Method A) is executed offline on LaFAN1 to collect supervised pairs:

$$\mathcal{D} = \{(S_i, \boldsymbol{\tau}_i, \boldsymbol{\theta}_i)\}_{i=1}^{N}$$

- **Source**: LaFAN1 dataset (30 BVH files, 300 frames each)
- **Targets**: 2 robots (Unitree G1, Fourier N1), 4 chains each
- **Total**: 72,000 samples

#### 3.8.2 Model Architecture

| Component | Dimensions |
|-----------|-----------|
| Input | $\text{concat}(S, \boldsymbol{\tau}, \boldsymbol{\theta}^{prev}) = 61$ |
| Shape encoder | $24 \to 256 \to 256$ (GELU) |
| Descriptor encoder | $25 \to 256 \to 256$ (GELU) |
| Previous angles encoder | $12 \to 256$ (GELU) |
| Fusion | $768 \to 256 \to 256 \to 256 \to 12$ (GELU + Tanh) |

Output $\hat{\boldsymbol{\theta}} \in [-1, 1]^{n_{max}}$, scaled to joint limits.

#### 3.8.3 Training

**Loss:** Masked MSE on valid joints:

$$\mathcal{L} = \frac{\sum_{k} m_k (\hat{\theta}_k - \theta^*_k)^2}{\sum_k m_k}$$

**Hyperparameters:** AdamW ($\text{lr}=10^{-3}$), cosine annealing, batch 256, 300 epochs.

**Result:** Validation angle error $\approx 3.2°$.

### 3.9 Method C: Hybrid (NN Warm Start + Chain LM)

The NN prediction is injected as `prev_qpos` into Chain LM's warm start:

$$\text{For each frame } t:$$

1. **NN predict**: $\hat{\boldsymbol{\theta}}_t = f_\phi(S_t, \boldsymbol{\tau}, \hat{\boldsymbol{\theta}}_{t-1})$ per chain
2. **Inject**: Set $\texttt{prev\_qpos} \leftarrow \hat{\boldsymbol{\theta}}_t$
3. **Chain LM**: Full Step 1 + Step 2, starting from NN initial guess

This preserves all Chain LM properties (root yaw, waist, bend direction, joint limits) while benefiting from NN initialization for per-chain angles.

### 3.10 Comparison of Methods

| Property | GMR IK | Chain LM | Chain LM-NN | Hybrid |
|----------|:------:|:--------:|:-----------:|:------:|
| Per-pair config | Required | **None** | **None** | **None** |
| Root yaw | IK solver | LM Step 1 | Fixed | LM Step 1 |
| Waist/torso | IK solver | LM Step 1 | Not trained | LM Step 1 |
| Limb chains | IK solver | LM Step 2 | NN forward | NN + LM |
| Joint limits | Crash-prone | Bounded LM | Implicit (Tanh) | Bounded LM |
| Bend direction | N/A | Cross-product | N/A | Cross-product |
| Training required | No | No | Yes (offline) | Yes (offline) |
| New robot | New JSON | Immediate | Retrain | Retrain |

## 4. Evaluation

### 4.1 Metrics

Following GMR (Ze et al., ICRA 2026):

**Global Mean Position Body Part Error:**

$$E_{g	ext{-}mpbpe} = rac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \left\| \mathbf{p}^{src}_{t,k} - \mathbf{p}^{tgt}_{t,k} ight\|$$

**Root-relative Mean Position Body Part Error:**

$$E_{mpbpe} = rac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \left\| (\mathbf{p}^{src}_{t,k} - \mathbf{p}^{src}_{t,root}) - (\mathbf{p}^{tgt}_{t,k} - \mathbf{p}^{tgt}_{t,root}) ight\|$$

### 4.2 Experimental Setup

- **Source**: LaFAN1 dataset (29 motion types, 1 representative file each, full frames)
- **Total frames**: 180,806
- **Target**: Unitree G1 (29 DOF humanoid)
- **GMR IK solver**: daqp
- **NN training**: 30 BVH x 300 frames x 2 robots = 72,000 samples
- **GPU**: NVIDIA RTX 5090 (34 GB VRAM)

### 4.3 Results

#### 4.3.1 Overall Comparison (29 motions, 180,806 frames)

| Method | $E_{g	ext{-}mpbpe}$ (m) | $E_{mpbpe}$ (m) | Stability | Config |
|--------|:---:|:---:|:---:|:---:|
| **GMR IK (daqp)** | **0.0828** | **0.0916** | **100%** | Per-pair JSON |
| **Chain LM (ours)** | 0.1175 | 0.1175 | **100%** | **None** |

Average gap: **+2.6 cm** (Chain LM vs GMR IK).

#### 4.3.2 Per-Motion Comparison ($E_{mpbpe}$, meters)

| Motion | Chain LM | GMR IK | Diff |
|--------|:---:|:---:|:---:|
| walk1 | 0.101 | 0.087 | +1.4cm |
| walk2 | 0.103 | 0.088 | +1.5cm |
| walk3 | 0.109 | 0.089 | +2.0cm |
| walk4 | 0.105 | 0.086 | +1.8cm |
| run1 | 0.118 | 0.093 | +2.5cm |
| run2 | 0.103 | 0.093 | +1.1cm |
| sprint1 | 0.107 | 0.092 | +1.5cm |
| dance1 | 0.112 | 0.092 | +2.1cm |
| dance2 | 0.107 | 0.093 | +1.4cm |
| fight1 | 0.109 | 0.094 | +1.5cm |
| fightAndSports1 | 0.114 | 0.092 | +2.2cm |
| jumps1 | 0.104 | 0.091 | +1.3cm |
| aiming1 | 0.110 | 0.088 | +2.2cm |
| aiming2 | 0.110 | 0.090 | +2.0cm |
| obstacles1 | 0.102 | 0.089 | +1.3cm |
| obstacles2 | 0.109 | 0.089 | +2.1cm |
| obstacles3 | 0.104 | 0.089 | +1.5cm |
| obstacles4 | **0.098** | 0.091 | **+0.7cm** |
| obstacles5 | 0.140 | 0.098 | +4.2cm |
| obstacles6 | 0.106 | 0.092 | +1.4cm |
| push1 | 0.105 | 0.094 | +1.1cm |
| pushAndFall1 | 0.109 | 0.089 | +2.0cm |
| pushAndStumble1 | 0.113 | 0.093 | +2.0cm |
| multipleActions1 | 0.121 | 0.093 | +2.8cm |
| fallAndGetUp1 | 0.161 | 0.093 | +6.8cm |
| fallAndGetUp2 | 0.147 | 0.097 | +5.0cm |
| fallAndGetUp3 | 0.141 | 0.090 | +5.1cm |
| ground1 | 0.168 | 0.091 | +7.7cm |
| ground2 | 0.172 | 0.105 | +6.7cm |

#### 4.3.3 Per-Chain Breakdown (Chain LM, aiming1)

| Chain | $E_{mpbpe}$ (m) |
|-------|:-----------:|
| Left leg | 0.106 |
| Right leg | 0.109 |
| Waist/torso | 0.231 |
| Left arm | 0.065 |
| Right arm | 0.066 |

#### 4.3.4 Per-Body Breakdown (GMR IK, aiming1)

| Body | $E_{mpbpe}$ (m) |
|------|:-----------:|
| pelvis | 0.000 |
| hip_yaw (L/R) | 0.243 |
| knee (L/R) | 0.077 |
| ankle (L/R) | 0.017 |
| shoulder_yaw (L/R) | 0.184 |
| elbow (L/R) | 0.055 |
| wrist (L/R) | 0.068 |

#### 4.3.5 Joint Initialization Sweep

| $lpha$ | $E_{mpbpe}$ (m) |
|:---:|:---:|
| 0.0 (qpos0) | 0.1168 |
| 0.1 | 0.1227 |
| **0.2** | **0.1105** |
| 0.3 | 0.1106 |
| 0.4 | 0.1125 |
| 0.5 | 0.1177 |
| 1.0 (midpoint) | 0.1280 |

#### 4.3.6 NN Training Performance

| Stage | Value |
|-------|-------|
| Training samples | 72,000 |
| Val angle error | ~3.2 deg |
| Training time | ~2 min (RTX 5090) |

### 4.4 Analysis

**GMR IK (daqp)** achieves the lowest error ($E_{mpbpe} = 0.092$ m) with 100% stability when using the daqp solver. It requires per-pair JSON configuration with manually tuned position/rotation offsets and scale factors.

**Chain LM** achieves 100% stability with no per-pair configuration, at +2.6 cm average gap from GMR IK. Key observations:
- **Locomotion** (walk, run, sprint): 1-2 cm gap -- near GMR quality
- **Dynamic motions** (dance, fight, jumps, obstacles): 1-3 cm gap
- **Ground interactions** (fallAndGetUp, ground): 5-8 cm gap -- the main weakness, due to under-constrained waist chain (23.1 cm error) and lack of ground contact modeling
- **Best case** (obstacles4): only 0.7 cm gap from GMR IK

**Chain-level auto-scaling** (Section 3.6) computes per-chain length ratios from rest pose, replacing manual scale configuration. Combined with joint limit midpoint initialization ($lpha = 0.2$), this yields a fully automatic pipeline requiring only the target robot's URDF/MJCF.

## 5. Future Works

### 5.1 Universal Chain IK via Synthetic Data

Current NN is robot-specific (trained on G1 + N1). A universal model would train on synthetic random chains with:
- Random link length ratios (Dirichlet distribution)
- Random joint axes ($x$, $y$, or $z$ per joint)
- Random joint limits
- Unreachable targets (bounded LM for closest feasible)

Preliminary experiments with 200K synthetic samples from 5,000 chain configs showed severe overfitting (train ~2°, val ~29°), indicating the need for more sophisticated architectures or massively larger datasets (5M+ samples currently generating).

### 5.2 Cross-Morphology Retargeting

The chain decomposition naturally extends to non-humanoid skeletons (quadrupeds, manipulators). Preliminary experiments with a quadruped+arm robot (Unitree Go2 + HopeJR arm) revealed key challenges:

- **Graph topology mapping**: When source has $M$ chains and target has $N$ chains ($M \neq N$), chains must be grouped by role. Individual chain matching with automatic grouping (same source $\to$ same group) enables topology-aware position interpolation.
- **Chain start position alignment**: Target chain anchors should preserve the robot's rest-pose structure rather than matching source absolute positions. Using rest-pose offsets: $\mathbf{a}^{tgt}_i = \mathbf{p}^{src}_{root} + \boldsymbol{\delta}^{rest}_i$.
- **Per-chain length scaling**: Cross-morphology requires per-chain scale ratios ($s_c = \ell_{tgt} / \ell_{src}$) since global height ratios are meaningless.

### 5.3 Chain-normalized Latent Space

The chain decomposition provides a skeleton-agnostic representation for learned retargeting:

$$\text{Motion}_{any} \xrightarrow{\text{chain decomp.}} \{S_i\} \xrightarrow{\text{Flow}(\text{cond}=\tau_{tgt})} \{S'_i\} \xrightarrow{\text{Chain LM}} \boldsymbol{\theta}_{tgt}$$

Preliminary experiments:
- **VQ-VAE**: Codebook collapse (perplexity 2.9/512). Needs FSQ or codebook reset.
- **Direct Flow**: Low shape error (0.006) but poor end-to-end (0.182 m) due to denormalization losses.

### 5.4 Integration with RL Policies

Retargeted motion serves as reference trajectories for RL-based whole-body control. Joint optimization of retargeting quality and policy success rate could improve both.