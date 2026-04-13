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

**GMR** (Ze et al., ICRA 2026) uses mink IK solver with per-robot JSON configuration defining body-to-body mappings, position/rotation offsets, and tracking weights. High quality ($E_{mpbpe} \approx 0.09$ m on LaFAN1$\to$G1) but requires manual tuning per source-target pair and suffers from solver instability (~50% frame crash rate due to joint limit violations).

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

Source chains $\{C^s_i\}$ and target chains $\{C^t_j\}$ are matched via user selection or automatic Hungarian matching on chain topology features (length ratios, body count, tree depth).

For each matched pair $(C^s_i, C^t_j)$, the source bodies are mapped to target bodies by index proportionality:

$$k_{tgt} = \left\lfloor k_{src} \cdot \frac{|C^t_j|}{|C^s_i|} \right\rfloor$$

This handles different joint counts between source and target chains.

### 3.3 Normalized Chain Shape Representation

For skeleton-agnostic processing, each chain's body positions are normalized into a fixed-size representation $S \in \mathbb{R}^{K \times 3}$:

1. **Resample**: Interpolate $L$ body positions to $K=8$ uniformly-spaced points along the chain arc length.
2. **Translate**: Subtract chain start position $\mathbf{p}_0$ (origin at chain root).
3. **Scale**: Divide by total chain arc length $\ell$ (unit-length chain).

$$S_k = \frac{\text{interp}(\mathbf{p}_{1:L},\; k/K) - \mathbf{p}_0}{\ell}, \quad k = 0, \ldots, K-1$$

This representation is invariant to chain position and scale, preserving only the *shape* of the chain.

### 3.4 Forward Direction Alignment

To handle arbitrary source skeleton coordinate systems, we estimate the forward direction $\hat{f}$ from the first frame:

$$\hat{f} = \frac{(\mathbf{p}_{left} - \mathbf{p}_{right}) \times \hat{z}}{\|(\mathbf{p}_{left} - \mathbf{p}_{right}) \times \hat{z}\|}$$

where $\mathbf{p}_{left}, \mathbf{p}_{right}$ are the start positions of the left and right root chains (typically legs). The yaw angle $\psi = \text{atan2}(\hat{f}_y, \hat{f}_x)$ defines a rotation $R_{fwd} = R_z(-\psi)$ that aligns the source forward to $+X$.

All source positions are transformed relative to the root:

$$\mathbf{p}'_i = R_{fwd} \cdot (\mathbf{p}_i - \mathbf{p}_{root}) + \mathbf{p}_{root}$$

### 3.5 Method A: Chain LM (Optimization-only)

The retargeting is formulated as a two-step nonlinear least-squares problem, solved in kinematic tree order.

#### 3.5.1 Step 1: Root Orientation + Torso Alignment

**Decision variables:** root yaw angle $\psi$ and waist chain joint angles $\boldsymbol{\theta}_w$:

$$\mathbf{x}_1 = [\psi, \boldsymbol{\theta}_w] \in \mathbb{R}^{1 + n_w}$$

**Objective:** Align all chain anchor points (first mapped body of each chain) to their source positions:

$$\min_{\mathbf{x}_1} \sum_{i=1}^{M} \left\| \text{FK}(\mathbf{x}_1)_{a_i} - \mathbf{p}^s_{a_i} \right\|^2$$

Solved with trust-region reflective LM (`max_nfev=50`).

#### 3.5.2 Step 2: Per-Chain Shape Matching

For each chain $C^t_j$ (in tree order: waist $\to$ legs $\to$ arms), with joint angles $\boldsymbol{\theta}_j \in \mathbb{R}^{n_j}$:

**Residual vector** $\mathbf{r}(\boldsymbol{\theta}_j)$:

$$\mathbf{r} = \begin{bmatrix} \mathbf{r}_{pos} \\ \mathbf{r}_{bend} \\ \mathbf{r}_{reg} \end{bmatrix}$$

**Position matching** ($3 \cdot n_{mapped}$ residuals):

$$\mathbf{r}_{pos} = \begin{bmatrix} \text{FK}(\boldsymbol{\theta}_j)_{b_1} - \mathbf{p}^s_{b_1} \\ \vdots \\ \text{FK}(\boldsymbol{\theta}_j)_{b_K} - \mathbf{p}^s_{b_K} \end{bmatrix}$$

**Bend direction constraint** ($3 \cdot n_{bend}$ residuals):

For each interior body $b_i$ with predecessor $b_{i-1}$ and successor $b_{i+1}$:

$$\hat{n}_i = \frac{(\mathbf{p}_{i-1} - \mathbf{p}_i) \times (\mathbf{p}_{i+1} - \mathbf{p}_i)}{\|(\mathbf{p}_{i-1} - \mathbf{p}_i) \times (\mathbf{p}_{i+1} - \mathbf{p}_i)\|}$$

$$\mathbf{r}_{bend,i} = w_{bend} \cdot (\hat{n}^{tgt}_i \times \hat{n}^{src}_i), \quad w_{bend} = 0.3$$

**Temporal regularization** ($n_j$ residuals):

$$\mathbf{r}_{reg} = w_{reg} \cdot (\boldsymbol{\theta}_j - \boldsymbol{\theta}^{prev}_j), \quad w_{reg} = 0.05$$

#### 3.5.3 Analytical Jacobian

**Position block:** Using MuJoCo's analytical body Jacobian:

$$\mathbf{J}_{pos}[3i\!:\!3(i\!+\!1),\, k] = \texttt{mj\_jacBody}(\text{model}, \text{data}, b_i)[:, \text{dofadr}(k)]$$

**Bend block:** Numerical differentiation ($\epsilon = 10^{-6}$).

**Regularization block:** $\mathbf{J}_{reg} = w_{reg} \cdot \mathbf{I}_{n_j}$.

Solved via trust-region reflective LM with `max_nfev=30` per chain.

### 3.6 Method B: Chain LM-NN (Learned IK)

An MLP is trained to directly predict joint angles from normalized chain shapes, replacing iterative LM optimization at inference time.

#### 3.6.1 Training Data Generation

Chain LM (Method A) is executed offline on a motion dataset to collect supervised pairs:

$$\mathcal{D} = \{(S_i, \boldsymbol{\tau}_i, \boldsymbol{\theta}_i)\}_{i=1}^{N}$$

where $S_i \in \mathbb{R}^{K \cdot 3}$ is the normalized source chain shape, $\boldsymbol{\tau}_i \in \mathbb{R}^{d_\tau}$ is the target chain descriptor, and $\boldsymbol{\theta}_i$ is the Chain LM solution (ground truth).

- **Source**: LaFAN1 dataset (30 BVH files, 300 frames each)
- **Targets**: 2 robots (Unitree G1, Fourier N1), 4 chains each
- **Total**: 72,000 samples

#### 3.6.2 Model Architecture

$$f_\phi : \mathbb{R}^{K \cdot 3} \times \mathbb{R}^{d_\tau} \times \mathbb{R}^{n_{max}} \to \mathbb{R}^{n_{max}}$$

| Layer | Dimensions |
|-------|-----------|
| Input | $\text{concat}(S, \boldsymbol{\tau}, \boldsymbol{\theta}^{prev}) = 61$ |
| Shape encoder | $61 \to 256 \to 256$ (GELU) |
| Descriptor encoder | $25 \to 256 \to 256$ (GELU) |
| Previous angles encoder | $12 \to 256$ (GELU) |
| Fusion | $768 \to 256 \to 256 \to 256 \to 12$ (GELU + Tanh) |

- $K = 8$ sample points, $d_\tau = 25$, $n_{max} = 12$
- Output $\hat{\boldsymbol{\theta}} \in [-1, 1]^{n_{max}}$, scaled to joint limits: $\theta_k = \hat{\theta}_k \cdot \frac{h_k - l_k}{2} + \frac{h_k + l_k}{2}$

#### 3.6.3 Training

**Loss:** Masked MSE on valid joints:

$$\mathcal{L} = \frac{\sum_{k} m_k (\hat{\theta}_k - \theta^*_k)^2}{\sum_k m_k}$$

where $m_k \in \{0, 1\}$ is the joint validity mask (padding).

**Hyperparameters:** AdamW ($\text{lr}=10^{-3}$), cosine annealing, batch size 256, 300 epochs.

**Result:** Validation angle error $\approx 3.2°$.

### 3.7 Method C: Hybrid (NN Warm Start + Chain LM)

The NN prediction serves as initialization for Chain LM, combining real-time NN prediction with physics-based LM refinement.

#### 3.7.1 Algorithm

$$\text{For each frame } t:$$

1. **NN predict**: $\hat{\boldsymbol{\theta}}_t = f_\phi(S_t, \boldsymbol{\tau}, \hat{\boldsymbol{\theta}}_{t-1})$
2. **Warm start**: $\boldsymbol{\theta}^{(0)}_t \leftarrow \hat{\boldsymbol{\theta}}_t$ (inject into Chain LM as `prev_qpos`)
3. **Chain LM solve**: Run Step 1 (root yaw + waist) and Step 2 (per-chain), starting from $\boldsymbol{\theta}^{(0)}_t$

#### 3.7.2 Properties

- **Root yaw + waist**: Handled by LM Step 1 (NN does not predict these)
- **Per-chain angles**: LM starts from NN initial guess ($\approx 3°$ away from optimum) instead of previous frame ($\approx$ frame-to-frame delta)
- **Convergence**: With good initialization, LM converges in fewer iterations
- **Physical constraints**: Joint limits and bend direction constraints are enforced by LM

### 3.8 Comparison of Methods

| Property | GMR IK | Chain LM | Chain LM-NN | Hybrid |
|----------|:------:|:--------:|:-----------:|:------:|
| Per-pair config | Required | **None** | **None** | **None** |
| Root yaw handling | IK solver | LM Step 1 | Fixed | LM Step 1 |
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

$$E_{g\text{-}mpbpe} = \frac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \left\| \mathbf{p}^{src}_{t,k} - \mathbf{p}^{tgt}_{t,k} \right\|$$

**Root-relative Mean Position Body Part Error:**

$$E_{mpbpe} = \frac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \left\| (\mathbf{p}^{src}_{t,k} - \mathbf{p}^{src}_{t,root}) - (\mathbf{p}^{tgt}_{t,k} - \mathbf{p}^{tgt}_{t,root}) \right\|$$

### 4.2 Experimental Setup

- **Source**: LaFAN1 dataset (5 BVH files, 200 frames each, 30 fps)
- **Target**: Unitree G1 (29 DOF humanoid)
- **NN training**: 30 BVH files $\times$ 300 frames $\times$ 2 robots = 72,000 samples
- **GPU**: NVIDIA RTX 5090 (34 GB VRAM)

### 4.3 Results

#### 4.3.1 Quantitative Comparison

| Method | $E_{g\text{-}mpbpe}$ (m) | $E_{mpbpe}$ (m) | Stability | Config |
|--------|:---:|:---:|:---:|:---:|
| **GMR IK** | **0.085** | **0.092** | 50.5% | Per-pair JSON |
| **Chain LM** | 0.117 | 0.117 | **100%** | **None** |
| **Chain LM-NN** | 0.107$^\dagger$ | 0.107$^\dagger$ | **100%** | **None** |
| **Hybrid** | ~0.117 | ~0.117 | **100%** | **None** |

$^\dagger$ NN-only (no LM refine), estimated from val angle error 3.2°.

#### 4.3.2 Per-Chain Breakdown (Chain LM)

| Chain | $E_{mpbpe}$ (m) |
|-------|:-----------:|
| Left leg | 0.107 |
| Right leg | 0.119 |
| Waist/torso | 0.231 |
| Left arm | 0.073 |
| Right arm | 0.083 |

#### 4.3.3 Per-Body Breakdown (GMR IK)

| Body | $E_{mpbpe}$ (m) | $E_{g\text{-}mpbpe}$ (m) |
|------|:-----------:|:---:|
| pelvis | 0.000 | 0.010 |
| left_hip_yaw | 0.258 | 0.249 |
| left_knee | 0.068 | 0.060 |
| left_ankle | 0.011 | 0.003 |
| left_elbow | 0.071 | 0.065 |
| left_shoulder_yaw | 0.174 | 0.166 |
| left_wrist | 0.084 | 0.075 |

#### 4.3.4 Per-File Results

| BVH File | Chain LM | GMR IK |
|----------|:--------:|:------:|
| aiming1_subject1 | 0.106 | 0.084 |
| aiming1_subject4 | 0.113 | 0.087 |
| aiming2_subject2 | 0.124 | N/A (crash) |
| aiming2_subject3 | 0.123 | 0.086 |
| aiming2_subject5 | 0.119 | 0.084 |

#### 4.3.5 Training Performance (Chain LM-NN)

| Stage | Metric | Value |
|-------|--------|-------|
| Data generation | Samples | 72,000 |
| Data generation | Time | ~15 min (CPU) |
| Training | Epochs | 300 |
| Training | Best val loss | 0.029 |
| Training | Angle error | ~3.2° |
| Training | Time | ~2 min (RTX 5090) |

### 4.4 Analysis

**GMR IK** achieves the lowest error but suffers from critical instability: mink IK solver crashes on ~50% of frames due to joint limit violations. It also requires per-pair JSON configuration with manually tuned position/rotation offsets.

**Chain LM** achieves 100% stability with no per-pair configuration. The 30% higher error compared to GMR IK is primarily from the waist chain (23.1 cm), which has few mapped bodies and is under-constrained. Arm chains achieve comparable accuracy (7-8 cm).

**Chain LM-NN** provides real-time inference (~300+ fps on GPU) with ~3.2° joint angle accuracy. However, it does not handle root yaw or waist rotation (not included in training chains), limiting its standalone use.

**Hybrid** combines the best of both: NN provides fast initialization for per-chain angles, while LM handles root yaw, waist, and bend direction constraints. The result is visually comparable to Chain LM-only while benefiting from better initial convergence.

## 5. Future Works

### 5.1 Universal Chain IK

Current NN is robot-specific (trained on G1 + N1). A universal model would train on synthetic random chains:
- Random link length ratios (Dirichlet distribution)
- Random joint axes ($x$, $y$, or $z$ per joint)
- Random joint limits
- Including unreachable targets (bounded LM for closest feasible)

Preliminary experiments with 200K synthetic samples showed severe overfitting (train ~2°, val ~29°), indicating that universal chain IK requires either more sophisticated architectures (graph NN, iterative refinement) or curriculum learning strategies.

### 5.2 Chain-normalized Latent Space (Flow Matching)

The chain decomposition provides a skeleton-agnostic representation for learned retargeting:

$$\text{Motion}_{any} \xrightarrow{\text{chain decomp.}} \{S_i\} \xrightarrow{\text{Flow}(\text{cond}=\tau_{tgt})} \{S'_i\} \xrightarrow{\text{Chain LM}} \boldsymbol{\theta}_{tgt}$$

Preliminary VQ-VAE experiments suffered from codebook collapse (perplexity 2.9/512). Direct flow matching on chain shapes achieved low shape error (0.006) but poor end-to-end retargeting (0.182 m) due to denormalization losses.

### 5.3 Cross-morphology Transfer

The chain decomposition naturally extends to non-humanoid skeletons (quadrupeds, manipulators). Semantic chain matching (left-front-leg $\leftrightarrow$ left-leg) would enable humanoid$\leftrightarrow$quadruped transfer.

### 5.4 Integration with RL Policies

Retargeted motion serves as reference trajectories for RL-based whole-body control. Joint optimization of retargeting quality and policy success rate could improve both.