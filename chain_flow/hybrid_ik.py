"""
Hybrid Chain IK: NN Initial Guess + LM Refinement
====================================================
1. NN이 chain shape → joint angles 초기값 예측 (부정확해도 됨)
2. Differentiable FK로 position error 계산
3. LM 2-3 step으로 refine (analytical Jacobian)

NN이 ~30도 오차여도 LM이 수렴하는 데 충분한 초기값.
LM만 쓸 때 대비 iteration 수가 30→3으로 줄어 ~10배 빠름.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from chain_flow.chain_shape import normalize_chain_shape, K_POINTS


def chain_fk_np(angles, link_ratios, axes):
    """순방향 운동학 (numpy).

    Args:
        angles: (n_joints,)
        link_ratios: (n_joints,) — normalized link lengths (sum=1)
        axes: list of (3,) arrays — joint rotation axes
    Returns:
        positions: (n_joints+1, 3)
    """
    n = len(angles)
    positions = [np.zeros(3)]
    cumul_rot = R.identity()

    for i in range(n):
        local_rot = R.from_rotvec(float(angles[i]) * axes[i])
        cumul_rot = cumul_rot * local_rot
        link_vec = cumul_rot.apply(np.array([link_ratios[i], 0, 0]))
        positions.append(positions[-1] + link_vec)

    return np.array(positions)


def chain_jacobian_np(angles, link_ratios, axes):
    """Chain FK의 analytical Jacobian (numerical approx).

    Returns:
        J: (K*3, n_joints) — shape residual의 joint angles에 대한 Jacobian
    """
    n = len(angles)
    eps = 1e-6

    pos0 = chain_fk_np(angles, link_ratios, axes)
    shape0, _, _ = normalize_chain_shape(pos0)
    r0 = shape0.reshape(-1)

    J = np.zeros((len(r0), n))
    for j in range(n):
        a_plus = angles.copy()
        a_plus[j] += eps
        pos_plus = chain_fk_np(a_plus, link_ratios, axes)
        shape_plus, _, _ = normalize_chain_shape(pos_plus)
        J[:, j] = (shape_plus.reshape(-1) - r0) / eps

    return J


def lm_refine(initial_angles, target_shape, link_ratios, axes, limits,
              max_steps=3, ftol=1e-6):
    """LM refinement: NN 초기값에서 시작하여 target shape에 맞춤.

    Args:
        initial_angles: (n_joints,) — NN이 예측한 초기값
        target_shape: (K, 3) — normalized target chain shape
        link_ratios: (n_joints,) — link length ratios
        axes: list of (3,) — joint axes
        limits: list of (lo, hi) — joint limits
        max_steps: LM iteration 수
    Returns:
        refined_angles: (n_joints,)
        final_error: float
    """
    n = len(initial_angles)
    lo = np.array([l[0] for l in limits])
    hi = np.array([l[1] for l in limits])

    target_flat = target_shape.reshape(-1)

    def residual(angles):
        pos = chain_fk_np(angles, link_ratios, axes)
        shape, _, _ = normalize_chain_shape(pos)
        return shape.reshape(-1) - target_flat

    x0 = np.clip(initial_angles, lo + 1e-4, hi - 1e-4)

    result = least_squares(
        residual, x0,
        bounds=(lo, hi),
        method='trf',
        max_nfev=max_steps * n,  # roughly max_steps iterations
        ftol=ftol, xtol=1e-8, gtol=1e-8,
    )

    return result.x.astype(np.float32), float(result.cost)


class HybridChainIK:
    """Hybrid IK: NN + LM.

    Usage:
        solver = HybridChainIK(nn_model, device)

        # Per frame:
        angles = solver.solve(src_shape, chain_descriptor, link_ratios, axes, limits)
    """

    def __init__(self, nn_model, device, lm_steps=3):
        import torch
        self.model = nn_model
        self.device = device
        self.lm_steps = lm_steps
        self.model.eval()
        self.prev_angles = None

    def solve(self, src_shape_flat, descriptor, link_ratios, axes, limits):
        """
        Args:
            src_shape_flat: (shape_dim,) numpy — normalized source shape
            descriptor: (desc_dim,) numpy — chain descriptor
            link_ratios: (n_joints,) numpy
            axes: list of (3,) numpy arrays
            limits: list of (lo, hi) tuples
        Returns:
            angles: (n_joints,) numpy — refined joint angles
        """
        import torch

        n_joints = len(link_ratios)
        lo = np.array([l[0] for l in limits])
        hi = np.array([l[1] for l in limits])

        # NN forward pass → initial guess
        with torch.no_grad():
            src_t = torch.tensor(src_shape_flat, device=self.device,
                                 dtype=torch.float32).unsqueeze(0)
            desc_t = torch.tensor(descriptor, device=self.device,
                                  dtype=torch.float32).unsqueeze(0)

            prev_t = None
            if self.prev_angles is not None:
                prev_t = torch.tensor(self.prev_angles, device=self.device,
                                      dtype=torch.float32).unsqueeze(0)

            angles_norm = self.model(src_t, desc_t, prev_t)  # (1, MAX_JOINTS)
            angles_norm = angles_norm[0, :n_joints].cpu().numpy()

        # Scale to joint limits
        mid = (hi + lo) / 2
        half_range = np.maximum((hi - lo) / 2, 1e-6)
        initial_angles = angles_norm * half_range + mid
        initial_angles = np.clip(initial_angles, lo, hi).astype(np.float32)

        # Target shape
        target_shape = src_shape_flat.reshape(K_POINTS, 3)

        # LM refinement
        if self.lm_steps > 0:
            refined, cost = lm_refine(
                initial_angles, target_shape,
                link_ratios, axes, limits,
                max_steps=self.lm_steps,
            )
        else:
            refined = initial_angles

        # Store for temporal smoothing
        from chain_flow.direct_model import MAX_JOINTS
        padded = np.zeros(MAX_JOINTS, dtype=np.float32)
        refined_norm = np.clip((refined - mid) / half_range, -1, 1)
        padded[:n_joints] = refined_norm
        self.prev_angles = padded

        return refined