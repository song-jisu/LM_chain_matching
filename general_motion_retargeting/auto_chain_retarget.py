"""
Auto Chain Retargeting
======================
설정 파일(IK config)·CLI 입력·바디 이름 규칙 없이, 소스 스켈레톤 계층과
타겟 MJCF만으로 동작하는 체인 단위 LM 리타게팅.

파이프라인
1. 타겟 분석: 키네마틱 트리 → 체인 트리. 각 체인의 관절 중심(key point)은
   인접 hinge 축의 교점으로 정의한다. 회전 관절은 축만 의미가 있고 바디 원점의
   축 방향 위치는 임의이므로(예: G1 hip_yaw_link 원점은 허벅지 중간), 원점 대신
   축 교점을 써야 로봇 간에 일관된 관절 위치를 얻는다.
2. 소스 분석: BVH 계층 → 체인 트리. 서 있는 기준 프레임을 고르고, 다리 체인의
   무릎이 전방으로 굽는다는 사실로 좌/우와 전방을 판정한다.
3. 체인 매칭: 키 정규화된 체인 시작 위치(전방·좌측·높이)의 거리 + 좌우 부호
   불일치 패널티로 비용 행렬을 만들고 Hungarian 알고리즘으로 할당한다.
4. 체인 내부 대응: 타겟 key point의 호길이 분율 구조와 가장 잘 맞는 소스 구간
   (연속 bone 부분열)을 chamfer 비용으로 고른 뒤, 분율로 보간한다.
5. 최적화: Step 1은 루트 6-DOF(또는 yaw) + 내부 체인(자식 체인을 가진 체인,
   예: 허리) 관절로 말단 체인 시작점을 맞추고, Step 2는 말단 체인별로
   위치·굽힘·시간 잔차를 bounded TRF로 푼다. 모든 Jacobian은 해석적이다.
"""

import numpy as np
import mujoco as mj
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation as R

from .params import ROBOT_XML_DICT

EZ = np.array([0.0, 0.0, 1.0])
FREE, SLIDE, HINGE = (int(mj.mjtJoint.mjJNT_FREE), int(mj.mjtJoint.mjJNT_SLIDE),
                      int(mj.mjtJoint.mjJNT_HINGE))
PLANE, MESH = int(mj.mjtGeom.mjGEOM_PLANE), int(mj.mjtGeom.mjGEOM_MESH)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _right_jacobian(w):
    """SO(3) right Jacobian: q·exp(w)의 body 각속도 = Jr(w) ẇ."""
    t = np.linalg.norm(w)
    if t < 1e-8:
        return np.eye(3) - 0.5 * _skew(w)
    K = _skew(w)
    return np.eye(3) - (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K


def _polyline_fracs(pts):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return d, d / d[-1] if d[-1] > 1e-9 else np.zeros(len(pts))


def _closest_points(a1, u1, a2, u2):
    """두 직선의 최근접점 쌍. 평행이면 None."""
    w0 = a1 - a2
    b = u1 @ u2
    den = 1 - b * b
    if den < np.sin(np.deg2rad(10)) ** 2:
        return None
    d, e = u1 @ w0, u2 @ w0
    s = (b * e - d) / den
    t = (e - b * d) / den
    return a1 + s * u1, a2 + t * u2


class AutoChainRetargeting:

    def __init__(self, tgt_robot, root_rot="full", bend="vector",
                 w_bend=0.3, w_reg=0.05, w_reg_root=0.05,
                 w_null=0.01, w_end=0.3, w_pelvis=0.5,
                 max_nfev1=50, max_nfev2=30, init_alpha=0.2,
                 match_cost_max=0.35, verbose=False, xml_path=None, **_):
        assert root_rot in ("full", "yaw")
        assert bend in ("vector", "cross", "none")
        self.root_rot, self.bend = root_rot, bend
        self.w_bend, self.w_reg, self.w_reg_root = w_bend, w_reg, w_reg_root
        self.w_null, self.w_end, self.w_pelvis = w_null, w_end, w_pelvis
        self.max_nfev1, self.max_nfev2 = max_nfev1, max_nfev2
        self.match_cost_max = match_cost_max
        self.verbose = verbose

        self.xml_file = str(xml_path or ROBOT_XML_DICT[tgt_robot])
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self.data = mj.MjData(self.model)
        m = self.model
        self.has_freejoint = m.njnt > 0 and int(m.jnt_type[0]) == FREE
        self.root_bid = int(m.jnt_bodyid[0]) if self.has_freejoint else 1
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

        self._init_qpos = m.qpos0.copy()
        for j in range(1 if self.has_freejoint else 0, m.njnt):
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                self._init_qpos[m.jnt_qposadr[j]] = np.clip(init_alpha * (lo + hi) / 2, lo, hi)
        if self.has_freejoint:
            self._init_qpos[:3] = 0
            self._init_qpos[3:7] = [1, 0, 0, 0]

        self._analyze_robot()
        self.source_ready = False
        self.prev_qpos = None

    # ==================================================================
    # 타겟 분석
    # ==================================================================

    def _kin(self, qpos, data=None):
        data = self.data if data is None else data
        data.qpos[:] = qpos
        mj.mj_kinematics(self.model, data)
        mj.mj_comPos(self.model, data)

    def _robot_height(self, data):
        m = self.model
        zs = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0 or int(m.geom_type[g]) == PLANE:
                continue
            Rg = data.geom_xmat[g].reshape(3, 3)
            c = data.geom_xpos[g] + Rg @ m.geom_aabb[g][:3]
            hz = (np.abs(Rg) @ m.geom_aabb[g][3:])[2]
            zs += [c[2] - hz, c[2] + hz]
        if not zs:
            zs = list(data.xpos[1:, 2])
        return min(zs), max(zs)

    def _analyze_robot(self):
        m, d = self.model, mj.MjData(self.model)
        self._kin(self._init_qpos, d)
        mj.mj_forward(m, d)
        zmin, zmax = self._robot_height(d)
        if self.has_freejoint:  # 지면에 세운 rest pose
            self._init_qpos[2] = -zmin
            self._kin(self._init_qpos, d)
            mj.mj_forward(m, d)
            zmin, zmax = self._robot_height(d)
        self.robot_H = zmax - zmin
        self.robot_zmin = zmin

        children = {i: [] for i in range(m.nbody)}
        for i in range(1, m.nbody):
            children[int(m.body_parentid[i])].append(i)
        body_joints = {}
        for j in range(1 if self.has_freejoint else 0, m.njnt):
            if int(m.jnt_type[j]) in (HINGE, SLIDE):
                body_joints.setdefault(int(m.jnt_bodyid[j]), []).append(j)

        memo = {}

        def has_jd(b):
            if b not in memo:
                memo[b] = b in body_joints or any(has_jd(c) for c in children[b])
            return memo[b]

        chains = []

        def trace(start, parent_ci):
            ids, b = [start], start
            while True:
                k = [c for c in children[b] if has_jd(c)]
                if len(k) == 1:
                    b = k[0]
                    ids.append(b)
                else:
                    break
            ci = len(chains)
            chains.append({'body_ids': ids, 'parent': parent_ci, 'children': []})
            if parent_ci is not None:
                chains[parent_ci]['children'].append(ci)
            for c in [c for c in children[b] if has_jd(c)]:
                trace(c, ci)

        for c in [c for c in children[self.root_bid] if has_jd(c)]:
            trace(c, None)

        root_p = d.xpos[self.root_bid].copy()
        tau = 0.05 * self.robot_H
        for ci, ch in enumerate(chains):
            ch['name'] = (f"{mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, ch['body_ids'][0])}"
                          f"->{mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, ch['body_ids'][-1])}")
            joints = [j for b in ch['body_ids'] for j in body_joints.get(b, [])]
            ch['joints'] = joints
            ch['qpa'] = [int(m.jnt_qposadr[j]) for j in joints]
            ch['dof'] = [int(m.jnt_dofadr[j]) for j in joints]
            ch['lo'] = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in joints])
            ch['hi'] = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in joints])
            ch['internal'] = len(ch['children']) > 0
            ch['has_tip'] = False
            ch['keys'] = self._chain_keypoints(ch, d, children, body_joints, tau)
            pts = np.array([k[2] for k in ch['keys']])
            ch['key_len'], ch['key_frac'] = _polyline_fracs(pts) if len(pts) > 1 else (np.zeros(1), np.zeros(1))
            p0 = pts[0] - root_p
            # 정규화 특징: (전방, 좌측, 지면으로부터 높이) / 키. 로봇은 +X 전방, +Y 좌측(MJCF 관례).
            ch['feat'] = np.array([p0[0], p0[1], pts[0][2] - zmin]) / self.robot_H
            ch['len_n'] = ch['key_len'][-1] / self.robot_H
            ch['key_zn'] = (pts[:, 2] - zmin) / self.robot_H
        self.chains = chains

        if self.verbose:
            print(f"[auto] {self.xml_file}  H={self.robot_H:.3f}  chains={len(chains)}")
            for ci, ch in enumerate(chains):
                print(f"  T{ci} {ch['name']:<45s} internal={ch['internal']} "
                      f"joints={len(ch['joints'])} keys={len(ch['keys'])} feat={np.round(ch['feat'], 2)}")

    def _chain_keypoints(self, ch, d, children, body_joints, tau):
        """인접 hinge 축 교점 → 관절 중심. (owner body, local offset, world pos) 리스트."""
        m = self.model
        hinges = [j for j in ch['joints'] if int(m.jnt_type[j]) == HINGE]
        cands = []  # (owner_bid, world_pos)
        first_ok = False
        for i, (ja, jb) in enumerate(zip(hinges[:-1], hinges[1:])):
            a1, u1, a2, u2 = d.xanchor[ja], d.xaxis[ja], d.xanchor[jb], d.xaxis[jb]
            cp = _closest_points(a1, u1, a2, u2)
            owner = int(m.jnt_bodyid[jb])
            if cp is not None and np.linalg.norm(cp[0] - cp[1]) < 0.02 * self.robot_H \
                    and np.linalg.norm(cp[1] - a2) < 0.35 * self.robot_H:
                p = 0.5 * (cp[0] + cp[1])
                first_ok = first_ok or i == 0
            else:
                p = a2.copy()
            cands.append((owner, p))
        # 첫 관절의 anchor는 축 방향 위치가 임의이므로, 첫 두 축의 교점이 있으면 쓰지 않는다
        if hinges and not first_ok:
            j0 = hinges[0]
            cands.insert(0, (int(m.jnt_bodyid[j0]), d.xanchor[j0].copy()))
        if not cands:  # slide 관절만 있는 체인
            b = ch['body_ids'][0]
            cands.append((b, d.xpos[b].copy()))

        # 연속된 가까운 후보를 하나로(가장 말단 후보를 대표로)
        keys = [cands[0]]
        for c in cands[1:]:
            if np.linalg.norm(c[1] - keys[-1][1]) < tau:
                keys[-1] = c
            else:
                keys.append(c)

        # key가 3개 미만이면(예: 팔꿈치에서 끝나는 팔, 비틀림 관절로 끝나는 N1 팔) 말단 위치
        # 정보가 없으므로 tip = 마지막 바디(및 관절 없는 자손)의 geom 꼭짓점 중 가장 먼 점을 추가
        last = ch['body_ids'][-1]
        group, stack = [last], list(children[last])
        while stack:
            b = stack.pop()
            if b not in body_joints:
                group.append(b)
                stack += children[b]
        far, far_d = None, 0.0
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in group:
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            mid = int(m.geom_dataid[g])
            if int(m.geom_type[g]) == MESH and mid >= 0:
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                local_pts = m.mesh_vert[a:a + n]
            else:
                local_pts = m.geom_aabb[g][:3] + signs * m.geom_aabb[g][3:]
            corners = d.geom_xpos[g] + local_pts @ Rg.T
            dist = np.linalg.norm(corners - keys[-1][1], axis=1)
            if dist.max() > far_d:
                far, far_d = corners[np.argmax(dist)], dist.max()
        if len(keys) < 3 and far is not None and far_d > 0.03 * self.robot_H:
            keys.append((last, far.copy()))
            ch['has_tip'] = True

        out = []
        for bid, p in keys:
            local = d.xmat[bid].reshape(3, 3).T @ (p - d.xpos[bid])
            out.append((bid, local, p))
        return out

    # ==================================================================
    # 소스 분석 + 매칭
    # ==================================================================

    def prepare(self, frames, bone_hierarchy):
        """소스 모션(프레임 dict 리스트)과 bone 계층으로 매칭·스케일을 결정."""
        bones, parents = bone_hierarchy['bones'], bone_hierarchy['parents']
        self.src_root = bones[parents.index(-1)]
        P = np.array([[f[b][0] for b in bones] for f in frames])  # (T, B, 3)
        bidx = {b: i for i, b in enumerate(bones)}

        children = {b: [] for b in bones}
        for i, p in enumerate(parents):
            if p >= 0:
                children[bones[p]].append(bones[i])
        src_chains = []

        def trace(start, parent_ci):
            ids, b = [start], start
            while len(children[b]) == 1:
                b = children[b][0]
                ids.append(b)
            ci = len(src_chains)
            src_chains.append({'bones': ids, 'parent': parent_ci})
            for c in children[b]:
                trace(c, ci)

        for c in children[self.src_root]:
            trace(c, None)

        ri = bidx[self.src_root]
        P0 = P[0]  # bone 길이는 강체이므로 길이·분율 계산은 아무 프레임이나 쓴다
        self.src_ground = float(P0[:, 2].min())
        self.ground_dz = 0.0

        # 다리: 3개 이상 bone, 끝점 평균 높이가 가장 낮은 길이 유사 쌍
        cand = [i for i, c in enumerate(src_chains) if len(c['bones']) >= 3]
        end_z = {i: P[:, bidx[src_chains[i]['bones'][-1]], 2].mean() for i in cand}
        clen = {i: _polyline_fracs(P0[[bidx[b] for b in src_chains[i]['bones']]])[0][-1] for i in cand}
        best = None
        for a in cand:
            for b in cand:
                if a < b and abs(clen[a] - clen[b]) < 0.2 * max(clen[a], clen[b]):
                    s = end_z[a] + end_z[b]
                    if best is None or s < best[0]:
                        best = (s, a, b)
        if best is None:
            raise RuntimeError("source leg chains not found")
        legA, legB = best[1], best[2]

        def knee_idx(ci):
            bs = src_chains[ci]['bones']
            fr_ = _polyline_fracs(P0[[bidx[b] for b in bs]])[1]
            return int(np.argmin(np.abs(fr_[1:-1] - 0.5))) + 1

        # 서 있는 프레임: 양다리가 곧고(엉덩이-발목 거리 ≥ 97% 다리 길이) 수직(기울기 ≤ 18°)
        def upright(ci):
            bs, k = src_chains[ci]['bones'], knee_idx(ci)
            h, kn, an = (bidx[bs[0]], bidx[bs[k]], bidx[bs[k + 1]])
            L = np.linalg.norm(P0[h] - P0[kn]) + np.linalg.norm(P0[kn] - P0[an])
            v = P[:, h] - P[:, an]
            n = np.linalg.norm(v, axis=1)
            return (n / L > 0.97) & (v[:, 2] / (n + 1e-9) > 0.95)
        stand = P[:, ri, 2] - P[:, :, 2].min(axis=1)
        st = upright(legA) & upright(legB)
        if st.sum() < 10:  # 선 자세가 거의 없는 모션: 루트 높이 95 백분위 근처로 대체
            s95 = np.percentile(stand, 95)
            st = (stand >= 0.9 * s95) & (stand <= 1.05 * s95)
        self.n_stand_frames = int(st.sum())
        fr = int(np.flatnonzero(st)[np.argmin(np.abs(stand[st] - np.median(stand[st])))])
        Pr = P[fr]

        # 좌우: 무릎 오프셋이 전방(= (A-B) x z, A가 왼쪽일 때)과 같은 방향이면 A가 왼쪽
        def knee_off(ci):
            bs, k = src_chains[ci]['bones'], knee_idx(ci)
            hip, knee, ank = P[:, bidx[bs[0]]], P[:, bidx[bs[k]]], P[:, bidx[bs[k + 1]]]
            off = knee - 0.5 * (hip + ank)
            off[:, 2] = 0
            return off, hip
        offA, hipA = knee_off(legA)
        offB, hipB = knee_off(legB)
        fwdA = np.cross(hipA - hipB, EZ)
        fwdA /= np.linalg.norm(fwdA, axis=1, keepdims=True) + 1e-9
        score = np.mean(np.sum((offA + offB) * fwdA, axis=1))
        legL, legR = (legA, legB) if score > 0 else (legB, legA)
        self.knee_score = float(score)

        # 기준 프레임의 소스 body frame
        hipL = Pr[bidx[src_chains[legL]['bones'][0]]]
        hipR = Pr[bidx[src_chains[legR]['bones'][0]]]
        fwd = np.cross(hipL - hipR, EZ)
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        left = np.cross(EZ, fwd)
        zmin = Pr[:, 2].min()
        # 머리 관절 높이 ≈ 신장의 93%. 서 있는 프레임들의 중앙값으로 추정.
        H_src = np.median(P[st, :, 2].max(axis=1) - P[st, :, 2].min(axis=1)) / 0.93
        root = Pr[ri]
        self.src_H, self.src_zmin_ref = H_src, zmin
        # 서 있는 프레임들에서, 프레임별 골반 heading 좌표계로 본
        # 노드 위치의 중앙값(기준 프레임 한 장의 자세에 덜 민감)
        Ps = P[st]
        hl_all = Ps[:, bidx[src_chains[legL]['bones'][0]]]
        hr_all = Ps[:, bidx[src_chains[legR]['bones'][0]]]
        f_all = np.cross(hl_all - hr_all, EZ)
        f_all[:, 2] = 0
        f_all /= np.linalg.norm(f_all, axis=1, keepdims=True) + 1e-9
        l_all = np.cross(EZ, f_all)
        root_h = float(np.median(stand[st]))
        for ci, c in enumerate(src_chains):
            pts = Pr[[bidx[b] for b in c['bones']]]
            rel_all = Ps[:, [bidx[b] for b in c['bones']]] - Ps[:, ri][:, None]  # (T, n, 3)
            fx = np.einsum('tnk,tk->tn', rel_all, f_all)
            ly = np.einsum('tnk,tk->tn', rel_all, l_all)
            c['node_feat'] = np.c_[np.median(fx, 0), np.median(ly, 0),
                                   np.median(rel_all[..., 2], 0) + root_h] / H_src
            c['pts_ref'] = pts
            c['len_n'] = _polyline_fracs(pts)[0][-1] / H_src
            c['internal'] = any(o['parent'] == ci for o in src_chains)
            c['is_leg'] = False
        src_chains[legL]['is_leg'] = src_chains[legR]['is_leg'] = True
        self.src_chains = src_chains

        # 체인 매칭(Hungarian)
        T = self.chains
        C = np.zeros((len(src_chains), len(T)))
        self._start_node = {}
        for i, sc in enumerate(src_chains):
            for j, tc in enumerate(T):
                # 소스 체인의 첫 두 노드 중 타겟 시작점에 더 가까운 쪽과의 거리.
                # 예: 쇄골에서 시작하는 소스 팔은 어깨 노드가 로봇 어깨 시작점과 맞는다.
                # (근위부에 짧은 마디 하나가 더 붙은 경우만 허용 — 더 먼 노드는 자세에 따라 움직인다)
                nf = sc['node_feat'][:2 if len(sc['bones']) >= 3 else 1]
                dist = np.linalg.norm(nf - tc['feat'], axis=1)
                k = int(np.argmin(dist))
                self._start_node[i, j] = k
                c = dist[k] + abs(sc['len_n'] - tc['len_n'])
                ys, yt = nf[k][1], tc['feat'][1]
                if abs(ys) > 0.02 and abs(yt) > 0.02 and np.sign(ys) != np.sign(yt):
                    c += 1.0
                if sc['is_leg'] != self._tgt_is_leg(j):
                    c += 0.5
                if sc['internal'] != tc['internal']:
                    c += 0.5
                C[i, j] = c
        rows, cols = linear_sum_assignment(C)
        self.match = {}  # tgt chain idx -> src chain idx
        for i, j in zip(rows, cols):
            if C[i, j] <= self.match_cost_max:
                self.match[j] = i
        self.match_cost = C

        # 체인 내부 대응 + 스케일
        self.tgt_maps = {}
        leg_scales, all_scales = [], []
        for j, i in self.match.items():
            tc, sc = T[j], src_chains[i]
            mp = self._map_chain(tc, sc, self._start_node[i, j])
            self.tgt_maps[j] = mp
            if mp['scale'] is not None:
                all_scales.append(mp['scale'])
                if sc['is_leg']:
                    leg_scales.append(mp['scale'])
        self.s_global = float(np.mean(leg_scales or all_scales or [1.0]))
        self.bidx = bidx
        self.src_legs = (legL, legR)
        self.tgt_legs = [j for j, i in self.match.items() if i in (legL, legR)]
        # 지면 정렬: 서 있는 프레임에서 다리 말단 대응점의 지면 위 높이(소스, 스케일 적용)를
        # 로봇 기준 자세의 같은 점 높이에 맞춘다. 사람 발목 높이를 축소해도 로봇 발목-발바닥
        # 높이와 같지 않으므로 이 보정이 없으면 발이 지면을 관통한다.
        # 지면 = 전체 프레임 최저 관절 높이의 5 백분위(장애물 위에 선 구간에 덜 민감).
        # 다리 말단 높이는 각 서 있는 프레임의 최저 관절 기준(지지면 기준)으로 잰다.
        zlow = P[:, :, 2].min(axis=1)
        self.src_ground = float(np.percentile(zlow, 5))
        dz = []
        for j in self.tgt_legs:
            b0, b1, a = self.tgt_maps[j]['interp'][-1]
            zs = (1 - a) * P[st, bidx[b0], 2] + a * P[st, bidx[b1], 2]
            dz.append((T[j]['keys'][-1][2][2] - self.robot_zmin)
                      - self.s_global * np.median(zs - zlow[st]))
        self.ground_dz = float(np.mean(dz)) if dz else 0.0

        # Step 1 앵커: 매칭된 말단 체인의 첫 key point
        self.anchors = []  # (owner_bid, local, (b0,b1,alpha))
        for j, mp in self.tgt_maps.items():
            if T[j]['internal']:
                continue
            bid, local, _ = T[j]['keys'][0]
            self.anchors.append((bid, local, mp['interp'][0]))
        self._anchor_rest_c = (np.mean([T[j]['keys'][0][2] for j in self.tgt_maps
                                        if not T[j]['internal']], axis=0)
                               if self.anchors else np.zeros(3))
        self.n_leaf_anchors = len(self.anchors)
        # 내부 체인(허리 등)의 시작점도 약한 앵커로 둔다: 두 엉덩이 점만으로는 좌우축 회전
        # (골반 pitch)이 관측되지 않아 골반 pitch와 허리 pitch가 서로 바뀌는 모호성이 생긴다.
        for j, mp in self.tgt_maps.items():
            if T[j]['internal'] and self.anchors:
                bid, local, _ = T[j]['keys'][0]
                self.anchors.append((bid, local, mp['interp'][0]))
        self.anchor_w = np.r_[np.ones(self.n_leaf_anchors),
                              np.full(len(self.anchors) - self.n_leaf_anchors, self.w_pelvis)]

        # 말단 방향: 소스 구간 끝 뒤에 bone이 더 있으면(예: 발목 뒤 발끝) 그 방향을
        # 로봇 마지막 바디의 같은 방향 벡터(geom 꼭짓점)와 맞춘다 → 발 방향(발목 pitch/yaw) 결정
        for j, mp in self.tgt_maps.items():
            mp['end_dir'] = None
            sc = src_chains[self.match[j]]
            i1 = mp['span'][1]
            if T[j]['internal'] or i1 >= len(sc['bones']) - 1 or T[j]['has_tip']:
                continue
            ds = sc['node_feat'][i1 + 1] - sc['node_feat'][i1]
            ds /= np.linalg.norm(ds) + 1e-9
            v = self._terminal_vector(T[j], ds)
            if v is not None:
                mp['end_dir'] = (v[0], v[1], (sc['bones'][i1], sc['bones'][i1 + 1]))
        self.step1_chains = [j for j, ch in enumerate(T) if ch['internal']]
        self.step2_chains = [j for j in self.tgt_maps if not T[j]['internal']
                             and len(T[j]['keys']) >= 2 and T[j]['joints']]
        self.source_ready = True
        self.prev_qpos = None
        if self.verbose:
            self.print_match()

    def _tgt_is_leg(self, j):
        """타겟 다리 판정: 말단 체인, key ≥3, 시작점이 아래쪽이고 끝점이 지면 근처."""
        ch = self.chains[j]
        if ch['internal'] or len(ch['keys']) < 3:
            return False
        end_h = (ch['keys'][-1][2][2] - self.robot_zmin) / self.robot_H
        return end_h < 0.15 and ch['feat'][2] < 0.65

    def _terminal_vector(self, ch, direction):
        """rest pose에서 마지막 key → 마지막 바디 geom 꼭짓점 중 direction 방향으로 가장 먼 점.
        (owner bid, local 단위벡터) 반환."""
        m, d = self.model, mj.MjData(self.model)
        self._kin(self._init_qpos, d)
        mj.mj_forward(m, d)
        bid, local, _ = ch['keys'][-1]
        p0 = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local
        best, best_s = None, 0.0
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != bid:
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            mid = int(m.geom_dataid[g])
            if int(m.geom_type[g]) == MESH and mid >= 0:
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                lp = m.mesh_vert[a:a + n]
            else:
                sg = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                lp = m.geom_aabb[g][:3] + sg * m.geom_aabb[g][3:]
            w = d.geom_xpos[g] + lp @ Rg.T - p0
            sc = w @ direction
            k = int(np.argmax(sc))
            if sc[k] > best_s:
                best, best_s = w[k], sc[k]
        if best is None or np.linalg.norm(best) < 0.02 * self.robot_H:
            return None
        v = d.xmat[bid].reshape(3, 3).T @ (best / np.linalg.norm(best))
        return bid, v

    def _map_chain(self, tc, sc, i0):
        """소스 구간 [i0, i1] 선택 → key별 보간 (b0, b1, alpha).
        i0: 타겟 체인 시작점에 가장 가까운 소스 노드(매칭 단계에서 결정, 몸통 위의 점).
        i1: 기본은 소스 체인 끝. 다리만 서 있는 자세의 지면 높이로 발목/발끝을 고른다.
        tip(관절 이후 말단)이 있으면 tip은 i1에, 나머지 key는 [i0, i1-1]에 분율로 대응."""
        bones, pts = sc['bones'], sc['pts_ref']
        nb = len(bones)
        keys = tc['keys']
        i0 = min(i0, nb - 1)
        i1 = nb - 1
        if sc['is_leg'] and nb - 1 > i0 + 1:
            # 다리 끝: 로봇 마지막 key 분절(정강이)과 방향이 가장 비슷한 소스 분절로 끝낸다.
            # 서 있는 프레임 중앙값 기준이라 발목(수직 정강이 끝)과 발끝(수평 발등 끝)이 구분된다.
            kp = np.array([k[2] for k in keys[:len(keys) - int(tc['has_tip'])]])
            rd = kp[-1] - kp[-2]
            rd /= np.linalg.norm(rd) + 1e-9
            # 허벅지·정강이는 둘 다 수직이므로 분율 구조(chamfer) 비용을 더해 구분한다.
            nf = sc['node_feat']
            tf_ = _polyline_fracs(kp)[1]
            costs = []
            for e_ in range(i0 + 1, nb):
                sd = nf[e_] - nf[e_ - 1]
                cos = rd @ (sd / (np.linalg.norm(sd) + 1e-9))
                sf_ = _polyline_fracs(pts[i0:e_ + 1])[1]
                ch_ = np.mean([np.min(np.abs(sf_ - f)) for f in tf_]) + \
                    np.mean([np.min(np.abs(tf_ - f)) for f in sf_])
                costs.append(ch_ + (1 - cos))
            i1 = i0 + 1 + int(np.argmin(costs))
        real = len(keys) - 1 if (tc['has_tip'] and i1 > i0) else len(keys)
        e = i1 - 1 if real < len(keys) else i1
        rpts = np.array([k[2] for k in keys[:real]])
        rlen, rf = _polyline_fracs(rpts) if real > 1 else (np.zeros(1), np.zeros(1))
        slen, sf = _polyline_fracs(pts[i0:e + 1]) if e > i0 else (np.zeros(1), np.zeros(1))
        sub = bones[i0:e + 1]
        interp = []
        for f in rf:
            if e == i0:
                interp.append((sub[0], sub[0], 0.0))
                continue
            k = int(np.argmin(np.abs(sf - f)))
            if abs(sf[k] - f) < 0.1:
                interp.append((sub[k], sub[k], 0.0))
                continue
            k = int(np.clip(np.searchsorted(sf, f) - 1, 0, len(sf) - 2))
            a = (f - sf[k]) / max(sf[k + 1] - sf[k], 1e-9)
            interp.append((sub[k], sub[k + 1], float(np.clip(a, 0, 1))))
        if real < len(keys):
            interp.append((bones[i1], bones[i1], 0.0))
        scale = rlen[-1] / slen[-1] if (e > i0 and rlen[-1] > 1e-6) else None
        return {'span': (i0, i1), 'interp': interp, 'scale': scale}

    def print_match(self):
        print(f"[auto] knee score={self.knee_score:+.3f}  s_global={self.s_global:.3f}")
        for j, ch in enumerate(self.chains):
            if j in self.match:
                sc = self.src_chains[self.match[j]]
                mp = self.tgt_maps[j]
                pairs = []
                for (bid, _, _), (b0, b1, a) in zip(ch['keys'], mp['interp']):
                    bn = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid)
                    pairs.append(f"{bn}<-{b0}" if a == 0 else f"{bn}<-{b0}:{b1}@{a:.2f}")
                sc_s = f"{mp['scale']:.2f}" if mp['scale'] else "-"
                print(f"  T{j} {ch['name']:<42s} <- S[{sc['bones'][0]}..{sc['bones'][-1]}] "
                      f"cost={self.match_cost[self.match[j], j]:.2f} scale={sc_s}")
                print(f"       {'  '.join(pairs)}")
            else:
                print(f"  T{j} {ch['name']:<42s} (unmatched)")

    def match_summary(self):
        """논문/로그용 매칭 결과 dict."""
        out = []
        for j, ch in enumerate(self.chains):
            i = self.match.get(j)
            out.append({'target': ch['name'], 'internal': ch['internal'],
                        'source': None if i is None else
                        f"{self.src_chains[i]['bones'][0]}..{self.src_chains[i]['bones'][-1]}",
                        'interp': None if i is None else self.tgt_maps[j]['interp']})
        return out

    # ==================================================================
    # Retarget
    # ==================================================================

    def _src(self, hd, ip):
        b0, b1, a = ip
        p0 = np.asarray(hd[b0][0])
        return p0 if a == 0 else (1 - a) * p0 + a * np.asarray(hd[b1][0])

    def _S(self, p):
        """소스 world → 타겟 world 전역 상사변환(지면 기준 균일 스케일)."""
        q = np.asarray(p, dtype=float) * self.s_global
        q[2] = (p[2] - self.src_ground) * self.s_global + self.ground_dz
        return q

    def _transform(self, hd):
        """이 프레임의 소스 world → 타겟 world 변환.
        부유 베이스: 전역 상사변환 S. 고정 베이스: 소스를 앵커 중심·다리 heading 좌표계로
        옮긴 뒤 로봇 rest 앵커 중심에 붙인다(로봇이 따라 걸을 수 없으므로)."""
        if self.has_freejoint:
            return self._S
        L, Rr = self.src_legs
        hl = np.asarray(hd[self.src_chains[L]['bones'][0]][0])
        hr = np.asarray(hd[self.src_chains[Rr]['bones'][0]][0])
        f = np.cross(hl - hr, EZ)
        Rinv = R.from_euler('z', -np.arctan2(f[1], f[0]))
        c_src = np.mean([self._src(hd, ip) for _, _, ip in self.anchors[:self.n_leaf_anchors]], axis=0)
        c_tgt, s = self._anchor_rest_c, self.s_global
        return lambda p: c_tgt + s * Rinv.apply(np.asarray(p) - c_src)

    def _point(self, bid, local):
        return self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ local

    def _point_jac(self, bid, local, dofs):
        p = self._point(bid, local)
        self._jacp[:] = 0
        mj.mj_jac(self.model, self.data, self._jacp, self._jacr, p, bid)
        return self._jacp[:, dofs]

    def _initial_root(self, hd):
        """첫 프레임: 다리 시작점 중점과 무릎 기반 전방으로 루트 초기화."""
        q = self._init_qpos.copy()
        if not self.has_freejoint or len(self.tgt_legs) < 2:
            return q
        L, Rr = self.src_legs
        hl = np.asarray(hd[self.src_chains[L]['bones'][0]][0])
        hr = np.asarray(hd[self.src_chains[Rr]['bones'][0]][0])
        f = np.cross(hl - hr, EZ)
        yaw = np.arctan2(f[1], f[0])
        rest_mid = np.mean([self.chains[j]['keys'][0][2] for j in self.tgt_legs], axis=0)
        off = self._init_qpos[:3] - rest_mid
        q[:3] = self._S(0.5 * (hl + hr)) + R.from_euler('z', yaw).apply(off)
        q[3:7] = R.from_euler('z', yaw).as_quat(scalar_first=True)
        return q

    def retarget(self, human_data):
        assert self.source_ready, "call prepare(frames, bone_hierarchy) first"
        hd = human_data
        qpos = self.prev_qpos.copy() if self.prev_qpos is not None else self._initial_root(hd)
        first = self.prev_qpos is None
        self._step1(qpos, hd, first)
        for j in self.step2_chains:
            self._step2(qpos, hd, j, first)
        self._kin(qpos)
        self.prev_qpos = qpos.copy()
        Tf = self._transform(hd)  # 뷰어 마커용(소스를 타겟 좌표·스케일로)
        self.scaled_human_data = {b: [Tf(v[0]), v[1]] for b, v in hd.items()}
        return qpos

    # ---------------- Step 1: root + 내부 체인 ----------------

    def _step1(self, qpos, hd, first):
        if not self.anchors:
            return
        m = self.model
        tq = [a for j in self.step1_chains for a in self.chains[j]['qpa']]
        td = [a for j in self.step1_chains for a in self.chains[j]['dof']]
        tlo = np.concatenate([self.chains[j]['lo'] for j in self.step1_chains] or [np.zeros(0)])
        thi = np.concatenate([self.chains[j]['hi'] for j in self.step1_chains] or [np.zeros(0)])
        Tf = self._transform(hd)
        targets = np.array([Tf(self._src(hd, ip)) for _, _, ip in self.anchors])
        free = self.has_freejoint
        R_ref = R.from_quat(qpos[3:7], scalar_first=True) if free else None
        nr = 0
        if free:
            nr = 6 if self.root_rot == "full" else 4
            if self.root_rot == "full":
                x0 = np.r_[qpos[:3], np.zeros(3), qpos[tq]]
            else:
                yaw0 = R_ref.as_euler('ZYX')[0]
                x0 = np.r_[qpos[:3], yaw0, qpos[tq]]
            lo = np.r_[np.full(nr, -np.inf), tlo]
            hi = np.r_[np.full(nr, np.inf), thi]
            if self.root_rot == "full":
                lo[3:6], hi[3:6] = -np.pi, np.pi
        else:
            x0, lo, hi = qpos[tq].copy(), tlo, thi
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        x_prev = x0.copy()
        wr = np.r_[np.full(nr, self.w_reg_root), np.full(len(tq), self.w_reg)]
        if first:
            wr[:nr] = 0.0  # 첫 프레임은 루트 초기값에 묶지 않는다

        def apply(x):
            if free:
                qpos[:3] = x[:3]
                if self.root_rot == "full":
                    qpos[3:7] = (R_ref * R.from_rotvec(x[3:6])).as_quat(scalar_first=True)
                else:
                    qpos[3:7] = R.from_euler('z', x[3]).as_quat(scalar_first=True)
            qpos[tq] = x[nr:]
            self._kin(qpos)

        aw = np.repeat(self.anchor_w, 3)
        x_null = np.r_[x0[:nr], self._init_qpos[tq]]
        wn = np.r_[np.zeros(nr), np.full(len(tq), self.w_null)]

        def fun(x):
            apply(x)
            res = [self._point(b, l) - targets[k] for k, (b, l, _) in enumerate(self.anchors)]
            return np.r_[aw * np.concatenate(res), wr * (x - x_prev), wn * (x - x_null)]

        def jac(x):
            apply(x)
            J = np.zeros((3 * len(self.anchors) + 2 * len(x), len(x)))
            for k, (b, l, _) in enumerate(self.anchors):
                p = self._point(b, l)
                self._jacp[:] = 0
                mj.mj_jac(m, self.data, self._jacp, self._jacr, p, b)
                rows = slice(3 * k, 3 * k + 3)
                if free:
                    J[rows, :3] = self._jacp[:, 0:3]
                    Jw = self._jacp[:, 3:6]  # body-frame 각속도에 대한 열
                    if self.root_rot == "full":
                        J[rows, 3:6] = Jw @ _right_jacobian(x[3:6])
                    else:
                        Rb = self.data.xmat[self.root_bid].reshape(3, 3)
                        J[rows, 3] = Jw @ (Rb.T @ EZ)
                J[rows, nr:] = self._jacp[:, td]
            J[:3 * len(self.anchors)] *= aw[:, None]
            na = 3 * len(self.anchors)
            J[na:na + len(x), :] = np.diag(wr)
            J[na + len(x):, :] = np.diag(wn)
            return J

        r = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method='trf',
                          max_nfev=self.max_nfev1 * (3 if first else 1),
                          ftol=1e-8, xtol=1e-8, gtol=1e-8)
        apply(r.x)
        self.last_step1 = r

    # ---------------- Step 2: 말단 체인 ----------------

    def _step2(self, qpos, hd, j, first):
        ch, mp = self.chains[j], self.tgt_maps[j]
        keys, qa, dof = ch['keys'], ch['qpa'], ch['dof']
        scale = mp['scale'] if mp['scale'] is not None else self.s_global
        src = np.array([self._src(hd, ip) for ip in mp['interp']])
        self._kin(qpos)
        start = self._point(keys[0][0], keys[0][1])
        targets = start + scale * (src - src[0])
        tk = list(range(1, len(keys)))
        bends = list(range(1, len(keys) - 1)) if self.bend != "none" else []
        src_b = {}
        for k in bends:
            v1, v2 = src[k - 1] - src[k], src[k + 1] - src[k]
            c = np.cross(v1, v2)
            if self.bend == "vector":
                src_b[k] = c / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            else:
                n = np.linalg.norm(c)
                src_b[k] = c / n if n > 1e-9 else None
        bends = [k for k in bends if src_b[k] is not None]

        x0 = np.clip(qpos[qa], ch['lo'] + 1e-6, ch['hi'] - 1e-6)
        x_prev = x0.copy()
        x_null = self._init_qpos[qa]
        w_reg = 0.0 if first else self.w_reg
        ed = mp.get('end_dir') if self.w_end > 0 else None
        if ed is not None:
            ed_bid, ed_v, (ba, bb) = ed
            ds = np.asarray(hd[bb][0]) - np.asarray(hd[ba][0])
            ds = ds / (np.linalg.norm(ds) + 1e-9)

        def apply(x):
            qpos[qa] = x
            self._kin(qpos)

        def bend_vec(k, P):
            v1, v2 = P[k - 1] - P[k], P[k + 1] - P[k]
            c = np.cross(v1, v2)
            if self.bend == "vector":
                return c / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), v1, v2, c
            n = np.linalg.norm(c)
            return (c / n if n > 1e-9 else np.zeros(3)), v1, v2, c

        def fun(x):
            apply(x)
            P = np.array([self._point(b, l) for b, l, _ in keys])
            res = [P[k] - targets[k] for k in tk]
            for k in bends:
                b = bend_vec(k, P)[0]
                res.append(self.w_bend * (b - src_b[k] if self.bend == "vector"
                                          else np.cross(b, src_b[k])))
            if ed is not None:
                res.append(self.w_end * (self.data.xmat[ed_bid].reshape(3, 3) @ ed_v - ds))
            res.append(w_reg * (x - x_prev))
            res.append(self.w_null * (x - x_null))
            return np.concatenate(res)

        def jac(x):
            apply(x)
            P = np.array([self._point(b, l) for b, l, _ in keys])
            Jp = [self._point_jac(b, l, dof).copy() for b, l, _ in keys]
            rows = [Jp[k] for k in tk]
            for k in bends:
                b, v1, v2, c = bend_vec(k, P)
                dv1, dv2 = Jp[k - 1] - Jp[k], Jp[k + 1] - Jp[k]
                dc = np.cross(dv1.T, v2).T + np.cross(v1, dv2.T).T  # (3, n)
                if self.bend == "vector":
                    a1, a2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    da = (v1 @ dv1) / (a1 + 1e-12) * a2 + a1 * (v2 @ dv2) / (a2 + 1e-12)
                    db = dc / (a1 * a2 + 1e-9) - np.outer(c, da) / (a1 * a2 + 1e-9) ** 2
                    rows.append(self.w_bend * db)
                else:
                    n = np.linalg.norm(c)
                    if n < 1e-9:
                        rows.append(np.zeros((3, len(x))))
                        continue
                    dn = (np.eye(3) - np.outer(b, b)) @ dc / n
                    rows.append(self.w_bend * (-_skew(src_b[k]) @ dn))  # d(b×s) = -[s]× db
            if ed is not None:
                rv = self.data.xmat[ed_bid].reshape(3, 3) @ ed_v
                self._jacr[:] = 0
                mj.mj_jac(self.model, self.data, self._jacp, self._jacr,
                          self.data.xpos[ed_bid], ed_bid)
                rows.append(self.w_end * (-_skew(rv) @ self._jacr[:, dof]))  # d(Rv) = ω × Rv
            rows.append(w_reg * np.eye(len(x)))
            rows.append(self.w_null * np.eye(len(x)))
            return np.vstack(rows)

        r = least_squares(fun, x0, jac=jac, bounds=(ch['lo'], ch['hi']), method='trf',
                          max_nfev=self.max_nfev2 * (3 if first else 1),
                          ftol=1e-8, xtol=1e-8, gtol=1e-8)
        apply(r.x)

    # ==================================================================
    # 평가용
    # ==================================================================

    def own_reference_points(self, hd):
        """자기 기준 오차용 (robot world point, reference world point, chain idx) 리스트.
        reference는 전역 상사변환 S(p_src)."""
        out, Tf = [], self._transform(hd)
        for j, mp in self.tgt_maps.items():
            if self.chains[j]['internal']:
                continue
            for (b, l, _), ip in zip(self.chains[j]['keys'], mp['interp']):
                out.append((self._point(b, l), Tf(self._src(hd, ip)), j))
        return out

    def pelvis_points(self, hd):
        """루트 상대 오차의 기준점: 양 다리 시작점 중점(로봇/소스)."""
        if len(self.tgt_legs) < 2:
            return self.data.xpos[self.root_bid].copy(), self._transform(hd)(np.asarray(hd[self.src_root][0]))
        rp = np.mean([self._point(*self.chains[j]['keys'][0][:2]) for j in self.tgt_legs], axis=0)
        Tf = self._transform(hd)
        sp = np.mean([Tf(self._src(hd, self.tgt_maps[j]['interp'][0])) for j in self.tgt_legs], axis=0)
        return rp, sp
