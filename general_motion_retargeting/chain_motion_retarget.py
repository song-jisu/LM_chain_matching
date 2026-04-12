"""
Chain Motion Retargeting (v4 - clean rewrite)
===============================================
GMR config 없이 동작 가능. chain 자동 추출 + 자동 좌표 변환 + mink IK.

핵심 원리:
1. Source/Target에서 serial chain 자동 추출
2. Rest pose에서 per-link direction rotation 계산 (좌표 변환)
3. 매 프레임: source link direction → direction rotation → robot 좌표계 타겟
4. mink IK solver로 joint angles 계산
"""

import mujoco as mj
import mink
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print


class ChainMotionRetargeting:

    def __init__(self, src_human, tgt_robot, actual_human_height=None,
                 verbose=True, **kwargs):

        # ── Robot model ──
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self.data = mj.MjData(self.model)

        self.has_freejoint = (self.model.njnt > 0
                              and self.model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE)

        self.body_name2id = {}
        self.body_id2name = {}
        for i in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.body_name2id[name] = i
            self.body_id2name[i] = name

        # ── Chain 추출 ──
        self.chains = self._extract_chains()

        # ── Body 매핑 ──
        self._load_mapping(src_human, tgt_robot, actual_human_height)

        # ── mink ──
        self._mink_config = mink.Configuration(self.model)
        self._mink_tasks = {}
        self._mink_limits = [mink.ConfigurationLimit(self.model)]

        # ── State ──
        self.prev_qpos = None
        self.scaled_human_data = None
        self._rest_computed = False
        self._link_dir_rots = {}
        self._robot_link_lens = {}
        self._tgt_rest_anchors = {}

        if verbose:
            print(f"[Chain v4] Robot: {self.xml_file}")
            print(f"[Chain v4] Chains: {len(self.chains)}")
            for ch in self.chains:
                mapped = sum(1 for hb in ch['human_bodies'] if hb)
                print(f"  {ch['name']}: {len(ch['body_ids'])} bodies, {mapped} mapped")

    # ==================================================================
    # Chain 추출
    # ==================================================================

    def _extract_chains(self):
        children = {i: [] for i in range(self.model.nbody)}
        for i in range(1, self.model.nbody):
            children[int(self.model.body_parentid[i])].append(i)

        body_joints = {}
        start = 1 if self.has_freejoint else 0
        for i in range(start, self.model.njnt):
            if self.model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE:
                continue
            bid = int(self.model.jnt_bodyid[i])
            body_joints.setdefault(bid, []).append(i)

        def has_jd(bid):
            if bid in body_joints:
                return True
            return any(has_jd(c) for c in children[bid])

        chains = []

        def trace(start_bid):
            body_ids = [start_bid]
            bid = start_bid
            while True:
                kids = children[bid]
                jkids = [c for c in kids if has_jd(c)]
                if len(jkids) == 0:
                    break
                elif len(jkids) == 1:
                    bid = jkids[0]
                    body_ids.append(bid)
                else:
                    break
            if len(body_ids) >= 2:
                chains.append({
                    'name': f"{self.body_id2name[body_ids[0]]}->{self.body_id2name[body_ids[-1]]}",
                    'body_ids': body_ids,
                    'human_bodies': [None] * len(body_ids),
                })
            jkids = [c for c in children[bid] if has_jd(c)]
            if len(jkids) >= 2:
                for kid in jkids:
                    trace(kid)

        root_bid = int(self.model.jnt_bodyid[0]) if self.has_freejoint else 1
        jkids = [c for c in children[root_bid] if has_jd(c)]
        for kid in (jkids if len(jkids) >= 2 else jkids[:1]):
            trace(kid)
        return chains

    # ==================================================================
    # Body 매핑
    # ==================================================================

    def _load_mapping(self, src_human, tgt_robot, actual_human_height):
        """IK config가 있으면 자동 로드, 없으면 나중에 CLI matching."""
        self.human_root_name = None
        self.human_scale_table = {}
        self.root_chain_indices = []  # root yaw 결정에 사용할 chain 인덱스
        self._mapping_ready = False

        try:
            with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
                config = json.load(f)

            self.human_root_name = config.get("human_root_name", "Hips")
            self.human_scale_table = config.get("human_scale_table", {})

            if actual_human_height is not None:
                ratio = actual_human_height / config.get("human_height_assumption", 1.8)
                for k in self.human_scale_table:
                    self.human_scale_table[k] *= ratio

            robot_to_human = {}
            for table_key in ["ik_match_table1", "ik_match_table2"]:
                for rname, entry in config.get(table_key, {}).items():
                    if rname not in robot_to_human:
                        robot_to_human[rname] = entry[0]

            for chain in self.chains:
                for i, bid in enumerate(chain['body_ids']):
                    bname = self.body_id2name[bid]
                    chain['human_bodies'][i] = robot_to_human.get(bname)

            # root chains: 다리 chain 자동 감지 (hip/leg/thigh 포함)
            for ci, ch in enumerate(self.chains):
                if any(kw in ch['name'].lower() for kw in ['hip', 'leg', 'thigh']):
                    self.root_chain_indices.append(ci)
            if not self.root_chain_indices:
                self.root_chain_indices = [0, 1] if len(self.chains) >= 2 else [0]

            self._mapping_ready = True

        except (KeyError, FileNotFoundError):
            pass  # CLI matching 필요

    def _extract_source_chains(self, source_bones, human_data):
        """source BVH의 bone hierarchy에서 serial chain 추출.
        bone_hierarchy가 있으면 실제 parent 배열 사용."""
        bones = source_bones

        # bone_hierarchy에서 실제 parent-child 관계 사용
        if hasattr(self, '_bone_hierarchy') and self._bone_hierarchy:
            bh = self._bone_hierarchy
            bones = bh['bones']
            parent_indices = bh['parents']

            children = {b: [] for b in bones}
            for i in range(len(bones)):
                pi = parent_indices[i]
                if pi >= 0:
                    children[bones[pi]].append(bones[i])
        else:
            # fallback: 거리 기반 추정
            children = {b: [] for b in bones}
            for i in range(1, len(bones)):
                bone = bones[i]
                pos = np.asarray(human_data[bone][0])
                best_parent = bones[0]
                best_dist = np.inf
                for j in range(i):
                    p = bones[j]
                    d = np.linalg.norm(pos - np.asarray(human_data[p][0]))
                    if d < best_dist:
                        best_dist = d
                        best_parent = p
                children[best_parent].append(bone)

        # serial chain 추출
        chains = []

        def trace(start_bone):
            chain_bones = [start_bone]
            bone = start_bone
            while True:
                kids = children.get(bone, [])
                if len(kids) == 0:
                    break
                elif len(kids) == 1:
                    bone = kids[0]
                    chain_bones.append(bone)
                else:
                    break

            if len(chain_bones) >= 2:
                chains.append(chain_bones)

            if len(children.get(bone, [])) >= 2:
                for kid in children[bone]:
                    trace(kid)

        root = bones[0]
        kids = children.get(root, [])
        if len(kids) >= 2:
            for kid in kids:
                trace(kid)
        elif len(kids) == 1:
            trace(kids[0])

        return chains

    def setup_cli_matching(self, source_bones, human_data=None):
        """CLI로 source chain ↔ target chain 매칭 + root chain 선택."""
        print(f"\n{'='*60}")
        print(f"  Chain Matching Setup")
        print(f"{'='*60}")

        # Source chain 추출
        src_chains = self._extract_source_chains(source_bones, human_data) if human_data else []

        print(f"\n  Source chains:")
        if src_chains:
            for si, sc in enumerate(src_chains):
                print(f"    [S{si}] {sc[0]} -> {sc[-1]} ({len(sc)} bodies)")
                print(f"         {sc}")
        else:
            print(f"    (chain 추출 실패, bone 리스트 사용)")
            for i, b in enumerate(source_bones):
                print(f"    [{i}] {b}")

        print(f"\n  Target chains:")
        for ci, ch in enumerate(self.chains):
            bodies = [self.body_id2name[b] for b in ch['body_ids']]
            print(f"    [T{ci}] {ch['name']} ({len(bodies)} bodies)")
            print(f"         {bodies}")

        # Chain ↔ Chain 매칭
        print(f"\n  Match source chain → target chain.")
        print(f"  Format: S0->T0, S1->T1, ...")
        print(f"  Body mapping is automatic (by position order).\n")

        match_input = input(f"  Chain matching (e.g. 0->0,1->1,2->2): ").strip()

        chain_pairs = []  # (src_chain_idx, tgt_chain_idx)
        for pair in match_input.split(','):
            pair = pair.strip().replace('S', '').replace('T', '')
            if '->' not in pair:
                continue
            parts = pair.split('->')
            try:
                si = int(parts[0].strip())
                ti = int(parts[1].strip())
                chain_pairs.append((si, ti))
            except (ValueError, IndexError):
                print(f"    Invalid: {pair}")

        # 매칭된 chain 쌍의 body를 순서대로 대응
        for si, ti in chain_pairs:
            if not src_chains or si >= len(src_chains) or ti >= len(self.chains):
                continue
            src_chain_bones = src_chains[si]
            tgt_chain = self.chains[ti]
            n_src = len(src_chain_bones)
            n_tgt = len(tgt_chain['body_ids'])

            # 균등 분배: source body를 target body에 매핑
            for si_body in range(n_src):
                tgt_idx = si_body * n_tgt // n_src if n_src > 1 else 0
                tgt_idx = min(tgt_idx, n_tgt - 1)
                if src_chain_bones[si_body] in source_bones:
                    tgt_chain['human_bodies'][tgt_idx] = src_chain_bones[si_body]

            mapped = [(bi, hb) for bi, hb in enumerate(tgt_chain['human_bodies']) if hb]
            print(f"    S{si} -> T{ti}: {len(mapped)} mapped")

        # Root chain 선택
        print(f"\n  Select root chains (target chain indices for pelvis yaw).")
        print(f"  Usually leg chains whose start positions define body orientation.")
        root_input = input(f"  Root target chain indices (e.g. 0,1): ").strip()
        try:
            self.root_chain_indices = [int(x.strip()) for x in root_input.split(',')]
        except ValueError:
            self.root_chain_indices = [0, 1] if len(self.chains) >= 2 else [0]

        # Root body 선택
        print(f"\n  Select root body name from source bones (pelvis/hips).")
        root_body = input(f"  Root body name: ").strip()
        if root_body and root_body in source_bones:
            self.human_root_name = root_body
        else:
            for ri in self.root_chain_indices:
                if ri < len(self.chains):
                    mapped = [hb for hb in self.chains[ri]['human_bodies'] if hb]
                    if mapped:
                        self.human_root_name = mapped[0]
                        break
            if not self.human_root_name:
                self.human_root_name = source_bones[0]

        # Scale table
        for ch in self.chains:
            for hb in ch['human_bodies']:
                if hb and hb not in self.human_scale_table:
                    self.human_scale_table[hb] = 1.0
        if self.human_root_name not in self.human_scale_table:
            self.human_scale_table[self.human_root_name] = 1.0

        self._mapping_ready = True

        # 결과 요약
        print(f"\n  {'='*40}")
        print(f"  Mapping summary:")
        print(f"  Root body: {self.human_root_name}")
        print(f"  Root chains: {self.root_chain_indices}")
        for ci, ch in enumerate(self.chains):
            mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies']) if hb]
            if mapped:
                print(f"  T[{ci}] {ch['name']}: {mapped}")
        print(f"  {'='*40}\n")

    # ==================================================================
    # Rest pose 계산
    # ==================================================================

    def _compute_rest(self, human_data):
        data_tmp = mj.MjData(self.model)
        data_tmp.qpos[:] = self.model.qpos0
        if self.has_freejoint:
            data_tmp.qpos[2] = 0.793
            data_tmp.qpos[3] = 1.0
        mj.mj_forward(self.model, data_tmp)

        tgt_root_bid = self.body_name2id.get('pelvis', 1)
        tgt_root = data_tmp.xpos[tgt_root_bid].copy()

        for chain in self.chains:
            mapped = [(bi, hb) for bi, hb in enumerate(chain['human_bodies'])
                      if hb and hb in human_data]
            if len(mapped) < 2:
                continue

            # rest anchors (root rotation용)
            bi0 = mapped[0][0]
            self._tgt_rest_anchors[chain['name']] = \
                data_tmp.xpos[chain['body_ids'][bi0]] - tgt_root

            # per-link direction rotation + robot link lengths
            rots = []
            lens = []
            for k in range(len(mapped) - 1):
                bi1, hb1 = mapped[k]
                bi2, hb2 = mapped[k + 1]

                sp1, sp2 = np.asarray(human_data[hb1][0]), np.asarray(human_data[hb2][0])
                sd = sp2 - sp1
                sl = np.linalg.norm(sd)
                sd = sd / sl if sl > 1e-6 else np.array([0, 0, -1])

                rp1 = data_tmp.xpos[chain['body_ids'][bi1]]
                rp2 = data_tmp.xpos[chain['body_ids'][bi2]]
                rd = rp2 - rp1
                rl = np.linalg.norm(rd)
                rd = rd / rl if rl > 1e-6 else np.array([0, 0, -1])

                cross = np.cross(sd, rd)
                dot = np.dot(sd, rd)
                cn = np.linalg.norm(cross)
                if cn > 1e-9:
                    rots.append(R.from_rotvec(cross / cn * np.arccos(np.clip(dot, -1, 1))))
                else:
                    rots.append(R.identity() if dot > 0 else R.from_euler('z', np.pi))
                lens.append(rl)

            self._link_dir_rots[chain['name']] = rots
            self._robot_link_lens[chain['name']] = lens

        self._rest_computed = True

    # ==================================================================
    # Retarget
    # ==================================================================

    def _scale(self, human_data):
        if self.human_root_name not in human_data:
            return human_data
        rp = human_data[self.human_root_name][0]
        rs = self.human_scale_table.get(self.human_root_name, 1.0)
        sr = rp * rs
        result = {self.human_root_name: [sr, human_data[self.human_root_name][1]]}
        for b in human_data:
            if b == self.human_root_name or b not in self.human_scale_table:
                continue
            s = self.human_scale_table[b]
            result[b] = [(human_data[b][0] - rp) * s + sr, human_data[b][1]]
        return result

    def _ground(self, data):
        # root chain의 끝 body 또는 Foot/foot 이름으로 바닥 감지
        foot_z = []
        for b in data:
            if "Foot" in b or "foot" in b or "ankle" in b or "toe" in b:
                foot_z.append(data[b][0][2])
        # root chain 끝 body도 확인
        for ri in self.root_chain_indices:
            if ri < len(self.chains):
                ch = self.chains[ri]
                mapped = [hb for hb in ch['human_bodies'] if hb and hb in data]
                if mapped:
                    foot_z.append(data[mapped[-1]][0][2])
        if not foot_z:
            return data
        lo = min(foot_z)
        # 이미 지면 가까이 있으면 (|lo| < 5cm) 보정하지 않음
        if abs(lo) < 0.05:
            return data
        z = -lo + 0.01
        return {k: [v[0] + np.array([0, 0, z]), v[1]] for k, v in data.items()}

    def _compute_forward_from_root_chains(self, human_data):
        """root chain들의 첫 mapped body 위치에서 forward 방향 계산."""
        root_pos = np.asarray(human_data[self.human_root_name][0])
        anchor_positions = []

        for ri in self.root_chain_indices:
            if ri >= len(self.chains):
                continue
            ch = self.chains[ri]
            mapped = [hb for hb in ch['human_bodies'] if hb and hb in human_data]
            if mapped:
                anchor_positions.append(np.asarray(human_data[mapped[0]][0]))

        if len(anchor_positions) >= 2:
            lr = anchor_positions[0] - anchor_positions[1]
            fwd = np.cross(lr, [0, 0, 1])
            fwd[2] = 0
            fn = np.linalg.norm(fwd)
            if fn > 1e-6:
                fwd = fwd / fn
                yaw = np.arctan2(fwd[1], fwd[0])
                self._fwd_rotation = R.from_euler("z", -yaw)
                return

        self._compute_forward_rotation(human_data)

    def _compute_forward_rotation(self, human_data):
        """BVH forward 방향을 자동 추정하고 +X로 맞추는 rotation 계산.
        좌우 어깨/다리의 cross product → forward."""
        # 좌우 body 후보
        lr_pairs = [
            ('LeftShoulder', 'RightShoulder'),
            ('LeftArm', 'RightArm'),
            ('LeftUpLeg', 'RightUpLeg'),
        ]
        root_pos = np.asarray(human_data[self.human_root_name][0])

        for left_name, right_name in lr_pairs:
            if left_name in human_data and right_name in human_data:
                left_pos = np.asarray(human_data[left_name][0])
                right_pos = np.asarray(human_data[right_name][0])
                lr = left_pos - right_pos
                up = np.array([0, 0, 1])
                fwd = np.cross(lr, up)  # BVH forward 방향
                fwd[2] = 0
                fn = np.linalg.norm(fwd)
                if fn > 1e-6:
                    fwd = fwd / fn
                    yaw = np.arctan2(fwd[1], fwd[0])
                    self._fwd_rotation = R.from_euler('z', -yaw)
                    return
        self._fwd_rotation = R.identity()

    def _align_human_data(self, human_data):
        """human_data의 모든 position을 root 기준으로 forward rotation 적용.
        rotation(quat)은 건드리지 않음 — position만 정렬."""
        root_pos = np.asarray(human_data[self.human_root_name][0])
        result = {}
        for bname in human_data:
            pos, quat = human_data[bname]
            pos_rel = pos - root_pos
            pos_rotated = self._fwd_rotation.apply(pos_rel)
            result[bname] = [pos_rotated + root_pos, quat]
        return result

    def retarget(self, human_data, offset_to_ground=False):
        human_data = {k: [np.asarray(v[0]), np.asarray(v[1])]
                      for k, v in human_data.items()}

        # 매핑이 안 되어 있으면 CLI matching 호출
        if not self._mapping_ready:
            source_bones = list(human_data.keys())
            self.setup_cli_matching(source_bones, human_data)

        # forward detection (첫 프레임) — root chain의 시작 body로 계산
        if not hasattr(self, '_fwd_rotation'):
            self._compute_forward_from_root_chains(human_data)

        if not self._rest_computed:
            aligned = self._align_human_data(human_data)
            self._compute_rest(aligned)

        # 시각화용: 원본 BVH (회전 안 함)
        scaled_viz = self._scale(human_data)
        scaled_viz = self._ground(scaled_viz)
        self.scaled_human_data = scaled_viz

        # LM용: forward 정렬된 BVH
        aligned = self._align_human_data(human_data)
        scaled = self._scale(aligned)
        scaled = self._ground(scaled)

        # mink init
        if self.prev_qpos is not None:
            self._mink_config.data.qpos[:] = self.prev_qpos
        else:
            self._mink_config.data.qpos[:] = self.model.qpos0
            if self.has_freejoint:
                self._mink_config.data.qpos[2] = 0.793
                self._mink_config.data.qpos[3] = 1.0

        # joint limit 클램핑 (mink crash 방지)
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                addr = int(self.model.jnt_qposadr[i])
                lo, hi = self.model.jnt_range[i]
                self._mink_config.data.qpos[addr] = np.clip(
                    self._mink_config.data.qpos[addr], lo + 1e-4, hi - 1e-4)

        mj.mj_forward(self._mink_config.model, self._mink_config.data)

        # ── Root: position + forward rotation (BVH를 안 돌리고 robot을 돌림) ──
        if self.has_freejoint and self.human_root_name in scaled:
            src_root = scaled[self.human_root_name][0]
            self._mink_config.data.qpos[0] = src_root[0]
            self._mink_config.data.qpos[1] = src_root[1]
            self._mink_config.data.qpos[2] = src_root[2]
            # robot을 BVH forward 방향으로 회전 (BVH 좌표는 건드리지 않음)
            self._mink_config.data.qpos[3:7] = self._fwd_rotation.as_quat(scalar_first=True)

            mj.mj_forward(self._mink_config.model, self._mink_config.data)

        # ── Chain별 LM 최적화 ──
        from scipy.optimize import least_squares

        # ── Tree 순서 LM: root yaw(다리시작점) → waist → legs → arms ──
        from scipy.optimize import least_squares

        qpos = self.data.qpos.copy()
        qpos[:] = self.prev_qpos if self.prev_qpos is not None else self.model.qpos0

        # root position
        if self.has_freejoint and self.human_root_name in scaled:
            qpos[0] = scaled[self.human_root_name][0][0]
            qpos[1] = scaled[self.human_root_name][0][1]
            qpos[2] = scaled[self.human_root_name][0][2]

        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

        # Step 1: root yaw + waist joints로 모든 chain 시작점 위치 맞추기
        # 각 chain의 첫 mapped body 위치를 타겟으로, root yaw + waist를 최적화
        anchor_targets = []
        for ch in self.chains:
            mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                      if hb and hb in scaled]
            if mapped:
                bi0, hb0 = mapped[0]
                anchor_targets.append((ch['body_ids'][bi0], np.asarray(scaled[hb0][0])))

        if anchor_targets:
            anchor_bids = [a[0] for a in anchor_targets]
            anchor_pos = np.array([a[1] for a in anchor_targets])

            # 최적화 변수: root yaw + waist chain joints
            step1_qpa = []
            step1_dof = []
            step1_lo = []
            step1_hi = []

            # waist chain joints 수집
            for ch in self.chains:
                if 'waist' in ch['name'] or 'torso' in ch['name']:
                    for bid in ch['body_ids']:
                        start_j = 1 if self.has_freejoint else 0
                        for i in range(start_j, self.model.njnt):
                            if self.model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE:
                                continue
                            if int(self.model.jnt_bodyid[i]) == bid:
                                qa = int(self.model.jnt_qposadr[i])
                                if qa not in step1_qpa:
                                    lo, hi = self.model.jnt_range[i] if self.model.jnt_limited[i] else (-np.pi, np.pi)
                                    step1_qpa.append(qa)
                                    step1_dof.append(int(self.model.jnt_dofadr[i]))
                                    step1_lo.append(float(lo))
                                    step1_hi.append(float(hi))

            # x = [yaw, waist_joints...]
            has_yaw = self.has_freejoint
            if has_yaw:
                cur_yaw = R.from_quat(qpos[3:7], scalar_first=True).as_euler('xyz')[2]
                if self.prev_qpos is None:
                    cur_yaw = 0.0
                x0 = np.concatenate([[cur_yaw], [qpos[a] for a in step1_qpa]])
                lo_b = np.concatenate([[-np.pi], step1_lo])
                hi_b = np.concatenate([[np.pi], step1_hi])
            else:
                x0 = np.array([qpos[a] for a in step1_qpa])
                lo_b = np.array(step1_lo)
                hi_b = np.array(step1_hi)

            x0 = np.clip(x0, lo_b + 1e-4, hi_b - 1e-4)
            jacp_buf_s1 = np.zeros((3, self.model.nv))

            def step1_residual(x):
                if has_yaw:
                    qpos[3:7] = R.from_euler('z', x[0]).as_quat(scalar_first=True)
                    for k, a in enumerate(step1_qpa):
                        qpos[a] = x[1 + k]
                else:
                    for k, a in enumerate(step1_qpa):
                        qpos[a] = x[k]
                self.data.qpos[:] = qpos
                mj.mj_forward(self.model, self.data)
                res = []
                for i, bid in enumerate(anchor_bids):
                    res.extend((self.data.xpos[bid] - anchor_pos[i]).tolist())
                return np.array(res)

            result = least_squares(
                fun=step1_residual, x0=x0,
                bounds=(lo_b, hi_b), method='trf', max_nfev=50,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )

            if has_yaw:
                qpos[3:7] = R.from_euler('z', result.x[0]).as_quat(scalar_first=True)
                for k, a in enumerate(step1_qpa):
                    qpos[a] = result.x[1 + k]
            else:
                for k, a in enumerate(step1_qpa):
                    qpos[a] = result.x[k]
            self.data.qpos[:] = qpos
            mj.mj_forward(self.model, self.data)

        # Step 2: 각 chain 내부 LM (시작점은 step1에서 맞춰짐)
        chain_order = list(self.chains)

        jacp_buf = np.zeros((3, self.model.nv))
        jacr_buf = np.zeros((3, self.model.nv))
        w_reg = 0.05

        for step, ch in enumerate(chain_order):
            mapped = [(bi, hb) for bi, hb in enumerate(ch['human_bodies'])
                      if hb and hb in scaled]
            if not mapped:
                continue

            target_bids = [ch['body_ids'][bi] for bi, _ in mapped]
            target_pos_list = [np.asarray(scaled[hb][0]) for _, hb in mapped]

            # chain 시작 body anchor 타겟 추가
            first_bid = ch['body_ids'][0]
            if first_bid not in target_bids:
                target_bids.insert(0, first_bid)
                target_pos_list.insert(0, target_pos_list[0].copy())

            # (자식 chain 시작점은 Step 1에서 이미 맞춰짐)

            target_pos = np.array(target_pos_list)
            n_targets = len(target_bids)

            # 이 chain의 joints
            body_set = set(ch['body_ids'])
            chain_qpa = []
            chain_dof = []
            chain_lo = []
            chain_hi = []
            start_j = 1 if self.has_freejoint else 0
            for i in range(start_j, self.model.njnt):
                if self.model.jnt_type[i] != mj.mjtJoint.mjJNT_HINGE:
                    continue
                if int(self.model.jnt_bodyid[i]) in body_set:
                    lo, hi = self.model.jnt_range[i] if self.model.jnt_limited[i] else (-np.pi, np.pi)
                    chain_qpa.append(int(self.model.jnt_qposadr[i]))
                    chain_dof.append(int(self.model.jnt_dofadr[i]))
                    chain_lo.append(float(lo))
                    chain_hi.append(float(hi))

            if not chain_qpa:
                continue

            # 첫 step(waist)에 root yaw도 포함
            include_yaw = False  # yaw는 이미 step0에서 풀었음
            yaw_dof_idx = 5  # freejoint rz

            if include_yaw:
                cur_yaw = R.from_quat(qpos[3:7], scalar_first=True).as_euler('xyz')[2]
                x0 = np.concatenate([[cur_yaw], [qpos[a] for a in chain_qpa]])
                lo_b = np.concatenate([[-np.pi], chain_lo])
                hi_b = np.concatenate([[np.pi], chain_hi])
            else:
                x0 = np.array([qpos[a] for a in chain_qpa])
                lo_b = np.array(chain_lo)
                hi_b = np.array(chain_hi)

            x0 = np.clip(x0, lo_b + 1e-4, hi_b - 1e-4)
            x_prev = x0.copy()

            # Bend direction constraint: source에서 각 link가 꺾이는 방향을 계산하고
            # target chain의 중간 body들이 같은 방향으로 꺾이도록 유도
            w_bend = 0.3
            bend_targets = []  # list of (body_id, bend_direction_unit_vec)
            all_chain_bids = ch['body_ids']

            # source에서 연속 3점의 bend direction 계산
            mapped_pos = target_pos  # mapped body positions (source)
            if len(mapped_pos) >= 3:
                for mi in range(1, len(mapped_pos) - 1):
                    p_prev = mapped_pos[mi - 1]
                    p_curr = mapped_pos[mi]
                    p_next = mapped_pos[mi + 1]
                    # 굽힘 벡터: curr에서 prev-next 직선까지의 수선
                    v1 = p_prev - p_curr
                    v2 = p_next - p_curr
                    # 굽힘 방향 = 두 link가 이루는 평면의 법선
                    bend_normal = np.cross(v1, v2)
                    bn = np.linalg.norm(bend_normal)
                    if bn > 1e-9:
                        bend_normal = bend_normal / bn
                        # 대응하는 robot body
                        bid = target_bids[mi]
                        bend_targets.append((bid, bend_normal))

            # unmapped 중간 body에도 보간된 bend direction 적용
            for bi in range(1, len(all_chain_bids) - 1):
                bid = all_chain_bids[bi]
                if bid in [b for b, _ in bend_targets]:
                    continue
                # 가장 가까운 bend target에서 방향 가져오기
                if bend_targets:
                    bend_targets.append((bid, bend_targets[-1][1]))

            def _apply(x, _iy=include_yaw, _qpa=chain_qpa):
                if _iy:
                    qpos[3:7] = R.from_euler('z', x[0]).as_quat(scalar_first=True)
                    for k, a in enumerate(_qpa):
                        qpos[a] = x[1 + k]
                else:
                    for k, a in enumerate(_qpa):
                        qpos[a] = x[k]
                self.data.qpos[:] = qpos
                mj.mj_forward(self.model, self.data)

            def _make_residual(_iy=include_yaw, _qpa=chain_qpa,
                               _tbids=target_bids, _tpos=target_pos,
                               _xprev=x_prev, _bends=bend_targets,
                               _abids=all_chain_bids):
                def residual(x):
                    _apply(x, _iy, _qpa)
                    res = []
                    # position matching
                    for i, bid in enumerate(_tbids):
                        diff = self.data.xpos[bid] - _tpos[i]
                        res.extend(diff.tolist())
                    # bend direction: robot의 각 중간 body의 굽힘 방향이 source와 일치
                    if _bends:
                        for bid, src_bend in _bends:
                            # robot 현재 굽힘 방향 계산
                            bi_in_chain = _abids.index(bid) if bid in _abids else -1
                            if bi_in_chain > 0 and bi_in_chain < len(_abids) - 1:
                                rp = self.data.xpos[_abids[bi_in_chain - 1]]
                                rc = self.data.xpos[bid]
                                rn = self.data.xpos[_abids[bi_in_chain + 1]]
                                rv1 = rp - rc
                                rv2 = rn - rc
                                robot_bend = np.cross(rv1, rv2)
                                rbn = np.linalg.norm(robot_bend)
                                if rbn > 1e-9:
                                    robot_bend = robot_bend / rbn
                                    # bend 방향 차이
                                    bend_err = np.cross(robot_bend, src_bend)
                                    res.extend((w_bend * bend_err).tolist())
                                else:
                                    res.extend([0.0, 0.0, 0.0])
                            else:
                                res.extend([0.0, 0.0, 0.0])
                    # regularization
                    res.extend((w_reg * (x - _xprev)).tolist())
                    return np.array(res)
                return residual

            def _make_jacobian(_iy=include_yaw, _qpa=chain_qpa,
                               _dof=chain_dof, _tbids=target_bids,
                               _nt=n_targets, _bends=bend_targets):
                n_bend = len(_bends) * 3 if _bends else 0
                def jacobian(x):
                    _apply(x, _iy, _qpa)
                    n_x = len(x)
                    n_res = _nt * 3 + n_bend + n_x
                    jac = np.zeros((n_res, n_x))
                    # position jacobian
                    for i, bid in enumerate(_tbids):
                        jacp_buf[:] = 0
                        mj.mj_jacBody(self.model, self.data, jacp_buf, jacr_buf, bid)
                        if _iy:
                            jac[i*3:(i+1)*3, 0] = jacp_buf[:, yaw_dof_idx]
                            for k, da in enumerate(_dof):
                                if 0 <= da < self.model.nv:
                                    jac[i*3:(i+1)*3, 1+k] = jacp_buf[:, da]
                        else:
                            for k, da in enumerate(_dof):
                                if 0 <= da < self.model.nv:
                                    jac[i*3:(i+1)*3, k] = jacp_buf[:, da]
                    # bend는 수치 미분 (해석적 계산이 복잡)
                    if n_bend > 0:
                        eps = 1e-6
                        r0 = _make_residual()
                        res0 = r0(x)
                        bend_start = _nt * 3
                        for j in range(n_x):
                            xp = x.copy()
                            xp[j] += eps
                            resp = r0(xp)
                            jac[bend_start:bend_start+n_bend, j] = \
                                (resp[bend_start:bend_start+n_bend] - res0[bend_start:bend_start+n_bend]) / eps
                    # regularization
                    reg_off = _nt * 3 + n_bend
                    for k in range(n_x):
                        jac[reg_off + k, k] = w_reg
                    return jac
                return jacobian

            result = least_squares(
                fun=_make_residual(), x0=x0, jac=_make_jacobian(),
                bounds=(lo_b, hi_b), method='trf', max_nfev=30,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )

            # 결과 반영
            if include_yaw:
                qpos[3:7] = R.from_euler('z', result.x[0]).as_quat(scalar_first=True)
                for k, a in enumerate(chain_qpa):
                    qpos[a] = result.x[1 + k]
            else:
                for k, a in enumerate(chain_qpa):
                    qpos[a] = result.x[k]

            self.data.qpos[:] = qpos
            mj.mj_forward(self.model, self.data)

        # LM 결과를 prev_qpos에 저장 (다음 프레임 warm start용, 정렬된 좌표계)
        self.prev_qpos = qpos.copy()

        # 출력용: root를 BVH 원본 좌표계로 되돌리기
        if self.has_freejoint and self.human_root_name in scaled_viz:
            # position: 원본 BVH root 위치
            qpos[0:3] = scaled_viz[self.human_root_name][0]
            # rotation: 정렬 역변환 (LM좌표 → BVH좌표)
            r_lm = R.from_quat(qpos[3:7], scalar_first=True)
            r_out = self._fwd_rotation.inv() * r_lm
            qpos[3:7] = r_out.as_quat(scalar_first=True)

        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        return qpos