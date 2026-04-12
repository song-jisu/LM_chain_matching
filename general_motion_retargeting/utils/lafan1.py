import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
            
        if format == "lafan1":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        elif format == "robot":
            pass  # robot BVH: bone name 그대로 사용, FootMod 없음
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    if format == "robot":
        # robot BVH: 실제 높이 = 최고점 Z - 최저점 Z (첫 프레임 기준)
        all_z = [result[b][0][2] for b in result]
        human_height = max(all_z) - min(all_z) + 0.05
    else:
        human_height = 1.75

    # bone hierarchy 정보 (chain 추출용)
    bone_hierarchy = {
        "bones": list(data.bones),
        "parents": data.parents.tolist(),
    }

    return frames, human_height, bone_hierarchy


