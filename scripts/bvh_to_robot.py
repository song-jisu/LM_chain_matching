import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ChainMotionRetargeting
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "robot"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--save_bvh",
        default=None,
        help="Path to save robot motion as BVH file.",
    )

    parser.add_argument(
        "--method",
        choices=["ik", "chain"],
        default="ik",
        help="Retargeting method: 'ik' (GMR default) or 'chain' (chain-based)",
    )

    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    if args.save_bvh is not None:
        bvh_frames = []  # list of dicts {body_name: (pos, quat)}

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    if args.method == "chain":
        retargeter = ChainMotionRetargeting(
            src_human=f"bvh_{args.format}",
            tgt_robot=args.robot,
            actual_human_height=actual_human_height,
        )
    else:
        retargeter = GMR(
            src_human=f"bvh_{args.format}",
            tgt_robot=args.robot,
            actual_human_height=actual_human_height,
        )

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
        
        if args.save_path is not None:
            qpos_list.append(qpos)

        if args.save_bvh is not None:
            # robot body positions + orientations 수집
            import mujoco
            mujoco.mj_forward(retargeter.model, retargeter.data)
            frame = {}
            for bid in range(retargeter.model.nbody):
                name = mujoco.mj_id2name(retargeter.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if name and name != 'world':
                    pos = retargeter.data.xpos[bid].copy()
                    xmat = retargeter.data.xmat[bid].reshape(3, 3)
                    from scipy.spatial.transform import Rotation as Rot
                    quat = Rot.from_matrix(xmat).as_quat(scalar_first=True)
                    frame[name] = (pos.tolist(), quat.tolist())
            bvh_frames.append(frame)
    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Save BVH
    if args.save_bvh is not None and bvh_frames:
        import mujoco
        from scipy.spatial.transform import Rotation as Rot

        model = retargeter.model
        data = retargeter.data

        # Robot body tree → BVH hierarchy
        # BVH에 포함할 body: mapped body만 (chain의 human_bodies가 있는 body)
        # 간단히: 모든 non-world body를 포함
        children = {}
        for bid in range(model.nbody):
            children[bid] = []
        for bid in range(1, model.nbody):
            pid = int(model.body_parentid[bid])
            children[pid].append(bid)

        def body_name(bid):
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

        # BVH hierarchy 문자열 생성
        root_bid = 1  # first non-world body
        frame_time = 1.0 / motion_fps

        def write_hierarchy(bid, depth=0):
            name = body_name(bid)
            offset = model.body_pos[bid] * 100  # m → cm
            indent = "\t" * depth
            lines = []
            if depth == 0:
                lines.append(f"ROOT {name}")
            else:
                lines.append(f"{indent}JOINT {name}")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            if depth == 0:
                lines.append(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
            else:
                lines.append(f"{indent}\tCHANNELS 3 Zrotation Yrotation Xrotation")

            kids = children[bid]
            if not kids:
                lines.append(f"{indent}\tEnd Site")
                lines.append(f"{indent}\t{{")
                lines.append(f"{indent}\t\tOFFSET 0.000000 0.000000 0.000000")
                lines.append(f"{indent}\t}}")
            else:
                for kid in kids:
                    lines.extend(write_hierarchy(kid, depth + 1))

            lines.append(f"{indent}}}")
            return lines

        hierarchy = ["HIERARCHY"]
        hierarchy.extend(write_hierarchy(root_bid))

        # Motion data: 각 프레임의 body global pos/rot → local euler
        motion_lines = ["MOTION"]
        motion_lines.append(f"Frames: {len(bvh_frames)}")
        motion_lines.append(f"Frame Time: {frame_time:.6f}")

        # body 순서 (DFS)
        body_order = []
        def collect_order(bid):
            body_order.append(bid)
            for kid in children[bid]:
                collect_order(kid)
        collect_order(root_bid)

        for frame in bvh_frames:
            vals = []
            for bid in body_order:
                name = body_name(bid)
                if name not in frame:
                    if bid == root_bid:
                        vals.extend([0, 0, 0, 0, 0, 0])
                    else:
                        vals.extend([0, 0, 0])
                    continue

                pos, quat = frame[name]
                pos = np.array(pos)
                quat = np.array(quat)

                # global rotation → local rotation (parent 기준)
                pid = int(model.body_parentid[bid])
                pname = body_name(pid)
                if pid > 0 and pname in frame:
                    p_quat = np.array(frame[pname][1])
                    r_parent = Rot.from_quat(p_quat, scalar_first=True)
                    r_global = Rot.from_quat(quat, scalar_first=True)
                    r_local = r_parent.inv() * r_global
                else:
                    r_local = Rot.from_quat(quat, scalar_first=True)

                euler = r_local.as_euler('ZYX', degrees=True)  # BVH: ZYX order

                if bid == root_bid:
                    vals.extend([pos[0]*100, pos[1]*100, pos[2]*100])  # cm
                    vals.extend(euler.tolist())
                else:
                    vals.extend(euler.tolist())

            motion_lines.append(" ".join(f"{v:.6f}" for v in vals))

        # Write BVH file
        with open(args.save_bvh, "w") as f:
            f.write("\n".join(hierarchy) + "\n")
            f.write("\n".join(motion_lines) + "\n")

        print(f"Saved robot BVH to {args.save_bvh} ({len(bvh_frames)} frames, {len(body_order)} bodies)")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
