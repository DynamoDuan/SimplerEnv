"""
Panda Robot 抓取可乐罐的脚本（只需要抓起来）
"""
import os
# 设置 headless 渲染模式（必须在导入 sapien 之前）
if not os.environ.get("DISPLAY") or os.environ.get("DISPLAY") == "":
    import subprocess
    try:
        result = subprocess.run(['xdpyinfo', '-display', ':99'], 
                              capture_output=True, timeout=1)
        if result.returncode == 0:
            os.environ["DISPLAY"] = ":99"
    except:
        pass

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["EGL_PLATFORM"] = "device"

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sapien.core as sapien
import gymnasium as gym
import h5py

# 导入工具函数和运动规划器
import sys
sys.path.insert(0, '/data/peiqiduan/ManiSkill2_real2sim')
sys.path.insert(0, '/data/peiqiduan/SimplerEnv')

from simpler_env.utils.motionplanning import (
    PandaArmMotionPlanningSolver,
    compute_grasp_info_by_obb,
    get_actor_obb,
)
import simpler_env

# ========== 配置区域 ==========
OUTPUT_DIR = "./output"  # 输出目录
# ==============================

def solve(env, seed=None):
    """执行一次抓取任务（只抓取，不放置）"""
    np.random.seed(seed)
    
    # 随机初始化物体位置和旋转
    z_angle_deg = np.random.uniform(0, 360)
    quat = R.from_euler('z', z_angle_deg, degrees=True).as_quat(scalar_first=True)
    translation = np.array([np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03)]) + np.array([0.08, 0.05])
    
    options = {
        "obj_init_options": {
            "init_xy": translation,  # 2D array [x, y]
            "init_rot_quat": quat,  # 4D array [w, x, y, z]
        }
    }
    
    obs, reset_info = env.reset(seed=seed, options=options)

    # 创建 Panda 运动规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env_unwrapped = env.unwrapped

    # 获取目标物体（可乐罐）
    obj = env_unwrapped.obj
    obb = get_actor_obb(obj)

    # 记录初始姿态
    initial_pose = env_unwrapped.agent.tcp.pose
    
    # 计算抓取姿态
    approaching = np.array([0, 0, -1])  # 从上方接近
    target_closing = env_unwrapped.agent.tcp.pose.to_transformation_matrix()[:3, 1]
    
    # 基于 OBB 计算抓取信息
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env_unwrapped.agent.build_grasp_pose(approaching, closing, obj.pose.p)
    
    # 执行抓取序列（只抓取，不放置）
    # 1. 移动到抓取位置上方
    goal_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose(goal_pose)
    
    # 2. 下降到抓取位置
    planner.move_to_pose(grasp_pose)
    
    # 3. 闭合夹爪抓取
    planner.close_gripper()
    
    # 4. 抬起物体
    planner.move_to_pose(goal_pose)
    
    # 5. 返回初始位置
    planner.move_to_pose(initial_pose)
    planner.close()

    # 检查是否成功（检查是否抓取成功）
    if planner.failure:
        return None, None

    return [obs] + planner.obs_list, planner.action_list

if __name__ == "__main__":
    # 注册环境（通过导入 simpler_env 和 mani_skill2_real2sim.envs 自动注册）
    import mani_skill2_real2sim.envs.custom_scenes.grasp_single_in_scene
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建环境 - Panda Robot 抓取可乐罐
    # 使用 GraspSingleOpenedCokeCanInScene-v0 并指定 robot="panda"
    print("创建 Panda Robot 抓取可乐罐环境...")
    env = gym.make(
        "GraspSingleOpenedCokeCanInScene-v0",
        robot="panda",
        scene_name="dummy_tabletop",  # 使用简单的桌面场景
        scene_offset=np.array([0, -0.21, 0]),  # dummy_tabletop 的默认偏移
        obs_mode="rgbd",
        prepackaged_config=False,  # 不使用预打包配置，使用自定义配置
        # Headless 渲染配置
        renderer_kwargs={
            "offscreen_only": True,
            "device": "cuda:0",
        },
    )
    
    print("开始执行抓取任务...")
    obs_list, action_list = solve(env, seed=42)
    
    if obs_list is None:
        print("❌ 任务失败")
        exit(1)
    
    print(f"✅ 任务成功！生成了 {len(obs_list)} 个观测和 {len(action_list)} 个动作")
    
    # 保存轨迹数据
    output_file = output_dir / "panda_pick_coke_can.h5"
    with h5py.File(str(output_file), "w") as f:
        our_images = np.stack([obs['image']['base_camera']['our_rgb'] for obs in obs_list])
        our_depth = np.stack([obs['image']['base_camera']['new_depth'] for obs in obs_list])
        raw_images = np.stack([obs['image']['base_camera']['raw_rgb'] for obs in obs_list])
        raw_depth = np.stack([obs['image']['base_camera']['depth'] for obs in obs_list])
        
        f.create_dataset("our_image", data=our_images, compression="gzip")
        f.create_dataset("our_depth", data=our_depth, compression="gzip")
        f.create_dataset("raw_image", data=raw_images, compression="gzip")
        f.create_dataset("raw_depth", data=raw_depth, compression="gzip")
    
        tcp_poses = np.stack([obs['extra']['tcp_pose'] for obs in obs_list])
        f.create_dataset("state", data=tcp_poses, compression="gzip")
    
        actions = np.stack(action_list)
        f.create_dataset("action", data=actions, compression="gzip")
    
    print(f"轨迹已保存到: {output_file}")
    
    # 保存视频
    video_filename = output_dir / "panda_pick_coke_can.mp4"
    try:
        first_obs = obs_list[0]
        img_height = first_obs['image']['base_camera']['our_rgb'].shape[0]
        img_width = first_obs['image']['base_camera']['our_rgb'].shape[1]
        
        frame_size = (img_width * 2, img_height)
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_filename), fourcc, fps, frame_size)
        
        for obs in obs_list:
            our_frame = cv2.cvtColor(obs['image']['base_camera']['our_rgb'], cv2.COLOR_RGB2BGR)
            raw_frame = cv2.cvtColor(obs['image']['base_camera']['raw_rgb'], cv2.COLOR_RGB2BGR)
            concat_frame = np.hstack((our_frame, raw_frame))
            video_writer.write(concat_frame)
        
        video_writer.release()
        print(f"✅ 视频已保存到: {video_filename}")
    except Exception as e:
        print(f"❌ 保存视频失败: {e}")

