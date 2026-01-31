"""
Google Robot 抓取可乐罐 - 简单位置反馈控制版本（已修复）

基于 grasp_coke_simple_working.py 的成功经验，
使用简单的位置反馈控制，不使用IK，
但使用正确的 grasp_pose 计算（approaching方向正确）。
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

import cv2
import numpy as np
from pathlib import Path
import sapien.core as sapien
import sys
sys.path.insert(0, '/data/peiqiduan/SimplerEnv')

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.motionplanning.utils import get_actor_obb, compute_grasp_info_by_obb

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

OUTPUT_DIR = "./output"


def move_to_position(env, target_pos, gripper_state, steps=80, tolerance=0.01, step_size=0.01):
    """使用简单的位置反馈移动到目标位置

    Args:
        target_pos: 目标位置（世界坐标系）
        gripper_state: 夹爪状态 (1.0=开, -1.0=闭)
        steps: 最大步数
        tolerance: 位置容差（米）
        step_size: 每步最大移动距离（米）
    """
    tcp_link_name = env.unwrapped.agent.config.ee_link_name
    tcp_link = [link for link in env.unwrapped.agent.robot.get_links()
                if link.get_name() == tcp_link_name][0]

    obs_list = []

    for i in range(steps):
        current_pos = tcp_link.get_pose().p
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)

        if distance < tolerance:
            print(f"       ✓ 到达目标 (步骤 {i+1}/{steps}, 误差 {distance:.4f}m)")
            break

        # 限制移动速度
        if distance > step_size:
            delta = delta / distance * step_size

        # action: [delta_xyz, delta_rotation, gripper]
        action = np.concatenate([delta, np.zeros(3), [gripper_state]])
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs)

        if (i + 1) % 20 == 0:
            print(f"       步骤 {i+1}/{steps}, 距离目标: {distance:.4f}m")

    # 最后检查
    final_pos = tcp_link.get_pose().p
    final_error = np.linalg.norm(final_pos - target_pos)
    if final_error >= tolerance:
        print(f"       ⚠️  未完全到达目标 (最终误差: {final_error:.4f}m)")

    return obs_list


def solve(env, seed=None):
    """执行一次抓取任务"""
    np.random.seed(seed)
    obs, reset_info = env.reset(seed=seed)

    env_unwrapped = env.unwrapped
    obj = env_unwrapped.obj

    # 获取 TCP link
    tcp_link_name = env_unwrapped.agent.config.ee_link_name
    tcp_link = [link for link in env_unwrapped.agent.robot.get_links()
                if link.get_name() == tcp_link_name][0]

    initial_tcp_pose = tcp_link.get_pose()

    print(f"初始 TCP 位置（世界）: {initial_tcp_pose.p}")
    print(f"物体位置（世界）: {obj.pose.p}")

    # 计算抓取姿态
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(obj)

    # 关键修复：使用正确的approaching和closing方向
    approaching = np.array([0, 0, -1])  # 从上往下抓取
    target_closing = np.array([0, 1, 0])  # 水平方向作为closing

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing, center = grasp_info["closing"], grasp_info["center"]

    # 使用 obj.pose.p 作为抓取位置
    grasp_pose_world = env_unwrapped.agent.build_grasp_pose(approaching, closing, obj.pose.p)

    # 验证 grasp_pose
    grasp_mat = grasp_pose_world.to_transformation_matrix()
    print(f"\n抓取姿态验证:")
    print(f"  位置: {grasp_pose_world.p}")
    print(f"  Z轴 (approaching): {grasp_mat[:3, 2]} (期望: [0, 0, -1])")
    print(f"  Z轴误差: {np.linalg.norm(grasp_mat[:3, 2] - approaching):.6f}")

    obs_list = [obs]

    # 定义目标位置
    grasp_pos = grasp_pose_world.p
    above_pos = grasp_pos + np.array([0, 0, 0.1])  # 抓取位置上方10cm
    lift_pos = grasp_pos + np.array([0, 0, 0.08])  # 抬起8cm

    print(f"\n目标位置:")
    print(f"  1. 上方: {above_pos}")
    print(f"  2. 抓取: {grasp_pos}")
    print(f"  3. 抬起: {lift_pos}")

    # 执行抓取序列
    print("\n执行抓取动作...")

    # 1. 移动到抓取位置上方
    print("  1. 移动到抓取位置上方...")
    obs_list.extend(move_to_position(env, above_pos, gripper_state=1.0, steps=80, step_size=0.01))

    # 2. 下降到抓取位置
    print("  2. 下降到抓取位置...")
    obs_list.extend(move_to_position(env, grasp_pos, gripper_state=1.0, steps=40, step_size=0.005))

    # 检查到达状态
    tcp_pos = tcp_link.get_pose().p
    obj_pos = obj.pose.p
    print(f"     TCP位置: {tcp_pos}, 物体位置: {obj_pos}")
    print(f"     距离: {np.linalg.norm(tcp_pos - obj_pos):.4f}m")

    # 3. 闭合夹爪
    print("  3. 闭合夹爪...")
    for i in range(20):
        action = np.array([0, 0, 0, 0, 0, 0, -1.0])  # 保持位置，闭合夹爪
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs)

        if i % 5 == 4:
            gripper_qpos = env.unwrapped.agent.robot.get_qpos()[-2:]
            print(f"     步骤 {i+1}/20 - 夹爪: {gripper_qpos}, 物体高度: {obj.pose.p[2]:.4f}")

    # 检查夹爪状态
    gripper_final = env.unwrapped.agent.robot.get_qpos()[-2:]
    print(f"     夹爪闭合后: {gripper_final}")

    # 4. 抬起物体
    print("  4. 抬起物体...")
    obj_height_before = obj.pose.p[2]
    obs_list.extend(move_to_position(env, lift_pos, gripper_state=-1.0, steps=60, step_size=0.005))
    obj_height_after = obj.pose.p[2]

    height_change = obj_height_after - obj_height_before
    print(f"     物体高度变化: {obj_height_before:.4f} -> {obj_height_after:.4f} (Δ={height_change:.4f}m)")

    # 判断成功
    if height_change > 0.03:
        print("     ✅ 物体被成功抬起！")
        success = True
    else:
        print("     ❌ 物体没有被抓起")
        success = False

    print(f"\n✅ 完成！生成了 {len(obs_list)} 帧")

    return obs_list, success


if __name__ == "__main__":
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建环境
    print("创建 Google Robot 抓取可乐罐环境...")
    env = simpler_env.make(
        "google_robot_pick_coke_can",
        obs_mode="rgbd",
    )

    print("开始执行抓取任务...")
    obs_list, success = solve(env, seed=42)

    if obs_list is None:
        print("❌ 任务失败")
        exit(1)

    if success:
        print(f"\n✅ 抓取成功！生成了 {len(obs_list)} 个观测")
    else:
        print(f"\n⚠️ 抓取未成功，但生成了 {len(obs_list)} 个观测")

    # 保存视频
    video_filename = output_dir / "google_robot_pick_coke_can_simple_fixed.mp4"
    print(f"\n保存视频到: {video_filename}")

    try:
        if HAS_IMAGEIO:
            frames = []
            for obs in obs_list:
                img = get_image_from_maniskill2_obs_dict(env, obs)
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                if len(img.shape) == 3 and img.shape[-1] == 4:
                    img = img[..., :3]
                frames.append(img)
            imageio.mimsave(str(video_filename), frames, fps=10)
            print(f"✅ 视频已保存: {video_filename.absolute()} ({len(frames)} 帧)")
        else:
            print("⚠️ 未安装 imageio，跳过视频保存")
    except Exception as e:
        print(f"❌ 保存视频失败: {e}")
