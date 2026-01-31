"""
Google Robot 抓取可乐罐 - 仅使用 IK（不使用 mplib planner）

这个脚本使用环境自带的 IK 控制器来实现抓取，不需要 mplib。
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

# 尝试导入 imageio 作为备选方案
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

OUTPUT_DIR = "./output"


def interpolate_qpos(start_qpos, end_qpos, steps=20):
    """在两个关节位置之间插值"""
    result = []
    for i in range(steps):
        alpha = (i + 1) / steps
        qpos = (1 - alpha) * start_qpos + alpha * end_qpos
        result.append(qpos)
    return result


def move_to_pose_with_delta(env, target_pose_at_base, gripper_state, steps=50, tolerance=0.01, debug=False):
    """使用 delta pose 控制移动到目标位置

    Args:
        target_pose_at_base: 目标位姿（在 base 坐标系）
        gripper_state: 夹爪状态
        steps: 最大步数
        tolerance: 位置容差（米）
        debug: 是否打印详细调试信息
    """
    obs_list = []
    tcp_link_name = env.unwrapped.agent.config.ee_link_name
    tcp_link = [link for link in env.unwrapped.agent.robot.get_links()
                if link.get_name() == tcp_link_name][0]
    robot_base_pose = env.unwrapped.agent.robot.pose

    for i in range(steps):
        # 获取当前 TCP 位姿（在 base 坐标系）
        current_pose_world = tcp_link.get_pose()
        current_pose_at_base = robot_base_pose.inv() * current_pose_world

        # 计算到目标的距离
        pos_error = np.linalg.norm(current_pose_at_base.p - target_pose_at_base.p)

        if debug and i < 5:
            print(f"       [调试 {i}] 当前(base)={current_pose_at_base.p}, 目标(base)={target_pose_at_base.p}")

        if pos_error < tolerance:
            print(f"       已到达目标位置 (步骤 {i+1}/{steps}, 误差 {pos_error:.4f}m)")
            break

        # 计算 delta_pos（在 base 坐标系，用于 ee_align frame）
        delta_p_base = target_pose_at_base.p - current_pose_at_base.p
        max_step = 0.01  # 每步最大移动 1cm
        if np.linalg.norm(delta_p_base) > max_step:
            delta_p_base = delta_p_base / np.linalg.norm(delta_p_base) * max_step

        # *** 关键修复：不使用 delta_rot，设为0 ***
        # 之前计算delta_rot会干扰位置控制
        delta_rot = np.zeros(3)

        # 构建 action: ee_align frame = delta_pos 在 base, delta_rot 在 TCP
        action = np.concatenate([delta_p_base, delta_rot, [gripper_state]])

        if debug and i < 5:
            print(f"       [调试 {i}] delta_p(base)={delta_p_base}, action={action[:3]}")

        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs)

        # 每10步打印一次进度
        if (i + 1) % 10 == 0:
            print(f"       步骤 {i+1}/{steps}, 距离目标: {pos_error:.4f}m")

    # 最后再检查一次
    final_pose = robot_base_pose.inv() * tcp_link.get_pose()
    final_error = np.linalg.norm(final_pose.p - target_pose_at_base.p)
    if final_error >= tolerance:
        print(f"       ⚠️  未完全到达目标 (最终误差: {final_error:.4f}m)")

    return obs_list


def solve(env, seed=None):
    """执行一次抓取任务（仅使用 IK）"""
    np.random.seed(seed)
    obs, reset_info = env.reset(seed=seed)
    
    env_unwrapped = env.unwrapped
    obj = env_unwrapped.obj
    controller = env_unwrapped.agent.controller.controllers["arm"]
    
    # 获取 TCP link
    tcp_link_name = env_unwrapped.agent.config.ee_link_name
    tcp_link = [link for link in env_unwrapped.agent.robot.get_links() 
                if link.get_name() == tcp_link_name][0]
    
    initial_tcp_pose = tcp_link.get_pose()
    robot_base_pose = env_unwrapped.agent.robot.pose
    initial_tcp_pose_at_base = robot_base_pose.inv() * initial_tcp_pose

    print(f"机器人 base 位置: {robot_base_pose.p}, 旋转: {robot_base_pose.q}")
    print(f"初始 TCP 位置（世界）: {initial_tcp_pose.p}")
    print(f"初始 TCP 位置（base）: {initial_tcp_pose_at_base.p}")
    print(f"物体位置（世界）: {obj.pose.p}")
    
    # 计算抓取姿态
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(obj)
    approaching = np.array([0, 0, -1])  # 从上往下

    # 方案1: 使用OBB自动计算 closing（可能不稳定）
    # target_closing = initial_tcp_pose.to_transformation_matrix()[:3, 1]

    # 方案2: 手动指定 closing 为水平方向（推荐）
    # 对于从上往下抓取圆柱体，closing应该是水平的
    # 这里选择Y轴方向作为closing
    target_closing = np.array([0, 1, 0])

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing, center = grasp_info["closing"], grasp_info["center"]

    # 打印计算出的抓取方向信息
    print(f"\n[调试] approaching (输入): {approaching}")
    print(f"[调试] target_closing (输入): {target_closing}")
    print(f"[调试] closing (OBB计算): {closing}")
    print(f"[调试] 检查正交性: approaching·closing = {np.dot(approaching, closing):.6f}")

    # 使用 obj.pose.p 作为抓取位置（与 panda 实现一致）
    grasp_pose_world = env_unwrapped.agent.build_grasp_pose(approaching, closing, obj.pose.p)
    grasp_pose_at_base = robot_base_pose.inv() * grasp_pose_world

    # 打印世界坐标系的 pose（与物体位置在同一坐标系，便于对比）
    print(f"抓取姿态位置（世界坐标系）: {grasp_pose_world.p}")
    print(f"抓取姿态位置（base坐标系）: {grasp_pose_at_base.p}")

    # 检查 grasp_pose 的坐标轴
    grasp_mat_world = grasp_pose_world.to_transformation_matrix()
    print(f"\n[调试] grasp_pose (世界坐标系) 的坐标轴:")
    print(f"  X轴 (ortho):      {grasp_mat_world[:3, 0]}")
    print(f"  Y轴 (closing):    {grasp_mat_world[:3, 1]}")
    print(f"  Z轴 (approaching): {grasp_mat_world[:3, 2]}")
    print(f"  期望Z轴为:         [0, 0, -1]")
    print(f"  Z轴误差:          {np.linalg.norm(grasp_mat_world[:3, 2] - approaching):.6f}")

    obs_list = [obs]

    # 定义目标位姿（base坐标系）
    print("\n定义抓取序列目标位姿...")

    # 1. 抓取位置上方（base坐标系）
    above_grasp = sapien.Pose(
        p=grasp_pose_at_base.p + np.array([0, 0, 0.1]),
        q=grasp_pose_at_base.q
    )
    print(f"  1. 上方位置 (base): {above_grasp.p}")

    # 2. 抓取位置（base坐标系）
    print(f"  2. 抓取位置 (base): {grasp_pose_at_base.p}")

    # 3. 抬起位置（base坐标系）
    lift_pose = sapien.Pose(
        p=grasp_pose_at_base.p + np.array([0, 0, 0.08]),
        q=grasp_pose_at_base.q
    )
    print(f"  3. 抬起位置 (base): {lift_pose.p}")

    # 执行抓取序列
    print("\n执行抓取动作...")

    # 1. 移动到抓取位置上方
    print("  1. 移动到抓取位置上方...")
    obs_list.extend(move_to_pose_with_delta(env, above_grasp, gripper_state=0.0, steps=80))

    # 2. 下降到抓取位置
    print("  2. 下降到抓取位置...")
    obs_list.extend(move_to_pose_with_delta(env, grasp_pose_at_base, gripper_state=0.0, steps=40, debug=True))

    # 检查到达抓取位置后的状态
    tcp_pose_after_descend = tcp_link.get_pose()
    obj_pose_after_descend = obj.pose
    print(f"     抓取位置到达后 - TCP: {tcp_pose_after_descend.p}, 物体: {obj_pose_after_descend.p}")
    print(f"     距离: {np.linalg.norm(tcp_pose_after_descend.p - obj_pose_after_descend.p):.4f} m")

    # 3. 闭合夹爪
    print("  3. 闭合夹爪...")
    for i in range(20):
        # 保持当前位置，只改变夹爪状态
        action = np.concatenate([np.zeros(6), [1.0]])
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs)

        # 每5步打印一次夹爪状态
        if i % 5 == 4:
            gripper_qpos = env.unwrapped.agent.robot.get_qpos()[-2:]
            print(f"     步骤 {i+1}/20 - 夹爪状态: {gripper_qpos}, 物体高度: {obj.pose.p[2]:.4f}")

    # 检查夹爪闭合后的状态
    gripper_final = env.unwrapped.agent.robot.get_qpos()[-2:]
    obj_pose_after_grasp = obj.pose
    print(f"     夹爪闭合后 - 夹爪: {gripper_final}, 物体位置: {obj_pose_after_grasp.p}")

    # 4. 抬起物体
    print("  4. 抬起物体...")
    obj_height_before_lift = obj.pose.p[2]
    obs_list.extend(move_to_pose_with_delta(env, lift_pose, gripper_state=1.0, steps=60))
    obj_height_after_lift = obj.pose.p[2]
    print(f"     物体高度变化: {obj_height_before_lift:.4f} -> {obj_height_after_lift:.4f} (Δ={obj_height_after_lift-obj_height_before_lift:.4f})")

    # 判断是否成功抓取
    if obj_height_after_lift - obj_height_before_lift > 0.03:
        print("     ✅ 物体被成功抬起！")
    else:
        print("     ❌ 物体没有被抓起，可能抓取失败")

    # 5. 返回初始位置（可选，暂时跳过）
    # print("  5. 返回初始位置...")
    # obs_list.extend(move_to_pose_with_delta(env, initial_tcp_pose, gripper_state=-1.0, steps=80))
    
    print(f"\n✅ 完成！生成了 {len(obs_list)} 帧")
    
    return obs_list, None


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
    
    print("开始执行抓取任务（仅使用 IK）...")
    obs_list, action_list = solve(env, seed=42)
    
    if obs_list is None:
        print("❌ 任务失败")
        exit(1)
    
    print(f"✅ 任务成功！生成了 {len(obs_list)} 个观测")
    
    # 保存视频（简化版）
    video_filename = output_dir / "google_robot_pick_coke_can_ik.mp4"
    print(f"\n保存视频到: {video_filename}")

    try:
        if HAS_IMAGEIO:
            # 使用 imageio 保存
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
            print("⚠️  未安装 imageio，跳过视频保存")
    except Exception as e:
        print(f"❌ 保存视频失败: {e}")

