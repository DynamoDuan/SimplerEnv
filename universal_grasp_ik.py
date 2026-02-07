"""
通用物体抓取脚本 - 仅使用 IK（不使用 mplib planner）

支持所有 SimplerEnv 中的抓取类环境（排除 drawer 相关环境）
使用简单稳定的从上往下抓取策略
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

import argparse
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


def compute_target_closing_from_obb(obb, approaching=None):
    """
    根据OBB自动计算target_closing方向
    
    Args:
        obb: trimesh.primitives.Box - 物体的OBB
        approaching: np.array - 接近方向，默认[0,0,-1]（从上往下）
    
    Returns:
        target_closing: np.array - 计算得到的closing方向
    """
    if approaching is None:
        approaching = np.array([0, 0, -1])
    
    # 获取OBB的变换矩阵和尺寸
    T = np.array(obb.primitive.transform)
    extents = np.array(obb.primitive.extents)
    
    # OBB的三个轴方向（在变换矩阵的前3列）
    axes = T[:3, :3]  # (3, 3)，每列是一个轴的方向
    
    # 找到最接近approaching方向的轴（这个轴会被用作approaching，我们要排除它）
    angles = approaching @ axes  # [3]，每个轴与approaching的夹角
    inds_sorted = np.argsort(np.abs(angles))
    approaching_axis_idx = inds_sorted[-1]  # 最接近approaching的轴
    
    # 剩余的两个轴（垂直于approaching的轴）
    remaining_indices = [i for i in range(3) if i != approaching_axis_idx]
    remaining_axes = axes[:, remaining_indices]  # (3, 2)
    remaining_extents = extents[remaining_indices]  # (2,)
    
    # 策略1: 选择最短的轴作为closing方向（通常更适合抓取）
    # 因为较短的轴意味着物体在这个方向上更窄，更容易抓取
    shorter_axis_idx = np.argmax(remaining_extents)
    target_closing = remaining_axes[:, shorter_axis_idx]
    
    # # 策略2（备选）: 如果两个轴长度接近，选择最接近水平方向的轴
    # # 这对于圆柱体等物体可能更合适
    # if len(remaining_extents) == 2:
    #     extent_ratio = min(remaining_extents) / max(remaining_extents)
    #     if extent_ratio > 0.8:  # 如果两个轴长度接近（比例>0.8）
    #         # 选择最接近水平方向的轴（z分量最小的）
    #         horizontal_scores = np.abs(remaining_axes[2, :])  # z分量的绝对值
    #         more_horizontal_idx = np.argmin(horizontal_scores)
    #         target_closing = remaining_axes[:, more_horizontal_idx]
    
    # 归一化
    target_closing = target_closing / np.linalg.norm(target_closing)
    
    # 将closing轴投影到xy平面（z分量设为0）
    target_closing_xy = np.array([target_closing[0], target_closing[1], 0.0])
    
    # 如果投影后向量长度接近0，说明原向量几乎垂直，使用y轴方向作为默认值
    norm_xy = np.linalg.norm(target_closing_xy)
    if norm_xy < 1e-6:
        target_closing_xy = np.array([0.0, 1.0, 0.0])
    else:
        target_closing_xy = target_closing_xy / norm_xy
    
    return target_closing_xy


def get_target_object(env):
    """
    从环境中获取目标物体

    Returns:
        sapien.Actor: 目标物体
        str: 物体名称
    """
    env_unwrapped = env.unwrapped

    # 尝试不同的属性名称
    if hasattr(env_unwrapped, 'obj') and env_unwrapped.obj is not None:
        # GraspSingle 类环境
        return env_unwrapped.obj, env_unwrapped.obj.name
    elif hasattr(env_unwrapped, 'episode_source_obj') and env_unwrapped.episode_source_obj is not None:
        # MoveNear/PutOn 类环境
        return env_unwrapped.episode_source_obj, env_unwrapped.episode_source_obj.name
    else:
        raise ValueError("无法找到目标物体。该环境可能不支持物体抓取。")


def move_to_pose_with_delta(env, target_pose_at_base, gripper_state, steps=50, tolerance=0.01, debug=False):
    """使用 delta pose 控制移动到目标位置"""
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

        if pos_error < tolerance:
            if debug:
                print(f"       已到达目标位置 (步骤 {i+1}/{steps}, 误差 {pos_error:.4f}m)")
            break

        # 计算 delta_pos（在 base 坐标系）
        delta_p_base = target_pose_at_base.p - current_pose_at_base.p
        max_step = 0.01  # 每步最大移动 1cm
        if np.linalg.norm(delta_p_base) > max_step:
            delta_p_base = delta_p_base / np.linalg.norm(delta_p_base) * max_step

        # 不使用 delta_rot
        delta_rot = np.zeros(3)

        # 构建 action
        action = np.concatenate([delta_p_base, delta_rot, [gripper_state]])
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs)

    return obs_list


def solve(env, seed=None, debug=False):
    """执行一次抓取任务（仅使用 IK）"""
    if seed is not None:
        np.random.seed(seed)
        obs, reset_info = env.reset(seed=seed)
    else:
        obs, reset_info = env.reset()

    env_unwrapped = env.unwrapped

    # 获取目标物体
    try:
        obj, obj_name = get_target_object(env)
        if debug:
            print(f"目标物体: {obj_name}")
    except ValueError as e:
        print(f"错误: {e}")
        return None, None

    # 获取 TCP link
    tcp_link_name = env_unwrapped.agent.config.ee_link_name
    tcp_link = [link for link in env_unwrapped.agent.robot.get_links()
                if link.get_name() == tcp_link_name][0]

    initial_tcp_pose = tcp_link.get_pose()
    robot_base_pose = env_unwrapped.agent.robot.pose
    initial_tcp_pose_at_base = robot_base_pose.inv() * initial_tcp_pose

    if debug:
        print(f"机器人 base 位置: {robot_base_pose.p}")
        print(f"初始 TCP 位置（base）: {initial_tcp_pose_at_base.p}")
        print(f"物体位置（世界）: {obj.pose.p}")

    # 计算抓取姿态 - 使用简单的从上往下策略
    FINGER_LENGTH = 0.080
    obb = get_actor_obb(obj)
    approaching = np.array([0, 0, -1])  # 从上往下
    
    # 根据OBB自动计算target_closing方向
    target_closing = compute_target_closing_from_obb(obb, approaching=approaching)
    
    if debug:
        print(f"OBB尺寸: {obb.primitive.extents}")
        print(f"计算得到的target_closing: {target_closing}")

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing = grasp_info["closing"]

    # 使用 obj.pose.p 作为抓取位置
    grasp_pose_world = env_unwrapped.agent.build_grasp_pose(approaching, closing, obj.pose.p)
    grasp_pose_at_base = robot_base_pose.inv() * grasp_pose_world

    if debug:
        print(f"抓取位置（base坐标系）: {grasp_pose_at_base.p}")

    obs_list = [obs]

    # 定义抓取序列目标位姿（base坐标系）
    # 1. 抓取位置上方
    above_grasp = sapien.Pose(
        p=grasp_pose_at_base.p + np.array([0, 0, 0.1]),
        q=grasp_pose_at_base.q
    )

    # 2. 抓取位置（grasp_pose_at_base）

    # 3. 抬起位置
    lift_pose = sapien.Pose(
        p=grasp_pose_at_base.p + np.array([0, 0, 0.08]),
        q=grasp_pose_at_base.q
    )

    # 执行抓取序列
    if debug:
        print("\n执行抓取...")

    # 1. 移动到抓取位置上方
    obs_list.extend(move_to_pose_with_delta(env, above_grasp, gripper_state=0.0, steps=80, debug=debug))

    # 2. 下降到抓取位置
    obs_list.extend(move_to_pose_with_delta(env, grasp_pose_at_base, gripper_state=0.0, steps=40, debug=debug))

    # 检查位置
    obj_height_before_grasp = obj.pose.p[2]
    if debug:
        tcp_pose = tcp_link.get_pose()
        distance = np.linalg.norm(tcp_pose.p - obj.pose.p)
        print(f"   TCP到物体距离: {distance:.4f}m")

    # 3. 闭合夹爪
    for i in range(20):
        action = np.concatenate([np.zeros(6), [1.0]])
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs)

    # 4. 抬起物体
    obs_list.extend(move_to_pose_with_delta(env, lift_pose, gripper_state=1.0, steps=60, debug=debug))
    obj_height_after_lift = obj.pose.p[2]
    height_delta = obj_height_after_lift - obj_height_before_grasp

    # 判断成功
    success_threshold = 0.03  # 3cm
    success = height_delta > success_threshold

    if debug:
        print(f"   物体抬起高度: {height_delta*100:.1f}cm")
        print(f"   结果: {'成功 ✅' if success else '失败 ❌'}")

    return obs_list, success


if __name__ == "__main__":
    # 获取所有支持的环境
    from simpler_env import ENVIRONMENTS

    # 排除 drawer 相关环境
    supported_envs = [env for env in ENVIRONMENTS if "drawer" not in env.lower()]

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="通用物体抓取 - IK 控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
支持的环境:
{chr(10).join(f"  - {env}" for env in supported_envs)}
"""
    )
    parser.add_argument(
        "--env",
        type=str,
        default="google_robot_pick_coke_can",
        help="环境名称 (默认: google_robot_pick_coke_can)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        nargs='?',
        help="随机种子 (可选，不设置则使用随机值)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试输出"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不保存视频"
    )
    args = parser.parse_args()

    # 检查环境是否支持
    if args.env not in ENVIRONMENTS:
        print(f"❌ 错误: 环境 '{args.env}' 不存在")
        print(f"\n可用环境: {ENVIRONMENTS}")
        exit(1)

    if "drawer" in args.env.lower():
        print(f"❌ 错误: 该脚本不支持 drawer 相关环境")
        exit(1)

    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建环境
    print(f"创建环境: {args.env}")
    env = simpler_env.make(args.env, obs_mode="rgbd")

    seed_str = f"种子: {args.seed}" if args.seed is not None else "随机种子"
    print(f"开始抓取 ({seed_str})")
    obs_list, success = solve(env, seed=args.seed, debug=args.debug)

    if obs_list is None:
        print("❌ 任务执行失败")
        exit(1)

    print(f"✅ 生成了 {len(obs_list)} 帧")

    # 保存视频
    if not args.no_video:
        env_short_name = args.env.replace("google_robot_", "").replace("widowx_", "")
        video_filename = output_dir / f"grasp_{env_short_name}_seed{args.seed}.mp4"
        print(f"保存视频到: {video_filename}")

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
                print(f"✅ 视频已保存 ({len(frames)} 帧)")
            else:
                print("⚠️  未安装 imageio，跳过视频保存")
        except Exception as e:
            print(f"❌ 保存视频失败: {e}")

    print(f"\n{'='*50}")
    print(f"最终结果: {'✅ 成功' if success else '❌ 失败'}")
    print(f"{'='*50}")
