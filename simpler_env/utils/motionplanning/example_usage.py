"""
示例：在 simpler_env 中使用运动规划功能
"""
import simpler_env
from simpler_env.utils.motionplanning import (
    PandaArmMotionPlanningSolver,
    compute_grasp_info_by_obb,
    get_actor_obb,
)
import numpy as np

def example_grasp_planning():
    """示例：使用运动规划进行抓取"""
    
    # 1. 创建环境
    env = simpler_env.make("google_robot_pick_object")
    obs, info = env.reset()
    
    # 2. 创建运动规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    
    # 3. 获取物体和OBB
    obj = env.unwrapped.episode_objs[0]
    obb = get_actor_obb(obj)
    
    # 4. 计算抓取信息
    approaching = np.array([0, 0, -1])
    target_closing = env.unwrapped.agent.tcp.pose.to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.025,  # finger length
    )
    
    # 5. 构建抓取姿态
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.unwrapped.agent.build_grasp_pose(
        approaching, closing, obj.pose.p
    )
    
    # 6. 执行运动规划
    import sapien.core as sapien
    goal_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose(goal_pose)
    planner.move_to_pose(grasp_pose)
    planner.close_gripper()
    
    # 7. 获取轨迹
    obs_list = planner.obs_list
    action_list = planner.action_list
    
    print(f"生成了 {len(obs_list)} 个观测和 {len(action_list)} 个动作")
    
    return obs_list, action_list

if __name__ == "__main__":
    obs_list, action_list = example_grasp_planning()

