import mplib
import numpy as np
import sapien.core as sapien
import trimesh
from mani_skill2_real2sim import format_path
from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from scipy.spatial.transform import Rotation
OPEN = 1
CLOSED = -1


class PandaArmMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.control_mode = self.base_env.control_mode
        self.ee_modes_list = ["pd_ee_target_delta_pose", "pd_ee_target_delta_pose_align", "pd_ee_target_delta_pose_align2"]
        if self.control_mode in self.ee_modes_list:
            self.arm_controller = self.env_agent.controller.controllers['arm']
            self.gripper_controller = self.env_agent.controller.controllers['gripper']
            self._target_pose = self.arm_controller.ee_pose_at_base
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.obs_list = []
        self.action_list = []

        self.base_pose = base_pose

        self.planner = self.setup_planner()

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        self.elapsed_steps = 0
        self.failure = False

    def setup_planner(self):
        # 确保 headless 模式的环境变量已设置
        import os
        import time
        
        # 设置 DISPLAY 环境变量（如果未设置或为空）
        if not os.environ.get("DISPLAY") or os.environ.get("DISPLAY") == "":
            # 尝试使用 xvfb 的显示
            try:
                import subprocess
                result = subprocess.run(['xdpyinfo', '-display', ':99'], 
                                      capture_output=True, timeout=1)
                if result.returncode == 0:
                    os.environ["DISPLAY"] = ":99"
                else:
                    # 如果没有 xvfb，设置为空字符串
                    os.environ["DISPLAY"] = ""
            except:
                os.environ["DISPLAY"] = ""
        
        if "QT_QPA_PLATFORM" not in os.environ:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        # 添加延迟，确保环境变量生效
        time.sleep(0.1)
        
        link_names = [link.get_name() for link in self.robot.get_links()]
        all_joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        
        # Panda Robot 的 arm joints（7个，不包括 gripper joints）
        arm_joint_names = [
            j.get_name() for j in self.robot.get_active_joints()
            if 'finger' not in j.get_name().lower() 
            and 'gripper' not in j.get_name().lower()
        ]
        # 如果过滤后不是7个，取前7个（通常是 arm joints）
        if len(arm_joint_names) != 7:
            arm_joint_names = all_joint_names[:7]
        
        # 保存 arm joint indices 供后续使用
        self.arm_joint_indices = [
            i for i, j in enumerate(self.robot.get_active_joints())
            if j.get_name() in arm_joint_names
        ]
        
        # 尝试多次初始化，如果失败则重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                planner = mplib.Planner(
                    urdf=format_path(str(self.env_agent.urdf_path)),
                    srdf=format_path(str(self.env_agent.urdf_path).replace(".urdf", ".srdf")),
                    user_link_names=link_names,
                    user_joint_names=arm_joint_names,  # 只使用 arm joints
                    move_group="panda_hand_tcp",
                    joint_vel_limits=np.ones(7) * self.joint_vel_limits,
                    joint_acc_limits=np.ones(7) * self.joint_acc_limits,
                    verbose=False,  # 禁用详细输出
                )
                planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
                return planner
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 等待后重试
                    continue
                else:
                    raise RuntimeError(f"Failed to initialize mplib.Planner after {max_retries} attempts: {e}")

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_ee_target_delta_pose":
                prev_ee_pose_at_base = self._target_pose
                self._target_pose = self.arm_controller.compute_fk(qpos)
                
                delta_pose = prev_ee_pose_at_base.inv() * self._target_pose
                delta_pos = delta_pose.p
                delta_quat = delta_pose.q
                delta_rot = Rotation.from_quat(delta_quat, scalar_first=True).as_rotvec()
                action = np.hstack([delta_pos, delta_rot, self.gripper_state])
            elif self.control_mode == "pd_ee_target_delta_pose_align2":
                prev_ee_pose_at_base = self._target_pose
                cur_ee_pose_at_base = self.arm_controller.compute_fk(self.arm_controller.qpos)
                cur_ee_pos_at_base = sapien.Pose(p=cur_ee_pose_at_base.p)
                self._target_pose = self.arm_controller.compute_fk(qpos)
                
                delta_pose = cur_ee_pos_at_base.inv() * (self._target_pose * prev_ee_pose_at_base.inv()) * cur_ee_pos_at_base
                delta_pos = delta_pose.p
                delta_quat = delta_pose.q
                delta_rot = Rotation.from_quat(delta_quat, scalar_first=True).as_rotvec()
                action = np.hstack([delta_pos, delta_rot, self.gripper_state])
            else:
                assert 0
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_list.append(obs)
            self.action_list.append(action)
            self.elapsed_steps += 1

        return True

    def move_to_pose(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        if pose.p[-1] < 0.01:
            pose = sapien.Pose(np.array([pose.p[0], pose.p[1], 0.01]), pose.q)
        if not self.move_to_pose_with_RRTConnect(pose, dry_run, refine_steps):
            result = self.move_to_pose_with_screw(pose, dry_run, refine_steps)
            if result:
                return True
            else:
                self.failure = True
                return False

        return True

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = sapien.Pose(p=pose.p, q=pose.q)
        # 只使用 arm joints，不包括 gripper
        current_qpos = self.robot.get_qpos()
        current_arm_qpos = current_qpos[self.arm_joint_indices]
        
        result = self.planner.plan_pose(
            goal_pose=mplib.Pose(pose.p, pose.q),
            current_qpos=current_arm_qpos,
            time_step=self.base_env.control_timestep,
            wrt_world=True,
        )
        if result["status"] != "Success":
            result = self.planner.plan_pose(
                goal_pose=mplib.Pose(pose.p, pose.q),
                current_qpos=current_arm_qpos,
                time_step=self.base_env.control_timestep,
                wrt_world=True,
            )
            if result["status"] != "Success":
                return False
            
        if result['position'].shape[0] > 80:
            return False

        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):

        # try screw two times before giving up
        pose = sapien.Pose(p=pose.p , q=pose.q)
        # 只使用 arm joints，不包括 gripper
        current_qpos = self.robot.get_qpos()
        current_arm_qpos = current_qpos[self.arm_joint_indices]
        
        result = self.planner.plan_screw(
            goal_pose = mplib.Pose(pose.p, pose.q),
            current_qpos = current_arm_qpos,
            time_step=self.base_env.control_timestep,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                goal_pose = mplib.Pose(pose.p, pose.q),
                current_qpos = current_arm_qpos,
                time_step=self.base_env.control_timestep,
            )
            if result["status"] != "Success":
                return False
        if result['position'].shape[0] > 80:
            return False
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[:-2]
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode in self.ee_modes_list:
                action = np.hstack([np.zeros(6), self.gripper_state])
            else:
                assert 0
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_list.append(obs)
            self.action_list.append(action)
            self.elapsed_steps += 1

    def close_gripper(self, t=6, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[:-2]
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode in self.ee_modes_list:
                action = np.hstack([np.zeros(6), self.gripper_state])
            else:
                assert 0
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_list.append(obs)
            self.action_list.append(action)
            self.elapsed_steps += 1

    def close(self):
        pass