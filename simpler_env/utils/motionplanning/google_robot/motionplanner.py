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


class GoogleRobotArmMotionPlanningSolver:
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
        
        # Google Robot 使用 arm_pd_ee_delta_pose_align 等控制模式
        self.ee_modes_list = [
            "arm_pd_ee_delta_pose_align",
            "arm_pd_ee_delta_pose_align_interpolate_by_planner",
            "pd_ee_target_delta_pose",
            "pd_ee_target_delta_pose_align",
            "pd_ee_target_delta_pose_align2"
        ]
        
        if any(mode in self.control_mode for mode in self.ee_modes_list):
            self.arm_controller = self.env_agent.controller.controllers['arm']
            self.gripper_controller = self.env_agent.controller.controllers['gripper']
            if hasattr(self.arm_controller, 'ee_pose_at_base'):
                self._target_pose = self.arm_controller.ee_pose_at_base
            else:
                # 如果没有 ee_pose_at_base，使用 tcp pose
                self._target_pose = self.env_agent.tcp.pose
        
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.obs_list = []
        self.action_list = []

        self.base_pose = base_pose if base_pose is not None else self.robot.pose

        # 先设置这些属性，setup_planner 会用到
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        self.elapsed_steps = 0
        self.failure = False

        self.planner = self.setup_planner()

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
        
        # Google Robot 的 arm joints（不包括 finger、head 和 wheel）
        arm_joint_names = [
            j.get_name() for j in self.robot.get_active_joints()
            if 'finger' not in j.get_name().lower() 
            and 'head' not in j.get_name().lower()
            and 'wheel' not in j.get_name().lower()
        ]
        num_arm_joints = len(arm_joint_names)
        
        # 保存 arm joint indices 供后续使用
        self.arm_joint_indices = [
            i for i, j in enumerate(self.robot.get_active_joints())
            if j.get_name() in arm_joint_names
        ]
        
        # 检查 SRDF 文件是否存在
        urdf_path = format_path(str(self.env_agent.urdf_path))
        srdf_path = urdf_path.replace(".urdf", ".srdf")
        import os
        if not os.path.exists(srdf_path):
            # 如果 SRDF 不存在，尝试使用 None
            srdf_path = None
        
        # 尝试多次初始化，如果失败则重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Google Robot 的 move_group 通常是 link_gripper_tcp 或类似的名称
                # 尝试使用 ee_link_name
                move_group = getattr(self.env_agent.config, 'ee_link_name', 'link_gripper_tcp')
                
                if self.print_env_info:
                    print(f"  Initializing planner with move_group={move_group}, num_arm_joints={num_arm_joints}")
                    print(f"  URDF: {urdf_path}")
                    print(f"  SRDF: {srdf_path if srdf_path else 'None'}")
                
                # 使用 user_joint_names 指定只使用 arm joints，确保 planner 期望的关节数与传入的 qpos 匹配
                # 注意：如果同时提供 SRDF 和 user_joint_names，SRDF 可能会覆盖 user_joint_names
                # 因此优先尝试不使用 SRDF，仅使用 user_joint_names
                planner_kwargs_base = {
                    "urdf": urdf_path,
                    "move_group": move_group,
                    "user_link_names": link_names,
                    "user_joint_names": arm_joint_names,  # 只使用 arm joints
                    "verbose": False,
                }
                
                # 添加关节速度限制和加速度限制
                if num_arm_joints > 0:
                    planner_kwargs_base["joint_vel_limits"] = np.ones(num_arm_joints) * self.joint_vel_limits
                    planner_kwargs_base["joint_acc_limits"] = np.ones(num_arm_joints) * self.joint_acc_limits
                
                # 首先尝试不使用 SRDF（优先使用 user_joint_names）
                planner_kwargs = planner_kwargs_base.copy()
                planner = None
                planner_error = None
                
                if self.print_env_info:
                    print(f"  Attempting to create planner WITHOUT SRDF (using user_joint_names)...")
                    print(f"  Using {num_arm_joints} arm joints: {arm_joint_names}")
                
                try:
                    planner = mplib.Planner(**planner_kwargs)
                    if self.print_env_info:
                        print(f"  ✅ Planner created successfully without SRDF!")
                except Exception as e:
                    planner_error = e
                    if self.print_env_info:
                        print(f"  ⚠️  Planner creation without SRDF failed: {e}")
                        print(f"  Will try with SRDF as fallback...")
                
                # 如果失败，尝试使用 SRDF（但 user_joint_names 可能被忽略）
                if planner is None:
                    planner_kwargs = planner_kwargs_base.copy()
                    if srdf_path is not None and os.path.exists(srdf_path):
                        planner_kwargs["srdf"] = srdf_path
                        if self.print_env_info:
                            print(f"  Attempting with SRDF: {srdf_path}")
                    else:
                        # 如果没有标准 SRDF，尝试查找 _mplib.srdf 文件
                        mplib_srdf_path = urdf_path.replace(".urdf", "_mplib.srdf")
                        if os.path.exists(mplib_srdf_path):
                            planner_kwargs["srdf"] = mplib_srdf_path
                            if self.print_env_info:
                                print(f"  Attempting with _mplib.srdf: {mplib_srdf_path}")
                    
                    if "srdf" in planner_kwargs:
                        try:
                            planner = mplib.Planner(**planner_kwargs)
                            if self.print_env_info:
                                print(f"  ✅ Planner created with SRDF!")
                                print(f"  ⚠️  Warning: SRDF may override user_joint_names - verify joint count matches!")
                        except Exception as e:
                            if self.print_env_info:
                                print(f"  ❌ Planner creation with SRDF also failed: {e}")
                            raise RuntimeError(f"Failed to initialize planner with or without SRDF. Last error: {e}")
                    else:
                        # 没有 SRDF 且第一次尝试也失败
                        raise RuntimeError(f"Failed to initialize planner without SRDF: {planner_error}")
                
                # 验证 planner 期望的关节数是否匹配
                try:
                    # 尝试获取 planner 的关节信息
                    if hasattr(planner, 'get_joint_names'):
                        planner_joint_names = planner.get_joint_names()
                        planner_joint_count = len(planner_joint_names) if planner_joint_names else None
                        if planner_joint_count is not None:
                            if self.print_env_info:
                                print(f"  Planner expects {planner_joint_count} joints: {planner_joint_names}")
                            if planner_joint_count != num_arm_joints:
                                if self.print_env_info:
                                    print(f"  ⚠️  WARNING: Planner expects {planner_joint_count} joints but we're providing {num_arm_joints}!")
                                    print(f"  This may cause shape mismatch errors. Consider checking SRDF configuration.")
                except Exception as e:
                    if self.print_env_info:
                        print(f"  Could not verify planner joint count: {e}")
                
                if self.print_env_info:
                    print(f"  Setting base pose...")
                try:
                    planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
                    if self.print_env_info:
                        print(f"  ✅ Base pose set!")
                except Exception as e:
                    if self.print_env_info:
                        print(f"  ⚠️  Base pose setting failed (may be OK): {e}")
                    # 继续执行，base pose 设置失败可能不影响
                
                if self.print_env_info:
                    print(f"  ✅ Planner initialized successfully!")
                return planner
            except Exception as e:
                if self.print_env_info:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 等待后重试
                    continue
                else:
                    raise RuntimeError(f"Failed to initialize mplib.Planner after {max_retries} attempts: {e}")

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]

            # Google Robot 的控制模式处理
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif any(mode in self.control_mode for mode in self.ee_modes_list):
                # 使用 end-effector delta pose 控制
                prev_ee_pose_at_base = self._target_pose

                # 计算前向运动学
                if hasattr(self.arm_controller, 'compute_fk'):
                    # 获取当前 finger joint 位置
                    current_qpos = self.robot.get_qpos()
                    # 创建完整的 qpos: arm joints + finger joints
                    # 需要根据实际的关节顺序来拼接
                    full_qpos_for_fk = np.concatenate([qpos, current_qpos[-2:]])
                    self._target_pose = self.arm_controller.compute_fk(full_qpos_for_fk)
                else:
                    # 如果没有 compute_fk，直接使用当前 TCP pose（保持不动）
                    tcp_link_name = self.env_agent.config.ee_link_name
                    tcp_link = [link for link in self.robot.get_links() if link.get_name() == tcp_link_name][0]
                    self._target_pose = tcp_link.get_pose()
                
                delta_pose = prev_ee_pose_at_base.inv() * self._target_pose
                delta_pos = delta_pose.p
                delta_quat = delta_pose.q
                delta_rot = Rotation.from_quat(delta_quat, scalar_first=True).as_rotvec()
                action = np.hstack([delta_pos, delta_rot, self.gripper_state])
            else:
                # 默认：使用零动作，只控制 gripper
                action = np.hstack([np.zeros(6), self.gripper_state])
            
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
        # 获取当前 arm joints 的 qpos（不包括 finger 和 head）
        current_qpos = self.robot.get_qpos()
        current_arm_qpos = current_qpos[self.arm_joint_indices]
        
        if self.debug:
            print(f"  Planning to pose: {pose.p}")
            print(f"  Current arm qpos shape: {current_arm_qpos.shape}, values: {current_arm_qpos}")
            print(f"  Calling plan_pose...")
        
        try:
            import time
            start_time = time.time()
            result = self.planner.plan_pose(
                goal_pose=mplib.Pose(pose.p, pose.q),
                current_qpos=current_arm_qpos,
                time_step=self.base_env.control_timestep,
                wrt_world=True,  # 使用世界坐标
            )
            elapsed = time.time() - start_time
            
            if self.debug:
                print(f"  Planning completed in {elapsed:.2f}s")
                print(f"  Planning result: {result['status']}, steps: {result['position'].shape[0] if 'position' in result else 0}")
        except Exception as e:
            if self.debug:
                print(f"  Planning error: {e}")
                import traceback
                traceback.print_exc()
            return False
        
        if result["status"] != "Success":
            if self.debug:
                print(f"  First RRT planning attempt failed: {result['status']}, retrying...")
            try:
                result = self.planner.plan_pose(
                    goal_pose=mplib.Pose(pose.p, pose.q),
                    current_qpos=current_arm_qpos,
                    time_step=self.base_env.control_timestep,
                    wrt_world=True,
                )
            except Exception as e:
                if self.debug:
                    print(f"  Retry failed: {e}")
                return False
            
            if result["status"] != "Success":
                if self.debug:
                    print(f"  RRT planning failed after retry: {result['status']}")
                return False
            
        if result['position'].shape[0] > 80:
            return False

        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        # try screw two times before giving up
        pose = sapien.Pose(p=pose.p , q=pose.q)
        
        # 获取当前 arm joints 的 qpos
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
                if self.debug:
                    print(f"  Screw planning failed: {result['status']}")
                return False
        if result['position'].shape[0] > 80:
            if self.debug:
                print(f"  Screw planning path too long: {result['position'].shape[0]} steps")
            return False
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self):
        self.gripper_state = OPEN
        # Google Robot 的 gripper 控制可能需要特殊处理
        # 根据控制模式发送相应的动作
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                qpos = self.robot.get_qpos()
                action = np.hstack([qpos, self.gripper_state])
            elif any(mode in self.control_mode for mode in self.ee_modes_list):
                action = np.hstack([np.zeros(6), self.gripper_state])
            else:
                action = np.hstack([np.zeros(6), self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_list.append(obs)
            self.action_list.append(action)
            self.elapsed_steps += 1

    def close_gripper(self, t=6, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                qpos = self.robot.get_qpos()
                action = np.hstack([qpos, self.gripper_state])
            elif any(mode in self.control_mode for mode in self.ee_modes_list):
                action = np.hstack([np.zeros(6), self.gripper_state])
            else:
                action = np.hstack([np.zeros(6), self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_list.append(obs)
            self.action_list.append(action)
            self.elapsed_steps += 1

    def close(self):
        pass

