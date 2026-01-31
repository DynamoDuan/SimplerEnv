# Google Robot 运动规划解决方案

## pick_coke_can_ik.py

Google Robot 抓取可乐罐的解决方案 - **仅使用 IK，不使用 mplib planner**。

### 功能特点

- ✅ **不使用 mplib** - 仅使用环境自带的 IK 控制器
- ✅ 自动计算抓取姿态（基于 OBB）
- ✅ 执行完整的抓取序列
- ✅ 支持多种控制模式（自动适配）
- ✅ 保存视频（MP4 格式）
- ✅ 支持 headless 渲染

### 使用方法

```bash
cd /data/peiqiduan/SimplerEnv
python simpler_env/utils/motionplanning/google_robot/solutions/pick_coke_can_ik.py
```

### 工作原理

1. **计算抓取姿态**：基于物体的 OBB（Oriented Bounding Box）计算抓取位置和姿态
2. **IK 求解**：使用控制器的 `compute_ik()` 方法计算目标位姿对应的关节角度
3. **关节插值**：在起始和目标关节位置之间插值，生成平滑轨迹
4. **执行动作**：根据控制模式（delta pose 或 joint pos）构建并执行动作

### 控制模式支持

脚本会自动适配以下控制模式：
- `pd_joint_pos` - 直接关节位置控制
- `arm_pd_ee_delta_pose_*` - 末端执行器 delta pose 控制
- `pd_ee_target_delta_pose_*` - 目标末端执行器 delta pose 控制

### 输出文件

- `output/google_robot_pick_coke_can_ik.mp4`: 轨迹视频（左右拼接 our_rgb 和 raw_rgb）

### 任务流程

1. 移动到抓取位置上方（物体上方 10cm）
2. 下降到抓取位置
3. 闭合夹爪抓取物体
4. 抬起物体（向上 8cm）
5. 返回初始位置

### 环境要求

- `mani_skill2_real2sim`
- `simpler_env`
- `opencv-python`
- `scipy`

**注意**：不需要 `mplib`！

### 优势

相比使用 mplib planner 的版本：
- ✅ 不需要安装 mplib（避免版本兼容问题）
- ✅ 代码更简洁，易于理解和修改
- ✅ 执行速度更快（不需要路径规划）
- ✅ 适合简单抓取任务

### 限制

- 不处理复杂路径规划（如避障）
- 需要目标位姿可达（IK 有解）
- 适合简单、直接的抓取任务
