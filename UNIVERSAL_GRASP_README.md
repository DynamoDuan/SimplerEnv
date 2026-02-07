# 通用物体抓取脚本 (Universal Grasp IK)

## 简介

`universal_grasp_ik.py` 是一个通用的物体抓取脚本，可以在 SimplerEnv 中的任何抓取类环境中工作（除了 drawer 相关环境）。

### 主要特性

- ✅ **自动物体检测**：自动识别环境中的目标物体
- ✅ **通用抓取策略**：使用稳定的从上往下抓取方法
- ✅ **支持多种环境**：支持所有非drawer环境
- ✅ **仅使用 IK**：不依赖 mplib 运动规划库
- ✅ **简单易用**：命令行接口简洁明了

## 支持的环境

脚本支持以下类型的环境：

### Google Robot 环境
- `google_robot_pick_coke_can` - 抓取可乐罐
- `google_robot_pick_7up_can` - 抓取7up罐
- `google_robot_pick_horizontal_coke_can` - 抓取水平放置的可乐罐
- `google_robot_pick_vertical_coke_can` - 抓取垂直放置的可乐罐
- `google_robot_pick_standing_coke_can` - 抓取站立的可乐罐
- `google_robot_pick_object` - 抓取随机物体
- `google_robot_move_near` - 移动到物体附近

### WidowX 环境
- `widowx_spoon_on_towel` - 将勺子放在毛巾上
- `widowx_carrot_on_plate` - 将胡萝卜放在盘子上
- `widowx_stack_cube` - 堆叠方块
- `widowx_put_eggplant_in_basket` - 将茄子放入篮子

**注意**：不支持 drawer 相关的环境（开关抽屉、放入抽屉等）。

## 使用方法

### 基本用法

```bash
# 使用默认环境（google_robot_pick_coke_can）
python universal_grasp_ik.py

# 指定环境
python universal_grasp_ik.py --env google_robot_pick_object

# 指定种子
python universal_grasp_ik.py --env google_robot_pick_7up_can --seed 123

# 启用调试输出
python universal_grasp_ik.py --env google_robot_pick_coke_can --debug

# 不保存视频（加快运行速度）
python universal_grasp_ik.py --env google_robot_pick_coke_can --no-video
```

### 命令行参数

- `--env ENV_NAME`：环境名称（默认：`google_robot_pick_coke_can`）
- `--seed SEED`：随机种子（默认：42）
- `--debug`：启用调试输出
- `--no-video`：不保存视频（加快运行速度）

### 查看支持的环境列表

```bash
python universal_grasp_ik.py --help
```

## 工作原理

### 1. 自动物体检测

脚本会自动检测环境中的目标物体：

```python
# 对于 GraspSingle 类环境（单物体）
obj = env.unwrapped.obj

# 对于 MoveNear/PutOn 类环境（多物体）
obj = env.unwrapped.episode_source_obj
```

### 2. 抓取策略

使用简单稳定的**从上往下**抓取策略：

- **接近方向 (approaching)**：`[0, 0, -1]` - 从上往下
- **闭合方向 (closing)**：`[0, 1, 0]` - 水平方向

这种策略适用于大多数桌面物体（罐子、水果、方块等）。

### 3. 抓取序列

1. **移动到物体上方**（高度 +10cm）
2. **下降到抓取位置**
3. **闭合夹爪**（20步）
4. **抬起物体**（高度 +8cm）

### 4. 成功判断

如果物体被抬起超过 3cm，则判定为成功。

## 输出

### 控制台输出

```
创建环境: google_robot_pick_coke_can
开始抓取 (种子: 42)
✅ 生成了 220 帧
保存视频到: output/grasp_pick_coke_can_seed42.mp4
✅ 视频已保存 (220 帧)

==================================================
最终结果: ✅ 成功
==================================================
```

### 调试模式输出

启用 `--debug` 后会显示详细信息：

```
目标物体: opened_coke_can
机器人 base 位置: [0.35 0.2 0.079]
初始 TCP 位置（base）: [0.439 -0.218 0.962]
物体位置（世界）: [-0.263 0.398 0.920]
抓取位置（base坐标系）: [-0.613 0.198 0.841]

执行抓取...
       已到达目标位置 (步骤 45/80, 误差 0.0095m)
       已到达目标位置 (步骤 18/40, 误差 0.0098m)
   TCP到物体距离: 0.0254m
       已到达目标位置 (步骤 32/60, 误差 0.0097m)
   物体抬起高度: 5.2cm
   结果: 成功 ✅
```

### 视频文件

默认保存到 `./output/` 目录：

- 格式：`grasp_{env_short_name}_seed{seed}.mp4`
- 帧率：10 FPS
- 示例：`output/grasp_pick_coke_can_seed42.mp4`

## 代码示例

### 在自己的代码中使用

```python
import simpler_env
from universal_grasp_ik import solve

# 创建环境
env = simpler_env.make("google_robot_pick_object", obs_mode="rgbd")

# 执行抓取
obs_list, success = solve(env, seed=42, debug=True)

if success:
    print(f"成功！生成了 {len(obs_list)} 帧")
else:
    print("抓取失败")

# 关闭环境
env.close()
```

### 批量测试多个环境

```python
import simpler_env
from simpler_env import ENVIRONMENTS
from universal_grasp_ik import solve

# 排除 drawer 环境
test_envs = [env for env in ENVIRONMENTS if "drawer" not in env.lower()]

results = {}
for env_name in test_envs[:5]:  # 测试前5个环境
    print(f"\n测试: {env_name}")
    env = simpler_env.make(env_name, obs_mode="rgbd")
    obs_list, success = solve(env, seed=42)
    results[env_name] = success
    env.close()

# 打印结果
print("\n=== 测试结果 ===")
for env_name, success in results.items():
    status = "✅" if success else "❌"
    print(f"{status} {env_name}")
```

## 限制和注意事项

### 不支持的环境

- ❌ 所有包含 "drawer" 的环境（需要不同的策略）
- ❌ 纯导航环境（不涉及抓取）

### 性能考虑

- **运行时间**：每个环境约 30-60 秒（取决于硬件）
- **步数**：总共约 200 步（80 + 40 + 20 + 60）
- **视频保存**：可能需要额外 5-10 秒

### 可能的失败原因

1. **物体太远**：TCP 无法到达物体位置
2. **物体形状特殊**：从上往下抓取不适用
3. **夹爪无法闭合**：物体太大或形状不规则
4. **碰撞**：机器人与场景发生碰撞

## 自定义和扩展

### 修改抓取策略

如果需要为特定物体定制抓取方向，修改 `solve()` 函数中的策略：

```python
# 原来：固定从上往下
approaching = np.array([0, 0, -1])
target_closing = np.array([0, 1, 0])

# 修改为：从侧面抓取
approaching = np.array([1, 0, 0])  # 从X轴正方向
target_closing = np.array([0, 0, -1])  # Z轴负方向
```

### 调整抓取高度

修改抓取序列中的高度偏移：

```python
# 上方高度（默认 +10cm）
above_grasp = sapien.Pose(
    p=grasp_pose_at_base.p + np.array([0, 0, 0.15]),  # 改为 +15cm
    q=grasp_pose_at_base.q
)

# 抬起高度（默认 +8cm）
lift_pose = sapien.Pose(
    p=grasp_pose_at_base.p + np.array([0, 0, 0.12]),  # 改为 +12cm
    q=grasp_pose_at_base.q
)
```

### 修改成功判断阈值

```python
# 默认：3cm
success_threshold = 0.03

# 更严格：5cm
success_threshold = 0.05

# 更宽松：2cm
success_threshold = 0.02
```

## 依赖

- `simpler_env`
- `sapien`
- `numpy`
- `imageio` (可选，用于保存视频)

安装 imageio：
```bash
pip install imageio[ffmpeg]
```

## 故障排除

### 问题：ImportError: cannot import name 'get_actor_obb'

**解决方案**：确保 `simpler_env/utils/motionplanning/utils.py` 存在且包含该函数。

### 问题：视频无法保存

**解决方案**：
```bash
pip install imageio[ffmpeg]
```

或使用 `--no-video` 跳过视频保存。

### 问题：运行时间过长

**解决方案**：
1. 使用 `--no-video` 跳过视频保存
2. 减少步数（修改 `solve()` 函数中的 `steps` 参数）
3. 降低环境的模拟频率（需要修改环境配置）

## 贡献

欢迎提交 issue 和 pull request 来改进这个脚本！

可能的改进方向：
- 支持更多抓取策略（侧面抓取、多指抓取等）
- 自动检测物体形状并选择最佳策略
- 添加碰撞检测和避障
- 支持双臂协作
- 添加更多调试可视化

## 许可

与 SimplerEnv 项目保持一致。
