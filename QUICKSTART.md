# 通用抓取脚本 - 快速开始

## 🚀 快速使用

### 基本命令

```bash
# 1. 抓取可乐罐（默认环境）
python universal_grasp_ik.py

# 2. 指定环境并查看详细输出
python universal_grasp_ik.py --env google_robot_pick_coke_can --debug

# 3. 不保存视频（更快）
python universal_grasp_ik.py --env google_robot_pick_coke_can --no-video --debug

# 4. 测试不同的物体
python universal_grasp_ik.py --env google_robot_pick_object --seed 42 --debug

# 5. 查看所有支持的环境
python universal_grasp_ik.py --help
```

## 📋 支持的环境列表

### Google Robot（所有都能用）
```bash
# 抓取各种罐子
python universal_grasp_ik.py --env google_robot_pick_coke_can
python universal_grasp_ik.py --env google_robot_pick_7up_can
python universal_grasp_ik.py --env google_robot_pick_horizontal_coke_can
python universal_grasp_ik.py --env google_robot_pick_vertical_coke_can
python universal_grasp_ik.py --env google_robot_pick_standing_coke_can

# 抓取随机物体（最好的通用性测试）
python universal_grasp_ik.py --env google_robot_pick_object

# 移动任务
python universal_grasp_ik.py --env google_robot_move_near
```

### WidowX（所有都能用）
```bash
python universal_grasp_ik.py --env widowx_spoon_on_towel
python universal_grasp_ik.py --env widowx_carrot_on_plate
python universal_grasp_ik.py --env widowx_stack_cube
python universal_grasp_ik.py --env widowx_put_eggplant_in_basket
```

## 💡 使用技巧

### 1. 首次运行建议
```bash
# 使用 --no-video 加快速度，使用 --debug 查看详细输出
python universal_grasp_ik.py --env google_robot_pick_coke_can --no-video --debug
```

### 2. 批量测试多个环境
```python
# test_multiple_envs.py
import simpler_env
from universal_grasp_ik import solve

envs = [
    "google_robot_pick_coke_can",
    "google_robot_pick_7up_can",
    "google_robot_pick_object",
]

for env_name in envs:
    print(f"\n{'='*60}")
    print(f"测试: {env_name}")
    env = simpler_env.make(env_name, obs_mode="rgbd")
    obs_list, success = solve(env, seed=42, debug=True)
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
    env.close()
```

### 3. 在 Jupyter Notebook 中使用
```python
import simpler_env
from universal_grasp_ik import solve
import matplotlib.pyplot as plt
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# 创建环境
env = simpler_env.make("google_robot_pick_coke_can", obs_mode="rgbd")

# 执行抓取
obs_list, success = solve(env, seed=42, debug=True)

# 可视化结果
if obs_list:
    # 显示第一帧和最后一帧
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    img_first = get_image_from_maniskill2_obs_dict(env, obs_list[0])
    img_last = get_image_from_maniskill2_obs_dict(env, obs_list[-1])

    axes[0].imshow(img_first)
    axes[0].set_title('初始状态')
    axes[0].axis('off')

    axes[1].imshow(img_last)
    axes[1].set_title(f'最终状态 ({"成功" if success else "失败"})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

env.close()
```

## ⚙️ 自定义抓取策略

如果某个环境抓取失败，可以尝试调整策略：

### 修改抓取方向
编辑 `universal_grasp_ik.py` 中的 `solve()` 函数：

```python
# 原来：从上往下抓取（适合大多数物体）
approaching = np.array([0, 0, -1])
target_closing = np.array([0, 1, 0])

# 选项1：从侧面抓取
approaching = np.array([1, 0, 0])  # X轴方向
target_closing = np.array([0, 0, -1])

# 选项2：从另一侧抓取
approaching = np.array([-1, 0, 0])  # -X轴方向
target_closing = np.array([0, 0, -1])
```

### 针对特定环境的策略
```python
def solve(env, seed=None, debug=False):
    # ... 前面的代码 ...

    # 根据环境名称选择策略
    env_name = env.spec.id if hasattr(env, 'spec') else ""

    if "spoon" in env_name.lower():
        # 勺子：从侧面抓取
        approaching = np.array([1, 0, 0])
        target_closing = np.array([0, 0, -1])
    elif "cube" in env_name.lower():
        # 方块：从上往下
        approaching = np.array([0, 0, -1])
        target_closing = np.array([1, 0, 0])
    else:
        # 默认策略
        approaching = np.array([0, 0, -1])
        target_closing = np.array([0, 1, 0])

    # ... 后面的代码 ...
```

## 🐛 故障排除

### 问题：环境初始化很慢

**原因**：SAPIEN 模拟器首次加载资源需要时间

**解决方案**：
- 第一次运行会慢，后续会快一些
- 使用 `--no-video` 跳过视频保存
- 在服务器上运行（GPU 加速）

### 问题：所有环境都抓取失败

**可能原因**：
1. 抓取方向不适合
2. 机器人初始位置太远
3. 物体随机位置不好

**解决方案**：
```bash
# 尝试不同的随机种子
python universal_grasp_ik.py --env google_robot_pick_coke_can --seed 1
python universal_grasp_ik.py --env google_robot_pick_coke_can --seed 2
python universal_grasp_ik.py --env google_robot_pick_coke_can --seed 3
```

### 问题：ImportError

**解决方案**：
```bash
# 确保在正确的目录
cd /data/peiqiduan/SimplerEnv

# 检查 Python 路径
python -c "import sys; print(sys.path)"

# 确保 simpler_env 可以导入
python -c "import simpler_env; print('OK')"
```

## 📊 预期结果

### 成功的输出示例
```
创建环境: google_robot_pick_coke_can
开始抓取 (种子: 42)
目标物体: opened_coke_can
机器人 base 位置: [0.35 0.2 0.079]
初始 TCP 位置（base）: [0.439 -0.218 0.962]
物体位置（世界）: [-0.263 0.398 0.920]

执行抓取...
       已到达目标位置 (步骤 45/80, 误差 0.0095m)
       已到达目标位置 (步骤 18/40, 误差 0.0098m)
   TCP到物体距离: 0.0254m
       已到达目标位置 (步骤 32/60, 误差 0.0097m)
   物体抬起高度: 5.2cm
   结果: 成功 ✅
✅ 生成了 220 帧

==================================================
最终结果: ✅ 成功
==================================================
```

### 失败的输出示例
```
创建环境: google_robot_pick_object
开始抓取 (种子: 99)
目标物体: sponge
...
   物体抬起高度: 1.1cm
   结果: 失败 ❌
✅ 生成了 220 帧

==================================================
最终结果: ❌ 失败
==================================================
```

## 📁 输出文件

### 视频保存位置
```
./output/grasp_pick_coke_can_seed42.mp4
./output/grasp_pick_object_seed123.mp4
./output/grasp_spoon_on_towel_seed1.mp4
```

### 查看视频
```bash
# 使用任何视频播放器
vlc output/grasp_pick_coke_can_seed42.mp4
mpv output/grasp_pick_coke_can_seed42.mp4
```

## 🎯 核心特性总结

✅ **任何非 drawer 环境都能用**
- Google Robot：所有 pick 和 move 环境
- WidowX：所有环境

✅ **自动检测目标物体**
- 单物体环境：`env.unwrapped.obj`
- 多物体环境：`env.unwrapped.episode_source_obj`

✅ **稳定的抓取策略**
- 默认从上往下抓取
- 可针对特定环境自定义

✅ **简单的命令行接口**
- `--env`：选择环境
- `--seed`：设置随机种子
- `--debug`：查看详细信息
- `--no-video`：加快速度

## 📖 更多信息

- 详细文档：`UNIVERSAL_GRASP_README.md`
- 源代码：`universal_grasp_ik.py`
- 测试脚本：`test_universal_grasp.py`

开始使用吧！🚀
