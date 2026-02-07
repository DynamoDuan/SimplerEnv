"""快速验证 universal_grasp_ik.py 的基本功能"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

import sys
sys.path.insert(0, '/data/peiqiduan/SimplerEnv')

print("测试 1: 导入模块...")
try:
    import simpler_env
    from universal_grasp_ik import get_target_object
    print("✅ 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

print("\n测试 2: 创建环境...")
try:
    env = simpler_env.make("google_robot_pick_coke_can", obs_mode="rgbd")
    print("✅ 环境创建成功")
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    exit(1)

print("\n测试 3: 重置环境...")
try:
    obs, info = env.reset(seed=42)
    print("✅ 环境重置成功")
except Exception as e:
    print(f"❌ 环境重置失败: {e}")
    exit(1)

print("\n测试 4: 检测目标物体...")
try:
    obj, obj_name = get_target_object(env)
    print(f"✅ 目标物体检测成功: {obj_name}")
    print(f"   物体位置: {obj.pose.p}")
except Exception as e:
    print(f"❌ 目标物体检测失败: {e}")
    exit(1)

print("\n测试 5: 测试不同环境的物体检测...")
test_envs = [
    "google_robot_pick_coke_can",
    "google_robot_pick_object",
]

for env_name in test_envs:
    try:
        test_env = simpler_env.make(env_name, obs_mode="rgbd")
        obs, info = test_env.reset(seed=42)
        obj, obj_name = get_target_object(test_env)
        print(f"✅ {env_name}: {obj_name}")
        test_env.close()
    except Exception as e:
        print(f"❌ {env_name}: {e}")

print("\n" + "="*60)
print("基本功能测试完成！")
print("="*60)
print("\n💡 提示：运行完整抓取测试:")
print("  python universal_grasp_ik.py --env google_robot_pick_coke_can --debug --no-video")
