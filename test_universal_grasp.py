"""快速测试通用抓取脚本"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

import sys
sys.path.insert(0, '/data/peiqiduan/SimplerEnv')

import simpler_env
from universal_grasp_ik import solve

# 测试不同的环境
test_envs = [
    "google_robot_pick_coke_can",
    "google_robot_pick_object",
    "widowx_spoon_on_towel",
]

for env_name in test_envs:
    print(f"\n{'='*60}")
    print(f"测试环境: {env_name}")
    print(f"{'='*60}")

    try:
        env = simpler_env.make(env_name, obs_mode="rgbd")
        obs_list, success = solve(env, seed=42, debug=True)

        if obs_list is not None:
            print(f"✅ {env_name}: {'成功' if success else '失败'} ({len(obs_list)} 帧)")
        else:
            print(f"❌ {env_name}: 执行失败")

        env.close()
    except Exception as e:
        print(f"❌ {env_name}: 异常 - {e}")

print(f"\n{'='*60}")
print("测试完成")
print(f"{'='*60}")
