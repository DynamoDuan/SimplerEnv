import os
import argparse
import numpy as np
import gymnasium as gym
import mani_skill2_real2sim.envs
import imageio
from pathlib import Path
import re
import math
import json
import simpler_env  # Registers environments

def extract_depth_from_position(position_texture):
    """
    Extract depth from Position texture.
    Position texture contains (x, y, z, w) in camera space.
    Depth is the negative of the Z component.
    """
    return -position_texture[..., 2]

def save_depth_uint16(depth, filepath, depth_scale=0.001):
    """Save depth map as uint16 PNG."""
    depth_uint16 = (depth / depth_scale).astype(np.uint16)
    if depth_uint16.ndim == 3 and depth_uint16.shape[-1] == 1:
        depth_uint16 = depth_uint16.squeeze(-1)
    imageio.imwrite(filepath, depth_uint16)
    print(f"Saved depth to {filepath} with scale {depth_scale}")

def to_uint8(img):
    if img is None: return None
    img = np.asarray(img)
    if img.ndim < 3: return None
    if img.shape[-1] >= 3: img = img[..., :3]
    if img.dtype == np.uint8: return img
    mx = float(img.max()) if img.size else 0.0
    if mx <= 1.5: img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def quat_from_yaw_pitch(yaw, pitch):
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    w = cy*cp
    x = cy*sp
    y = sy*sp
    z = sy*cp
    return np.array([w, x, y, z], dtype=np.float32)

def set_drawer_top_camera_pose(env, cam_name, cam_z=0.75, cam_forward=0.20):
    """Repositions camera for drawers/cabinets to get a better 'first frame' view."""
    u = env.unwrapped
    if not (hasattr(u, "_cameras") and cam_name in u._cameras):
        return False
        
    cam = u._cameras[cam_name]
    target = np.zeros(3, dtype=np.float32)
    
    # Try to find a target object (drawer/cabinet)
    found_target = False
    for name in ("cabinet", "_cabinet", "drawer", "_drawer", "obj", "_obj"):
        if hasattr(u, name):
            o = getattr(u, name)
            if hasattr(o, "pose"):
                target = o.pose.p
                found_target = True
                break
    
    if not found_target:
        # Fallback to actor list search
        if hasattr(u, "scene"):
             for a in u.scene.get_all_actors():
                n = (a.name or "").lower()
                if "cabinet" in n or "drawer" in n:
                    target = a.get_pose().p
                    found_target = True
                    break

    cam_yaw = 0.0
    cam_pitch = math.radians(85)
    
    forward = np.array([math.sin(cam_yaw), -math.cos(cam_yaw), 0.0], dtype=np.float32)
    pos = target + cam_z * np.array([0.0, 0.0, 1.0], dtype=np.float32) + cam_forward * forward
    q = quat_from_yaw_pitch(cam_yaw, cam_pitch)
    
    try:
        cam.set_pose(pos, q)
        return True
    except:
        import sapien
        cam.set_pose(sapien.Pose(pos, q))
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=".*", help="Regex pattern for env IDs")
    parser.add_argument("--out_dir", type=str, default="first_frames")
    parser.add_argument("--camera", type=str, default="overhead_camera")
    parser.add_argument("--depth_scale", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(list(gym.registry.keys()))
    rx = re.compile(args.pattern)
    env_ids = [eid for eid in all_ids if rx.search(eid)]
    
    print(f"Found {len(env_ids)} environments matching pattern '{args.pattern}'")

    for env_id in env_ids:
        print(f"\nProcessing {env_id}...")
        try:
            # We enforce obs_mode='rgbd' to get textures
            env = gym.make(
                env_id, 
                obs_mode="rgbd",
                control_mode="arm_pd_ee_delta_pose_gripper_pd_joint_pos"
            )
            
            # Set resolution if possible
            u = env.unwrapped
            if hasattr(u, "_camera_cfgs") and args.camera in u._camera_cfgs:
                cfg = u._camera_cfgs[args.camera]
                cfg.width = args.width
                cfg.height = args.height

            obs, _ = env.reset()
            
            # Reposition camera for drawers if needed (mimicking shell script logic)
            if "drawer" in env_id.lower() or "cabinet" in env_id.lower():
                set_drawer_top_camera_pose(env, args.camera)
                obs = env.get_obs()

            # Metadata info
            metadata = {
                "env_id": env_id,
                "camera_name": args.camera,
                "depth_scale": args.depth_scale
            }

            # Extract camera info
            if hasattr(env.unwrapped, '_cameras') and args.camera in env.unwrapped._cameras:
                cam = env.unwrapped._cameras[args.camera]
                # Use get_params() which returns dict with matrices
                params = cam.get_params()
                
                # Save YAML params as requested
                # fx: 425.0
                # fy: 425.0
                # cx: 320.0
                # cy: 256.0
                # depth_scale: 0.001
                # depth_max: 1.6
                
                # Ensure we have intrinsic from valid source
                fx = params['intrinsic_cv'][0, 0]
                fy = params['intrinsic_cv'][1, 1]
                cx = params['intrinsic_cv'][0, 2]
                cy = params['intrinsic_cv'][1, 2]
                
                yaml_content = (
                    f"fx: {fx}\n"
                    f"fy: {fy}\n"
                    f"cx: {cx}\n"
                    f"cy: {cy}\n"
                    f"depth_scale: {args.depth_scale}\n"
                    f"depth_max: 1.6 # this is to clip the real depth for removing outliers\n"
                )
                yaml_path = out_dir / f"{env_id}_params.yaml"
                with open(yaml_path, "w") as f:
                    f.write(yaml_content)
                print(f"Saved params to {yaml_path}")

            # Extract images
            if "image" in obs and args.camera in obs["image"]:
                cam_obs = obs["image"][args.camera]
                
                # RGB
                if "rgb" in cam_obs:
                    rgb = to_uint8(cam_obs["rgb"])
                    rgb_path = out_dir / f"{env_id}_rgb.png"
                    imageio.imwrite(rgb_path, rgb)
                    print(f"Saved RGB to {rgb_path}")
                
                # Depth saving removed as requested
            else:
                print(f"Camera {args.camera} not found in observation")

            env.close()

        except Exception as e:
            print(f"Failed {env_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

