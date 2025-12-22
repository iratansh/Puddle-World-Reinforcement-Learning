"""
Record demo GIFs of the trained PPO agent on all configurations.
Creates GIF files in the demos/ directory that can be embedded in README.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from ppo import (
    MultiConfigEnv,
    CONFIG_DIR,
    SEED,
    resolve_config_paths,
    resolve_model_path,
)


def record_episode_frames(model, env, max_steps=200):
    """Record frames from a single episode."""
    frames = []
    obs = env.reset()
    
    # Get the underlying environment for rendering
    inner_env = env.envs[0]._env
    
    # Capture initial frame
    frame = inner_env.render()
    if frame is not None:
        frames.append(frame)
    
    done = [False]
    step = 0
    total_reward = 0.0
    
    while not done[0] and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])
        
        # Capture frame
        frame = inner_env.render()
        if frame is not None:
            frames.append(frame)
        
        step += 1
    
    return frames, total_reward, step


def frames_to_gif(frames, output_path, duration=50):
    """Convert frames to a GIF file."""
    if not frames:
        print(f"No frames to save for {output_path}")
        return
    
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Record demo GIFs for all configs")
    parser.add_argument("--model-path", default="ppo_puddleWorld.zip")
    parser.add_argument("--norm-path", default="vec_normalize.pkl")
    parser.add_argument("--output-dir", default="demos")
    parser.add_argument("--duration", type=int, default=50, help="Frame duration in ms")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_path = resolve_model_path(args.model_path)
    norm_path = Path(args.norm_path)
    
    if not norm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found at {norm_path}")
    
    config_paths = resolve_config_paths([])
    print(f"Recording demos for {len(config_paths)} configurations...")
    print()
    
    results = []
    
    for idx, config_path in enumerate(config_paths):
        config_name = Path(config_path).stem  # e.g., "pw1"
        print(f"Recording {config_name}...")
        
        # Create environment with rgb_array render mode for recording
        def make_env():
            return MultiConfigEnv(
                config_paths,
                render_mode="rgb_array",
                fixed_config_idx=idx,
                obs_mode="both",
                shaping_coef=0.0,
            )
        
        env = DummyVecEnv([make_env])
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
        
        model = PPO.load(str(model_path), env=env)
        
        # Record episode
        frames, reward, steps = record_episode_frames(model, env)
        
        # Save GIF
        gif_path = output_dir / f"{config_name}.gif"
        frames_to_gif(frames, gif_path, duration=args.duration)
        
        results.append({
            'config': config_name,
            'reward': reward,
            'steps': steps,
            'gif_path': gif_path,
        })
        
        env.close()
        print(f"  Reward: {reward:.2f}, Steps: {steps}")
        print()
    
    # Print summary
    print("=" * 50)
    print("DEMO RECORDING COMPLETE")
    print("=" * 50)
    for r in results:
        print(f"{r['config']}: reward={r['reward']:.2f}, steps={r['steps']}, file={r['gif_path']}")
    
    print()
    print(f"GIFs saved to: {output_dir.absolute()}")
    print()
    print("Add to README with:")
    print("```markdown")
    for r in results:
        print(f"![{r['config']}](demos/{r['config']}.gif)")
    print("```")


if __name__ == "__main__":
    main()
