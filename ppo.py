"""
Train or render a PPO agent on multiple PuddleWorld configurations.
Supports observation augmentation with goal and puddle geometry for generalization.
Author: Ishaan Ratanshi
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gym_puddle
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.utils import seeding

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "gym-puddle" / "gym_puddle" / "env_configs"
DEFAULT_CONFIG_GLOB = "pw*.json"
MAX_EPISODE_STEPS = 500  # Longer horizon for complex pw5 navigation.
SEED = 100
GAMMA = 0.99
N_ENVS = 20  # More parallel envs to diversify experience across configs.
TOTAL_BATCH_SIZE = 10240  # Larger batches to improve stability.
N_STEPS = TOTAL_BATCH_SIZE // N_ENVS
BATCH_SIZE = 512
LEARNING_RATE = 3e-4  # Standard learning rate.
N_EPOCHS = 10
CLIP_RANGE = 0.2
ENT_COEF = 0.05  # Higher entropy to encourage exploration.
TARGET_KL = None  # Disable early stopping on KL.
CLIP_REWARD = 100.0
TOTAL_TIMESTEPS = 5_000_000  # Longer training for better generalization.
SHAPING_COEF = 1.0  # Stronger goal-seeking to counter puddle avoidance.


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.show()


def test_model(model, env, num_episodes):
    episode_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode finished with a total reward of {total_reward}")
        episode_rewards.append(total_reward)
    plot_rewards(episode_rewards)


def resolve_config_paths(config_args):
    if config_args:
        paths = [Path(p).expanduser() for p in config_args]
    else:
        paths = sorted(CONFIG_DIR.glob(DEFAULT_CONFIG_GLOB))
    if not paths:
        raise FileNotFoundError(f"No config files found in {CONFIG_DIR}.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Config files not found: {missing_list}")
    return paths


def load_config(config_path):
    with Path(config_path).open() as f:
        return json.load(f)


def make_single_env(config_path, render_mode=None):
    env_setup = load_config(config_path)
    env = gym.make(
        "PuddleWorld-v0",
        render_mode=render_mode,
        start=env_setup["start"],
        goal=env_setup["goal"],
        goal_threshold=env_setup["goal_threshold"],
        noise=env_setup["noise"],
        thrust=env_setup["thrust"],
        puddle_top_left=env_setup["puddle_top_left"],
        puddle_width=env_setup["puddle_width"],
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env


class MultiConfigEnv(gym.Env):
    """Sample a config on reset and optionally augment observations.

    Augments observations with goal and puddle geometry so the agent can
    generalize across layouts without memorizing config IDs.

    Observation depends on obs_mode:
    - geometry: [x, y, goal_x, goal_y, puddle1_x, puddle1_y, puddle1_w, puddle1_h, ...]
    - one-hot: [x, y, config_one_hot...]
    - both: geometry + one-hot
    """
    def __init__(
        self,
        config_paths,
        render_mode=None,
        fixed_config_idx=None,
        obs_mode="both",
        shaping_coef=0.0,
    ):
        super().__init__()
        self.config_paths = [Path(p) for p in config_paths]
        if not self.config_paths:
            raise ValueError("config_paths must include at least one config file.")
        if fixed_config_idx is not None:
            if fixed_config_idx < 0 or fixed_config_idx >= len(self.config_paths):
                raise ValueError("fixed_config_idx is out of range.")
        if obs_mode not in {"geometry", "one-hot", "both"}:
            raise ValueError("obs_mode must be one of: geometry, one-hot, both.")
        self.render_mode = render_mode
        self._np_random, _ = seeding.np_random(None)
        self.fixed_config_idx = fixed_config_idx
        self.obs_mode = obs_mode
        self.shaping_coef = shaping_coef
        self.current_config_idx = 0
        self.current_config_path = None
        self.current_config = None
        self._goal = None
        self._prev_dist = None
        self._configs = [load_config(p) for p in self.config_paths]
        self._max_puddles = max(
            len(config["puddle_top_left"]) for config in self._configs
        )
        self._env = make_single_env(self.config_paths[0], render_mode=self.render_mode)
        
        # Base spaces from the underlying env.
        self._base_obs_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._n_configs = len(self.config_paths)
        
        # Build the observation space based on obs_mode.
        extra_dims = 0
        if self.obs_mode in {"geometry", "both"}:
            # Goal (2) + puddle info (max_puddles * 4).
            extra_dims += 2 + self._max_puddles * 4
        if self.obs_mode in {"one-hot", "both"}:
            extra_dims += self._n_configs

        low = np.concatenate([self._base_obs_space.low, np.zeros(extra_dims)])
        high = np.concatenate([self._base_obs_space.high, np.ones(extra_dims)])
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32
        )
        self.metadata = getattr(self._env, "metadata", {})

    def _goal_distance(self, obs):
        return float(np.linalg.norm(obs - self._goal, ord=2))
    
    def _in_puddle(self, pos):
        """Return the maximum puddle depth if inside any puddle."""
        max_depth = 0.0
        for top_left, width in zip(
            self.current_config["puddle_top_left"],
            self.current_config["puddle_width"]
        ):
            # Compute puddle bounds.
            left, top = top_left[0], top_left[1]
            right, bottom = left + width[0], top - width[1]
            
            # Check whether the position is inside the puddle.
            if left <= pos[0] <= right and bottom <= pos[1] <= top:
                # Distance from puddle center (normalized).
                cx, cy = left + width[0]/2, top - width[1]/2
                dx = abs(pos[0] - cx) / (width[0]/2 + 1e-6)
                dy = abs(pos[1] - cy) / (width[1]/2 + 1e-6)
                depth = 1.0 - max(dx, dy)  # 1.0 at center, 0.0 at edge.
                max_depth = max(max_depth, depth)
        return max_depth

    def _augment_obs(self, obs):
        """Add configured context to the observation."""
        parts = [obs.astype(np.float32)]

        if self.obs_mode in {"geometry", "both"}:
            config = self.current_config
            # Goal position.
            goal = np.array(config["goal"], dtype=np.float32)

            # Puddle info: [x, y, w, h] for each puddle, padded to max count.
            puddle_info = np.zeros(self._max_puddles * 4, dtype=np.float32)
            for i, (top_left, width) in enumerate(
                zip(config["puddle_top_left"], config["puddle_width"])
            ):
                if i >= self._max_puddles:
                    break
                puddle_info[i * 4 : i * 4 + 4] = [
                    top_left[0], top_left[1], width[0], width[1]
                ]
            parts.extend([goal, puddle_info])

        if self.obs_mode in {"one-hot", "both"}:
            one_hot = np.zeros(self._n_configs, dtype=np.float32)
            one_hot[self.current_config_idx] = 1.0
            parts.append(one_hot)

        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        
        # Select a config (fixed if provided, otherwise random).
        if self.fixed_config_idx is None:
            self.current_config_idx = int(
                self._np_random.integers(len(self.config_paths))
            )
        else:
            self.current_config_idx = self.fixed_config_idx
        self.current_config_path = self.config_paths[self.current_config_idx]
        self.current_config = self._configs[self.current_config_idx]
        self._goal = np.array(self.current_config["goal"], dtype=np.float32)
        
        if self._env is not None:
            self._env.close()
        self._env = make_single_env(
            self.current_config_path, render_mode=self.render_mode
        )
        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_dist = self._goal_distance(obs)
        self._prev_obs = obs.copy()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self.shaping_coef:
            new_dist = self._goal_distance(obs)
            # Progress shaping: reward reductions in distance to the goal.
            # This guides the agent even when avoiding puddles.
            progress_bonus = self.shaping_coef * (self._prev_dist - new_dist)
            
            # No escape bonus; the environment's in-puddle penalty is sufficient.
            
            reward += progress_bonus
            self._prev_dist = new_dist
            self._prev_obs = obs.copy()
        return self._augment_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()


def make_env_fn(
    config_paths,
    render_mode=None,
    fixed_config_idx=None,
    obs_mode="both",
    shaping_coef=0.0,
):
    def _make():
        return Monitor(
            MultiConfigEnv(
                config_paths,
                render_mode=render_mode,
                fixed_config_idx=fixed_config_idx,
                obs_mode=obs_mode,
                shaping_coef=shaping_coef,
            )
        )

    return _make


def evaluate_configs(model, config_paths, norm_path, n_eval_episodes, obs_mode):
    for idx, config_path in enumerate(config_paths):
        eval_env = make_vec_env(
            make_env_fn(
                config_paths, fixed_config_idx=idx, obs_mode=obs_mode, shaping_coef=0.0
            ),
            n_envs=1,
            seed=SEED + 1,
        )
        eval_env = VecNormalize.load(str(norm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        print(f"{Path(config_path).name}: {mean_reward:.2f} +/- {std_reward:.2f}")
        eval_env.close()


def train(config_paths, eval_episodes, obs_mode, shaping_coef):
    # Create a vectorized environment with multiple parallel envs.
    vec_env = make_vec_env(
        make_env_fn(config_paths, obs_mode=obs_mode, shaping_coef=shaping_coef),
        n_envs=N_ENVS,
        seed=SEED,
    )
    
    # Normalize observations (and optionally rewards) for stability.
    # This mitigates differing reward scales (-1 per step vs. -400 * dist in puddles).
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=CLIP_REWARD,
        gamma=GAMMA,
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        # Larger network for multi-config learning.
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        ),
        gamma=GAMMA,
        gae_lambda=0.98,  # Higher GAE lambda for longer credit assignment.
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=CLIP_RANGE,
        target_kl=TARGET_KL,
        seed=SEED,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save the model and normalization statistics.
    model.save("ppo_puddleWorld")
    vec_env.save("vec_normalize.pkl")

    # Evaluation requires the same normalization statistics.
    if len(config_paths) > 1:
        evaluate_configs(
            model, config_paths, "vec_normalize.pkl", eval_episodes, obs_mode
        )
    else:
        eval_env = make_vec_env(
            make_env_fn(config_paths, obs_mode=obs_mode, shaping_coef=0.0),
            n_envs=1,
            seed=SEED + 1,
        )
        eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
        eval_env.training = False  # Do not update stats during eval.
        eval_env.norm_reward = False  # Use true rewards for eval.
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        eval_env.close()
    vec_env.close()


def resolve_model_path(model_path):
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    if model_path.suffix != ".zip":
        zipped = model_path.with_suffix(".zip")
        if zipped.exists():
            return zipped
    raise FileNotFoundError(
        f"Model not found at {model_path} (or {model_path.with_suffix('.zip')}). "
        "Train first to create it."
    )


def play(
    model_path,
    norm_path,
    config_paths,
    episodes,
    fixed_config_idx=None,
    obs_mode="both",
):
    model_path = resolve_model_path(model_path)
    norm_path = Path(norm_path)
    if not norm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at {norm_path}. Train first to create it."
        )

    eval_env = make_vec_env(
        make_env_fn(
            config_paths,
            render_mode="human",
            fixed_config_idx=fixed_config_idx,
            obs_mode=obs_mode,
            shaping_coef=0.0,
        ),
        n_envs=1,
        seed=SEED + 1,
    )
    eval_env = VecNormalize.load(str(norm_path), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(str(model_path), env=eval_env)
    obs = eval_env.reset()
    for episode_idx in range(episodes):
        done = [False]
        episode_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            eval_env.render()
            episode_reward += float(reward[0])
        print(f"Episode {episode_idx + 1} reward: {episode_reward:.2f}")
        obs = eval_env.reset()
    eval_env.close()


def eval_all_configs(model_path, norm_path, config_paths, episodes_per_config, obs_mode, render=False):
    """Evaluate the model on each config separately and report results."""
    model_path = resolve_model_path(model_path)
    norm_path = Path(norm_path)
    if not norm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at {norm_path}. Train first to create it."
        )

    print(f"\n{'='*60}")
    print(f"Evaluating model on {len(config_paths)} configurations")
    print(f"Episodes per config: {episodes_per_config}")
    print(f"{'='*60}\n")

    all_results = {}
    
    for idx, config_path in enumerate(config_paths):
        config_name = Path(config_path).name
        render_mode = "human" if render else None
        
        eval_env = make_vec_env(
            make_env_fn(
                config_paths,
                render_mode=render_mode,
                fixed_config_idx=idx,
                obs_mode=obs_mode,
                shaping_coef=0.0,
            ),
            n_envs=1,
            seed=SEED + idx,
        )
        eval_env = VecNormalize.load(str(norm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        model = PPO.load(str(model_path), env=eval_env)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(episodes_per_config):
            obs = eval_env.reset()
            done = [False]
            episode_reward = 0.0
            episode_length = 0
            
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                if render:
                    eval_env.render()
                episode_reward += float(reward[0])
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if render:
                print(f"  Episode {ep+1}: reward={episode_reward:.2f}, steps={episode_length}")
        
        eval_env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        all_results[config_name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'rewards': episode_rewards,
        }
        
        # Determine status.
        if mean_reward > -50:
            status = "PASS"
        elif mean_reward > -100:
            status = "OK"
        else:
            status = "FAIL"
        
        print(f"{config_name}: {mean_reward:7.2f} +/- {std_reward:5.2f} (avg {mean_length:.0f} steps) {status}")
    
    # Summary.
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_mean = np.mean([r['mean_reward'] for r in all_results.values()])
    passed = sum(1 for r in all_results.values() if r['mean_reward'] > -50)
    
    print(f"Average reward across all configs: {total_mean:.2f}")
    print(f"Configs with reward > -50: {passed}/{len(all_results)}")
    
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train or render a PPO agent.")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Render a trained policy instead of training.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the model on each config separately.",
    )
    parser.add_argument("--model-path", default="ppo_puddleWorld.zip")
    parser.add_argument("--norm-path", default="vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to env config JSON. Repeatable. Defaults to all pw*.json configs.",
    )
    parser.add_argument(
        "--fixed-config",
        help="Run a single config while keeping the full config list for observations.",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["one-hot", "geometry", "both"],
        default="both",
        help="Observation context for multi-config training.",
    )
    parser.add_argument(
        "--shaping-coef",
        type=float,
        default=SHAPING_COEF,
        help="Potential-based shaping coefficient (0 disables shaping).",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation (use with --eval).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_paths = resolve_config_paths(args.config)
    fixed_config_idx = None
    if args.fixed_config:
        fixed_path = Path(args.fixed_config).expanduser().resolve()
        resolved = [Path(p).resolve() for p in config_paths]
        if fixed_path not in resolved:
            raise ValueError(
                f"--fixed-config {fixed_path} is not in the config list."
            )
        fixed_config_idx = resolved.index(fixed_path)
    
    if args.eval:
        eval_all_configs(
            args.model_path,
            args.norm_path,
            config_paths,
            args.episodes,
            args.obs_mode,
            render=args.render,
        )
    elif args.play:
        play(
            args.model_path,
            args.norm_path,
            config_paths,
            args.episodes,
            fixed_config_idx=fixed_config_idx,
            obs_mode=args.obs_mode,
        )
    else:
        train(config_paths, args.eval_episodes, args.obs_mode, args.shaping_coef)




