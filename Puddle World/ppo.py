import argparse
import json
from pathlib import Path

import gymnasium as gym
import gym_puddle
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.utils import seeding

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "gym-puddle" / "gym_puddle" / "env_configs"
DEFAULT_CONFIG_GLOB = "pw*.json"
MAX_EPISODE_STEPS = 500  # Increased to give agent more time to reach goal
SEED = 100
GAMMA = 0.99
N_ENVS = 8  # Parallel environments for better sample diversity
TOTAL_BATCH_SIZE = 4096
N_STEPS = TOTAL_BATCH_SIZE // N_ENVS
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
N_EPOCHS = 5
CLIP_RANGE = 0.1
ENT_COEF = 0.001
TARGET_KL = 0.01


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
    def __init__(self, config_paths, render_mode=None):
        super().__init__()
        self.config_paths = [Path(p) for p in config_paths]
        if not self.config_paths:
            raise ValueError("config_paths must include at least one config file.")
        self.render_mode = render_mode
        self._np_random, _ = seeding.np_random(None)
        self.current_config_path = None
        self._env = make_single_env(self.config_paths[0], render_mode=self.render_mode)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.metadata = getattr(self._env, "metadata", {})

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        self.current_config_path = self._np_random.choice(self.config_paths)
        if self._env is not None:
            self._env.close()
        self._env = make_single_env(
            self.current_config_path, render_mode=self.render_mode
        )
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()


def make_env_fn(config_paths, render_mode=None):
    def _make():
        return Monitor(MultiConfigEnv(config_paths, render_mode=render_mode))

    return _make


def evaluate_configs(model, config_paths, norm_path, n_eval_episodes):
    for config_path in config_paths:
        eval_env = make_vec_env(
            make_env_fn([config_path]),
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


def train(config_paths, eval_episodes):
    # Create vectorized environment with multiple parallel envs
    vec_env = make_vec_env(make_env_fn(config_paths), n_envs=N_ENVS, seed=SEED)
    
    # Normalize observations and rewards - CRITICAL for stable learning
    # This helps with the varying reward scales (-1 per step vs -400*dist in puddles)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        # Larger network with separate value/policy networks for better value estimation
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        gamma=GAMMA,
        gae_lambda=0.95,  # GAE for better advantage estimation
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,  # Entropy bonus to encourage exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        clip_range=CLIP_RANGE,
        target_kl=TARGET_KL,
        seed=SEED,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)  # More timesteps for convergence

    # Save model and normalization stats
    model.save("ppo_puddleWorld")
    vec_env.save("vec_normalize.pkl")

    # Evaluation - need to use the same normalization
    if len(config_paths) > 1:
        evaluate_configs(model, config_paths, "vec_normalize.pkl", eval_episodes)
    else:
        eval_env = make_vec_env(make_env_fn(config_paths), n_envs=1, seed=SEED + 1)
        eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
        eval_env.training = False  # Don't update stats during eval
        eval_env.norm_reward = False  # Use true rewards for eval
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


def play(model_path, norm_path, config_paths, episodes):
    model_path = resolve_model_path(model_path)
    norm_path = Path(norm_path)
    if not norm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at {norm_path}. Train first to create it."
        )

    eval_env = make_vec_env(
        make_env_fn(config_paths, render_mode="human"),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train or render a PPO agent.")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Render a trained policy instead of training.",
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
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_paths = resolve_config_paths(args.config)
    if args.play:
        play(args.model_path, args.norm_path, config_paths, args.episodes)
    else:
        train(config_paths, args.eval_episodes)
