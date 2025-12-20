# Puddle World (RL)

Train and evaluate a PPO agent on the `gym_puddle` Puddle World environment (Gymnasium).

## What’s in this repo

- `ppo.py`: PPO training + evaluation + interactive “play” mode (renders the learned policy)
- `gym-puddle/`: local Gymnasium environment package (`gym_puddle`) with multiple environment configs
- `ppo_puddleWorld.zip`: saved PPO policy (created by training)
- `vec_normalize.pkl`: saved `VecNormalize` stats (created by training; required for correct eval/render)

## Environment configs

There are multiple Puddle World configurations in `gym-puddle/gym_puddle/env_configs/`:

- `pw1.json`, `pw2.json`, `pw3.json`, `pw4.json`, `pw5.json`

`ppo.py` currently uses `pw3.json` via:

- `JSON_FILE = ROOT / "gym-puddle" / "gym_puddle" / "env_configs" / "pw3.json"`

## Setup (conda)

```bash
conda create -n puddle-world python=3.10 -y
conda activate puddle-world

# core deps used by `ppo.py`
pip install gymnasium==0.29.1 numpy==1.26.4 pygame==2.5.2 stable-baselines3 matplotlib

# install the local environment package
pip install -e "./gym-puddle"
```

## Train PPO

```bash
conda activate puddle-world
python ppo.py
```

This produces:
- `ppo_puddleWorld.zip` (trained model)
- `vec_normalize.pkl` (normalization statistics; required for correct evaluation/rendering)

### PPO hyperparameters (current)

From `ppo.py`:
- `N_ENVS=8`, `TOTAL_BATCH_SIZE=4096`, `N_STEPS=512`
- `LEARNING_RATE=1e-4`, `N_EPOCHS=5`, `BATCH_SIZE=256`
- `CLIP_RANGE=0.1`, `ENT_COEF=0.001`, `TARGET_KL=0.01`
- `GAMMA=0.99`, `gae_lambda=0.95`
- `VecNormalize` enabled for obs + reward normalization

## Play / Render PPO

```bash
conda activate puddle-world
python ppo.py --play
```

Optional args:

```bash
python ppo.py --play --model-path ppo_puddleWorld.zip --norm-path vec_normalize.pkl --episodes 5
```

## Results

### Training (latest)

Final training snapshot (SB3 logs):
- `total_timesteps`: ~806,912+
- `ep_rew_mean`: ~-25.8
- `ep_len_mean`: ~26.8
- `explained_variance`: ~0.99

Interpretation: the agent reliably reaches the goal in ~27 steps; since the environment gives `-1` per step (outside puddles) and `0` at the goal, a return near `-27` indicates an efficient path that largely avoids puddles.

### Evaluation (play mode)

Per-episode returns (5 episodes):
- Episode 1 reward: -24.00
- Episode 2 reward: -25.00
- Episode 3 reward: -25.00
- Episode 4 reward: -25.00
- Episode 5 reward: -25.00

## Notes

- If VS Code shows `Import "gym_puddle" could not be resolved`, select the `puddle-world` interpreter (it’s usually just an editor environment selection issue).

## Credits

This project vendors/uses the Puddle World Gymnasium environment from the original `gym-puddle` implementation by EhsanEI:

- https://github.com/EhsanEI/gym-puddle
