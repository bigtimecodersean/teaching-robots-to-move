"""
Watch Humanoid checkpoints in order to see learning progression.

Usage:
  python watch_humanoid.py 1   (Level 1: PPO naive)
  python watch_humanoid.py 2   (Level 2: SAC naive)
  python watch_humanoid.py 3   (Level 3: PPO engineered)
  python watch_humanoid.py 4   (Level 4: SAC engineered)
"""
import gymnasium as gym
import glob
import time
import sys
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

if len(sys.argv) < 2 or sys.argv[1] not in ("1", "2", "3", "4"):
    print("Usage: python watch_humanoid.py [1|2|3|4]")
    sys.exit(1)

level = sys.argv[1]

configs = {
    "1": ("PPO (naive)", PPO,
          "models/humanoid_ppo_naive_ckpts/humanoid_ppo_naive_*.zip", None),
    "2": ("SAC (naive)", SAC,
          "models/humanoid_sac_naive_ckpts/humanoid_sac_naive_*.zip", None),
    "3": ("PPO (engineered)", PPO,
          "models/humanoid_ppo_eng_ckpts/humanoid_ppo_eng_*.zip",
          "humanoid_ppo_eng_vecnormalize.pkl"),
    "4": ("SAC (engineered)", SAC,
          "models/humanoid_sac_eng_ckpts/humanoid_sac_eng_*.zip",
          "humanoid_sac_eng_vecnormalize.pkl"),
}

name, AlgoClass, pattern, vecnorm_path = configs[level]
checkpoints = sorted(glob.glob(pattern), key=lambda x: int(x.split("_")[-2]))

if not checkpoints:
    print(f"No checkpoints found at {pattern}")
    sys.exit(1)

print(f"Watching: {name}")
print(f"Found {len(checkpoints)} checkpoints\n")

for path in checkpoints:
    steps = path.split("_")[-2]
    input(f"--- Press Enter to play {name} at {steps} steps ---")

    if vecnorm_path:
        def make_env():
            def _init():
                return gym.make("Humanoid-v5", render_mode="human")
            return _init

        vec_env = DummyVecEnv([make_env()])
        try:
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
        except FileNotFoundError:
            print(f"  Warning: {vecnorm_path} not found, using raw observations")
        vec_env.training = False
        vec_env.norm_reward = False

        model = AlgoClass.load(path)
        obs = vec_env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env.step(action)
            total_reward += reward[0]
            time.sleep(0.02)
            if terminated[0] or truncated[0]:
                print(f"Reward: {total_reward:.1f}")
                break
        vec_env.close()
    else:
        env = gym.make("Humanoid-v5", render_mode="human")
        model = AlgoClass.load(path)
        obs, info = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            time.sleep(0.02)
            if terminated or truncated:
                print(f"Reward: {total_reward:.1f}")
                break
        env.close()

    time.sleep(0.5)

print("\nDone — all checkpoints played.")