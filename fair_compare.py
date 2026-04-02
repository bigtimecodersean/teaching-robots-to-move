import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC

env = gym.make("Reacher-v5")

models = [
    ("PPO Baseline",  PPO.load("ppo_reacher")),
    ("SAC",           SAC.load("sac_reacher")),
    ("PPO + Smooth",  PPO.load("ppo_smooth_reacher")),
]

for name, model in models:
    rewards = []
    for _ in range(50):
        obs, info = env.reset()
        total = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                rewards.append(total)
                break
    print(f"{name}: mean = {np.mean(rewards):.2f}, std = {np.std(rewards):.2f}")

env.close()