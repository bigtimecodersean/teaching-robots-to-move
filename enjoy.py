import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC

model = PPO.load("ppo_sparse_reacher")
env = gym.make("Reacher-v5", render_mode="human")

for episode in range(50):
    obs, info = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode {episode+1}: reward = {total_reward:.1f}")
            break

env.close()