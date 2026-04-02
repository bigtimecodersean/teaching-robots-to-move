
import gymnasium as gym
import glob
import time
from stable_baselines3 import PPO

checkpoints = sorted(glob.glob("models/ant_checkpoints/ant_*.zip"),
                     key=lambda x: int(x.split("_")[-2]))

for path in checkpoints:
    steps = path.split("_")[-2]
    input(f"\n--- Press Enter to play checkpoint at {steps} steps ---")

    env = gym.make("Ant-v5", render_mode="human",
                    default_camera_config={"type": 1, "trackbodyid": 1, "distance": 8})
    model = PPO.load(path)

    obs, info = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        time.sleep(0.03)
        if terminated or truncated:
            print(f"Reward: {total_reward:.1f}")
            break

    env.close()
    time.sleep(0.5)