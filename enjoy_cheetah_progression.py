import gymnasium as gym
import glob
import time
from stable_baselines3 import SAC

env = gym.make("HalfCheetah-v5", render_mode="human",
               default_camera_config={"type": 1, "trackbodyid": 1, "distance": 10})

checkpoints = sorted(glob.glob("models/cheetah_checkpoints/cheetah_*.zip"))
for path in checkpoints:
    steps = path.split("_")[-2]
    print(f"\n--- Checkpoint at {steps} steps ---")
    model = SAC.load(path)

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