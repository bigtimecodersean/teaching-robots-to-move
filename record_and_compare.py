"""
Record videos for all trained Humanoid agents and plot learning curves.
Run this after all four training scripts have finished.
 
Usage: python record_and_compare.py
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
 
 
# ============================================================
# Part 1: Record videos
# ============================================================
 
def record_naive(name, algo_class, model_path, n_episodes=3):
    """Record video for models trained WITHOUT VecNormalize."""
    video_dir = f"videos/{name}"
    os.makedirs(video_dir, exist_ok=True)
 
    env = RecordVideo(
        gym.make("Humanoid-v5", render_mode="rgb_array"),
        video_folder=video_dir,
        episode_trigger=lambda e: True,  # record every episode
        name_prefix=name,
    )
    model = algo_class.load(model_path)
 
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"  {name} episode {ep+1}: reward = {total_reward:.1f}")
                break
    env.close()
 
 
def record_engineered(name, algo_class, model_path, vecnorm_path, n_episodes=3):
    """Record video for models trained WITH VecNormalize."""
    video_dir = f"videos/{name}"
    os.makedirs(video_dir, exist_ok=True)
 
    # Need to reconstruct the VecNormalize wrapper for correct obs scaling
    def make_env():
        def _init():
            return gym.make("Humanoid-v5", render_mode="rgb_array")
        return _init
 
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
 
    model = algo_class.load(model_path)
 
    for ep in range(n_episodes):
        obs = vec_env.reset()
        total_reward = 0
        frames = []
 
        # Manually collect frames since RecordVideo doesn't work with VecEnv
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env.step(action)
            total_reward += reward[0]
 
            # Get frame from the underlying env
            frame = vec_env.envs[0].render()
            if frame is not None:
                frames.append(frame)
 
            if terminated[0] or truncated[0]:
                print(f"  {name} episode {ep+1}: reward = {total_reward:.1f}")
                break
 
        # Save as mp4
        if frames:
            import imageio
            output_path = os.path.join(video_dir, f"{name}_ep{ep+1}.mp4")
            imageio.mimsave(output_path, frames, fps=30)
            print(f"    Saved → {output_path}")
 
    vec_env.close()
 
 
print("Recording videos...")
print()
 
# Level 1: PPO naive
try:
    print("Level 1 — PPO (naive):")
    record_naive("level1_ppo_naive", PPO, "humanoid_ppo_naive")
except Exception as e:
    print(f"  Skipped: {e}")
 
# Level 2: SAC naive
try:
    print("Level 2 — SAC (naive):")
    record_naive("level2_sac_naive", SAC, "humanoid_sac_naive")
except Exception as e:
    print(f"  Skipped: {e}")
 
# Level 3: PPO engineered
try:
    print("Level 3 — PPO (engineered):")
    record_engineered("level3_ppo_eng", PPO,
                      "humanoid_ppo_eng", "humanoid_ppo_eng_vecnormalize.pkl")
except Exception as e:
    print(f"  Skipped: {e}")
 
# Level 4: SAC engineered
try:
    print("Level 4 — SAC (engineered):")
    record_engineered("level4_sac_eng", SAC,
                      "humanoid_sac_eng", "humanoid_sac_eng_vecnormalize.pkl")
except Exception as e:
    print(f"  Skipped: {e}")
 
 
# ============================================================
# Part 2: Learning curves
# ============================================================
 
print("\nPlotting learning curves...")
 
fig, ax = plt.subplots(figsize=(12, 7))
 
experiments = [
    ("Level 1: PPO (naive)",       "logs/humanoid_ppo_naive/evaluations.npz"),
    ("Level 2: SAC (naive)",       "logs/humanoid_sac_naive/evaluations.npz"),
    ("Level 3: PPO (engineered)",  "logs/humanoid_ppo_eng/evaluations.npz"),
    ("Level 4: SAC (engineered)",  "logs/humanoid_sac_eng/evaluations.npz"),
]
 
for name, path in experiments:
    try:
        data = np.load(path)
        steps = data["timesteps"]
        means = data["results"].mean(axis=1)
        stds  = data["results"].std(axis=1)
        ax.plot(steps, means, label=name, linewidth=2)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15)
    except FileNotFoundError:
        print(f"  Skipping {name} — no log at {path}")
 
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Mean Eval Reward", fontsize=12)
ax.set_title("Humanoid-v5: Algorithm × Engineering Comparison", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("humanoid_learning_curves.png", dpi=150)
print("Saved → humanoid_learning_curves.png")
plt.show()