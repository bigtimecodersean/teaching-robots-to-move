import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class SparseReacher(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        fingertip_to_target = obs[8:10]
        dist = np.linalg.norm(fingertip_to_target)
        sparse_reward = 1.0 if dist < 0.02 else 0.0
        return obs, sparse_reward, terminated, truncated, info

train_env = Monitor(SparseReacher(gym.make("Reacher-v5")))
eval_env  = Monitor(SparseReacher(gym.make("Reacher-v5")))

model = PPO("MlpPolicy", train_env, verbose=1, seed=42)

eval_cb = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10,
                       best_model_save_path="./models/ppo_sparse_best/",
                       log_path="./logs/ppo_sparse/", deterministic=True)

model.learn(total_timesteps=500_000, callback=eval_cb)
model.save("ppo_sparse_reacher")
print("Done.")