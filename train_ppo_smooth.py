import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class SmoothReacher(gym.Wrapper):
    def __init__(self, env, jerk_penalty=0.1):
        super().__init__(env)
        self.prev_action = np.zeros(env.action_space.shape)
        self.jerk_penalty = jerk_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        jerk = np.sum((action - self.prev_action) ** 2)
        reward -= self.jerk_penalty * jerk
        info["jerk"] = jerk
        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.env.action_space.shape)
        return self.env.reset(**kwargs)
    


# Wrap the env with the jerkiness penalty, then wrap with Monitor for logging
train_env = Monitor(SmoothReacher(gym.make("Reacher-v5")))
eval_env  = Monitor(SmoothReacher(gym.make("Reacher-v5")))

model = PPO("MlpPolicy", train_env, verbose=1, seed=42)

eval_cb = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10,
                       best_model_save_path="./models/ppo_smooth_best/",
                       log_path="./logs/ppo_smooth/", deterministic=True)

model.learn(total_timesteps=500_000, callback=eval_cb)
model.save("ppo_smooth_reacher")
print("Done.")