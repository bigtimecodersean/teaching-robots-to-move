"""
Level 1: Naive PPO on Humanoid-v5
Default everything — small network (64x64), no normalization, single env.
This is the bare minimum baseline.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
 
train_env = Monitor(gym.make("Humanoid-v5"))
eval_env  = Monitor(gym.make("Humanoid-v5"))
 
model = PPO("MlpPolicy", train_env, verbose=1, seed=42)
 
eval_cb = EvalCallback(eval_env, eval_freq=20000, n_eval_episodes=10,
                       best_model_save_path="./models/humanoid_ppo_naive_best/",
                       log_path="./logs/humanoid_ppo_naive/", deterministic=True)
 
checkpoint_cb = CheckpointCallback(save_freq=500_000,
                                   save_path="./models/humanoid_ppo_naive_ckpts/",
                                   name_prefix="humanoid_ppo_naive")
 
model.learn(total_timesteps=2_000_000, callback=[eval_cb, checkpoint_cb])
model.save("humanoid_ppo_naive")
print("Done — Level 1 (PPO naive)")
 