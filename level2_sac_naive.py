"""
Level 2: Naive SAC on Humanoid-v5
Off-policy with replay buffer and automatic entropy tuning.
Default network and hyperparameters — same "just plug it in" approach as Level 1.
"""
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
 
train_env = Monitor(gym.make("Humanoid-v5"))
eval_env  = Monitor(gym.make("Humanoid-v5"))
 
model = SAC("MlpPolicy", train_env, verbose=1, seed=42)
 
eval_cb = EvalCallback(eval_env, eval_freq=20000, n_eval_episodes=10,
                       best_model_save_path="./models/humanoid_sac_naive_best/",
                       log_path="./logs/humanoid_sac_naive/", deterministic=True)
 
checkpoint_cb = CheckpointCallback(save_freq=500_000,
                                   save_path="./models/humanoid_sac_naive_ckpts/",
                                   name_prefix="humanoid_sac_naive")
 
model.learn(total_timesteps=2_000_000, callback=[eval_cb, checkpoint_cb])
model.save("humanoid_sac_naive")
print("Done — Level 2 (SAC naive)")
 