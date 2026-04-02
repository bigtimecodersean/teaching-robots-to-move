import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

train_env = Monitor(gym.make("Ant-v5"))
eval_env  = Monitor(gym.make("Ant-v5"))

model = PPO("MlpPolicy", train_env, verbose=1, seed=42)

eval_cb = EvalCallback(eval_env, eval_freq=10000, n_eval_episodes=10,
                       best_model_save_path="./models/ant_ppo_best/",
                       log_path="./logs/ant_ppo/", deterministic=True)

checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path="./models/ant_checkpoints/",
                                   name_prefix="ant")

model.learn(total_timesteps=500_000, callback=[eval_cb, checkpoint_cb])
model.save("ant_ppo")
print("Done.")