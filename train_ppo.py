import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

train_env = Monitor(gym.make("Reacher-v5"))
eval_env  = Monitor(gym.make("Reacher-v5"))

model = PPO("MlpPolicy", train_env, verbose=1, seed=42)

eval_cb = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10,
                       best_model_save_path="./models/ppo_best/",
                       log_path="./logs/ppo/", deterministic=True)

model.learn(total_timesteps=500_000, callback=eval_cb)
model.save("ppo_reacher")
print("Done.")