import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

train_env = Monitor(gym.make("HalfCheetah-v5"))
eval_env  = Monitor(gym.make("HalfCheetah-v5"))

model = SAC("MlpPolicy", train_env, verbose=1, seed=42)

eval_cb = EvalCallback(eval_env, eval_freq=10000, n_eval_episodes=10,
                       best_model_save_path="./models/cheetah_sac_best/",
                       log_path="./logs/cheetah_sac/", deterministic=True)

# Save a snapshot every 100k steps
checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path="./models/cheetah_checkpoints/",
                                   name_prefix="cheetah")

model.learn(total_timesteps=1_000_000, callback=[eval_cb, checkpoint_cb])
model.save("cheetah_sac")
print("Done.")