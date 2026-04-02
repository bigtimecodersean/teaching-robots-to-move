"""


Level 3: Engineered PPO on Humanoid-v5

Three key improvements over Level 1:
  1. SubprocVecEnv — 8 parallel envs for 8x data throughput
  2. VecNormalize — normalizes observations and rewards on the fly
     (Humanoid obs span wildly different scales; this alone can double performance)
  3. Larger network (256x256) and tuned hyperparameters from rl-baselines3-zoo

This tests whether ENGINEERING beats ALGORITHM CHOICE.
 
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env():
    def _init():
        return Monitor(gym.make("Humanoid-v5"))
    return _init


if __name__ == '__main__':
    train_vec_env = SubprocVecEnv([make_env() for _ in range(8)])
    train_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec_env = SubprocVecEnv([make_env() for _ in range(4)])
    eval_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    class SyncNormCallback(BaseCallback):
        def _on_step(self):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            return True

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=42,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_cb = EvalCallback(eval_env, eval_freq=20000 // 8, n_eval_episodes=10,
                           best_model_save_path="./models/humanoid_ppo_eng_best/",
                           log_path="./logs/humanoid_ppo_eng/", deterministic=True,
                           callback_after_eval=SyncNormCallback())

    checkpoint_cb = CheckpointCallback(save_freq=500_000 // 8,
                                       save_path="./models/humanoid_ppo_eng_ckpts/",
                                       name_prefix="humanoid_ppo_eng")

    model.learn(total_timesteps=2_000_000, callback=[eval_cb, checkpoint_cb])

    model.save("humanoid_ppo_eng")
    train_env.save("humanoid_ppo_eng_vecnormalize.pkl")
    print("Done — Level 3 (PPO engineered)")