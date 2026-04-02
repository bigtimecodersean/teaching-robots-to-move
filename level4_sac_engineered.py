"""
Level 4: Engineered SAC on Humanoid-v5
Same engineering improvements as Level 3 but applied to SAC:
  1. VecNormalize for observation normalization
  2. Larger network (256x256)
  3. Tuned hyperparameters
 
SAC is single-env (off-policy doesn't benefit as much from parallel envs
since it reuses experience via replay buffer), but normalization and
network size still help enormously.
"""


"""
Level 4: Engineered SAC on Humanoid-v5
"""
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env():
    def _init():
        return Monitor(gym.make("Humanoid-v5"))
    return _init


if __name__ == '__main__':
    train_vec_env = DummyVecEnv([make_env()])
    train_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    class SyncNormCallback(BaseCallback):
        def _on_step(self):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            return True

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=42,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_cb = EvalCallback(eval_env, eval_freq=20000, n_eval_episodes=10,
                           best_model_save_path="./models/humanoid_sac_eng_best/",
                           log_path="./logs/humanoid_sac_eng/", deterministic=True,
                           callback_after_eval=SyncNormCallback())

    checkpoint_cb = CheckpointCallback(save_freq=500_000,
                                       save_path="./models/humanoid_sac_eng_ckpts/",
                                       name_prefix="humanoid_sac_eng")

    model.learn(total_timesteps=2_000_000, callback=[eval_cb, checkpoint_cb])

    model.save("humanoid_sac_eng")
    train_env.save("humanoid_sac_eng_vecnormalize.pkl")
    print("Done — Level 4 (SAC engineered)")

    