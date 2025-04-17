#!python3

# Parameters taken from https://huggingface.co/sb3/sac-MountainCarContinuous-v0

from collections import OrderedDict

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pydsmc.utils import create_eval_envs

if __name__ == "__main__":
    # Train and save the model
    envs = create_eval_envs(
        num_threads=1,
        num_envs_per_thread=1,
        env_seed=100,
        gym_id="MountainCarContinuous-v0",
        wrappers=[Monitor],
        vecenv_cls=DummyVecEnv,  # gym.SyncVectorEnv, sb3.DummyVecEnv, sb3.SubprocVecEnv
        # The following kwargs are passed to the gym.make function, which passes unknown args to the env
    )

    params = OrderedDict(
        [
            ("env", envs[0]),
            ("batch_size", 512),
            ("buffer_size", 50000),
            ("ent_coef", 0.1),
            ("gamma", 0.9999),
            ("gradient_steps", 32),
            ("learning_rate", 0.0003),
            ("learning_starts", 0),
            ("policy", "MlpPolicy"),
            ("policy_kwargs", {"log_std_init": -3.67, "net_arch": [64, 64]}),
            ("tau", 0.01),
            ("train_freq", 32),
            ("use_sde", True),
        ],
    )
    model = SAC(verbose=1, **params)

    model.learn(total_timesteps=100000, progress_bar=True, log_interval=20)

    model.save("mountaincar")
