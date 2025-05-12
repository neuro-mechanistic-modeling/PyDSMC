#!python3

import pathlib

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from pydsmc import create_eval_envs

if __name__ == "__main__":
    script_path = pathlib.Path(__file__).parent.resolve()

    env = create_eval_envs(
        gym_id="HumanoidStandup-v5",
        num_envs_per_thread=16,
        env_seed=42,
        num_threads=1,
        wrappers=None,
        vecenv_cls=DummyVecEnv,
    )[0]

    agent = SAC("MlpPolicy", env, learning_starts=10_000, verbose=1)

    agent.learn(total_timesteps=2_000_000)
    agent.save(script_path / "sac_agent")
