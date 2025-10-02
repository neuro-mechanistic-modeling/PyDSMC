import shutil

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import pydsmc.property as prop
from pydsmc import Evaluator
from pydsmc.utils import create_eval_envs


@pytest.mark.parametrize("vec_env_cls", [SubprocVecEnv, AsyncVectorEnv, SyncVectorEnv, DummyVecEnv])
def test_vecenvs(vec_env_cls):
    LOG_DIR = "logs_test_vecenvs"
    shutil.rmtree(LOG_DIR, ignore_errors=True)

    envs = create_eval_envs(
        num_threads=1,
        num_envs_per_thread=1,
        env_seed=1,
        gym_id="MountainCarContinuous-v0",
        wrappers=[gym.wrappers.FlattenObservation],
        vecenv_cls=vec_env_cls,
        max_episode_steps=100,
    )

    agent = SAC.load("example_agents/mountaincar_continuous/sac_agent.zip")
    property_ = prop.create_custom_property(
        name="goal_reaching_prob_binomial_abs_sound",
        binomial=True,
        sound=True,
        epsilon=0.1,
        kappa=0.25,
        relative_error=False,
        bounds=(0, 1),
        check_fn=lambda self, t: np.sum(np.fromiter((s[2] for s in t), dtype=np.float32))
        >= self.goal_reward - 1e-8,
        goal_reward=1,
    )
    evaluator = Evaluator(env=envs, log_dir=LOG_DIR)
    evaluator.register_property(property_)

    evaluator.eval(
        agent=agent,
        episode_limit=2000,
        save_every_n_episodes=500,
        num_initial_episodes=200,
        num_episodes_per_policy_run=100,
        stop_on_convergence=False,
        save_full_results=False,
        deterministic=True,
    )

    shutil.rmtree(LOG_DIR, ignore_errors=True)
