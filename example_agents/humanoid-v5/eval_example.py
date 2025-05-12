#!python3

import pathlib
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import pydsmc.property as prop
from pydsmc.evaluator import Evaluator
from pydsmc.utils import create_eval_envs

if __name__ == "__main__":
    script_path = pathlib.Path(__file__).parent.resolve()

    NUM_THREADS = 1
    NUM_PAR_ENVS = 100
    SEED = 42
    TRUNCATE_LIMIT = 1000

    # Evaluation environments can also be created manually but to make us of parallelism
    # the envs have to be a list of VectorEnv's with at least length num_threads.
    # To automate the creation, we use this utility function
    # Passing only a gym.Env to eval is also supported but will probably be slower
    envs = create_eval_envs(
        num_threads=NUM_THREADS,
        num_envs_per_thread=NUM_PAR_ENVS,
        env_seed=SEED,
        gym_id="Humanoid-v5",
        wrappers=[Monitor],
        vecenv_cls=gym.vector.AsyncVectorEnv,  # gym.vector.SyncVectorEnv, sb3.DummyVecEnv, sb3.SubprocVecEnv
        # The following kwargs are passed to the gym.make function, which passes unknown args to the env
        max_episode_steps=TRUNCATE_LIMIT,
        # render_mode='human'
    )

    # create the agent
    agent = SAC.load(script_path / "sac_agent.zip")

    # initialize the evaluator
    evaluator = Evaluator(env=envs, log_dir=sys.argv[1] or script_path / "logs")

    # create and register a predefined property
    properties = []
    properties.append(
        prop.create_predefined_property(
            property_id="return",
            name="returnGamma0.99",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            bounds=(None, None),
            sound=False,
        ),
    )  # soundness not supported since not bounded. We could compute some bounds though
    properties.append(
        prop.create_predefined_property(
            property_id="return",
            name="returnUndiscounted",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            bounds=(None, None),
            sound=False,  # soundness not supported since not bounded. We could compute some bounds though
            gamma=1.0,
        ),
    )
    properties.append(
        prop.create_predefined_property(
            property_id="episode_length",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            sound=True,
            bounds=(0, TRUNCATE_LIMIT + 1),
        ),
    )

    # To demonstrate custom properties we define the goal_reaching probably as a custom property
    properties.append(
        prop.create_custom_property(
            name="ctrl_cost",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            check_fn=lambda self, t: (np.sum([sartti[5]["reward_ctrl"] for sartti in t]).item()),
        ),
    )
    properties.append(
        prop.create_custom_property(
            name="contact_cost",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            check_fn=lambda self, t: (np.sum([sartti[5]["reward_contact"] for sartti in t]).item()),
        ),
    )

    evaluator.register_properties(properties)

    try:
        # evaluate the agent with respect to the registered properties
        results = evaluator.eval(
            agent=agent,
            predict_fn=agent.predict,
            episode_limit=None,
            save_every_n_episodes=1000,
            num_initial_episodes=1000,
            num_episodes_per_policy_run=1000,
            save_full_results=False,
            stop_on_convergence=True,
            num_threads=NUM_THREADS,
            # Using non-deterministic policy might require setting a numpy seed for reproducibility
            # (this is for example the case with stable_baselines3 DQN) `np.random.seed(42)`
            deterministic=True,
        )

    finally:
        for env in envs:
            env.close()
