#!python3


import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN

import pydsmc.property as prop
from pydsmc.evaluator import Evaluator
from pydsmc.utils import create_eval_envs

if __name__ == "__main__":
    NUM_THREADS = 1
    NUM_PAR_ENVS = 256
    SEED = 42
    TRUNCATE_LIMIT = 144

    # Evaluation environments can also be created manually but to make us of parallelism
    # the envs have to be a list of VectorEnv's with at least length num_threads.
    # To automate the creation, we use this utility function
    # Passing only a gym.Env to eval is also supported but will probably be slower
    envs = create_eval_envs(
        num_threads=NUM_THREADS,
        num_envs_per_thread=NUM_PAR_ENVS,
        env_seed=SEED,
        gym_id="MiniGrid-Dynamic-Obstacles-6x6-v0",
        wrappers=[FlatObsWrapper],
        vecenv_cls=gym.vector.AsyncVectorEnv,  # gym.vector.SyncVectorEnv, sb3.DummyVecEnv, sb3.SubprocVecEnv
        # The following kwargs are passed to the gym.make function, which passes unknown args to the env
        max_episode_steps=TRUNCATE_LIMIT
        # render_mode='human'
    )

    # create the agent
    agent = DQN.load("dqn_agent")

    # initialize the evaluator
    evaluator = Evaluator(env=envs, log_dir="./logs")

    # create and register a predefined property
    properties = []
    properties.append(prop.create_predefined_property(
                                        property_id='return',
                                        name='returnGamma0.99',
                                        eps=0.025,
                                        kappa=0.05,
                                        relative_error=True,
                                        bounds=(-1, 1),
                                        sound=True))
    properties.append(prop.create_predefined_property(
                                        property_id='return',
                                        name='returnUndiscounted',
                                        eps=0.025,
                                        kappa=0.05,
                                        relative_error=True,
                                        bounds=(-1, 1),
                                        sound=True,
                                        gamma=1.0))
    properties.append(prop.create_predefined_property(
                                        property_id='episode_length',
                                        eps=0.025,
                                        kappa=0.05,
                                        relative_error=True,
                                        bounds=(0, TRUNCATE_LIMIT+1)))
    properties.append(prop.create_predefined_property(
                                        property_id='goal_reaching_prob',
                                        eps=0.025,
                                        kappa=0.05,
                                        relative_error=False,
                                        goal_reward=1e-5))

    # define a custom property
    properties.append(prop.create_custom_property(
                                        name='obstacle_collision_prob',
                                        eps=0.025,
                                        kappa=0.05,
                                        relative_error=False,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: float(t[-1][2] == -1),
                                        binomial=True,
                                        sound=False
                                        ))

    evaluator.register_properties(properties)

    try:
        # evaluate the agent with respect to the registered properties
        results = evaluator.eval(agent=agent,
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
                                deterministic=True
                            )

    finally:
        for env in envs:
            env.close()
