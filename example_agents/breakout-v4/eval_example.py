#!python3


import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

import pydsmc.property as prop
from pydsmc.evaluator import Evaluator
from pydsmc.utils import create_eval_envs


if __name__ == "__main__":
    NUM_THREADS = 1 #! BREAKS FOR HIGHER VALUE's. I guess the emulator has some kind of global state?
    NUM_PAR_ENVS = 50
    SEED = 42

    # Necessary because of legacy reasons I suppose
    class RemoveLastDimensionWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            obs_shape = self.observation_space.shape[:-1]  # Remove the last dimension
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low[..., 0],
                high=self.observation_space.high[..., 0],
                shape=obs_shape,
                dtype=self.observation_space.dtype,
            )

        def observation(self, observation):
            return observation[..., 0]  # Remove the last dimension

    # Evaluation environments can also be created manually but to make us of parallelism
    # the envs have to be a list of VectorEnv's with at least length num_threads.
    # To automate the creation, we use this utility function
    # Passing only a gym.Env to eval is also supported but will probably be slower
    envs = create_eval_envs(
        num_threads=NUM_THREADS,
        num_envs_per_thread=NUM_PAR_ENVS,
        env_seed=SEED,
        gym_id="BreakoutNoFrameskip-v4",
        wrappers=[lambda e: AtariWrapper(e, frame_skip=1, terminal_on_life_loss=True), lambda e: FrameStackObservation(e, stack_size=4), RemoveLastDimensionWrapper, Monitor],
        vecenv_cls=gym.vector.AsyncVectorEnv, # gym.vector.SyncVectorEnv, sb3.DummyVecEnv, sb3.SubprocVecEnv
        # The following kwargs are passed to the gym.make function, which passes unknown args to the env
        # max_episode_steps=TRUNCATE_LIMIT
        # render_mode='human'
    )

    # create the agent
    agent = PPO.load("ppo_agent")

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
        bounds=(0, 864),
        sound=True
    ))
    properties.append(prop.create_predefined_property(
        property_id='return',
        name='returnUndiscounted',
        eps=0.025,
        kappa=0.05,
        relative_error=True,
        bounds=(0, 864),
        sound=True,
        gamma=1.0
    ))
    properties.append(prop.create_predefined_property(
        property_id='episode_length',
        eps=0.025,
        kappa=0.05,
        relative_error=True,
        bounds=(0, None)
    ))
    # define a custom property
    properties.append(prop.create_custom_property(
                                        name='first_life_lost',
                                        eps=0.025,
                                        kappa=0.05,
                                        relative_error=True,
                                        bounds=(0, None),
                                        check_fn=lambda self, t: next((i for i, x in enumerate(t) if 'lives' in x[5] and x[5]['lives'] < 5), len(t))))


    evaluator.register_properties(property)

    try:
        # evaluate the agent with respect to the registered properties
        results = evaluator.eval(agent=agent,
                                predict_fn=agent.predict,
                                episode_limit=None,
                                save_every_n_episodes=100,
                                num_initial_episodes=100,
                                num_episodes_per_policy_run=100,
                                save_full_results=False,
                                seed=45,
                                stop_on_convergence=True,
                                num_threads=NUM_THREADS,
                                # Using non-deterministic policy might require setting a numpy seed for reproducibility
                                # (this is for example the case with stable_baselines3 DQN) `np.random.seed(42)`
                                deterministic=True
                            )

    finally:
        for env in envs:
            env.close()
