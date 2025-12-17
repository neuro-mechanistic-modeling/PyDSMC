#!/usr/bin/env python3

import pathlib
import sys

import gymnasium as gym
import torch
from minigrid.wrappers import ImgObsWrapper
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import pydsmc.property as prop
from pydsmc import Evaluator
from pydsmc.utils import create_eval_envs


class MinigridFeatExtract(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):
        super().__init__(observation_space, features_dim)

        # fmt: off
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        # fmt: on

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = observations.float()

        # if observations are HWC, convert to CHW
        if x.ndim == 4 and x.shape[-1] in (3, 4):
            x = x.permute(0, 3, 1, 2)

        B = x.shape[0]

        x = self.feat_extract(x)
        x = x.reshape(B, -1)
        return self.fc(x)


class MinigridLSTMPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            lstm_hidden_size=1024,
            n_lstm_layers=2,
            net_arch=[],
            ortho_init=True,
            features_extractor_class=MinigridFeatExtract,
            features_extractor_kwargs={},
            shared_lstm=True,
            enable_critic_lstm=False,
            **kwargs,
        )


if __name__ == "__main__":
    script_path = pathlib.Path(__file__).parent.resolve()

    NUM_THREADS = 1
    NUM_PAR_ENVS = 16
    SEED = 42
    TRUNCATE_LIMIT = 240

    # Evaluation environments can also be created manually but to make us of parallelism
    # the envs have to be a list of VectorEnv's with at least length num_threads.
    # To automate the creation, we use this utility function
    # Passing only a gym.Env to eval is also supported but will probably be slower
    gym_id = "MiniGrid-MultiRoom-N6-v0"
    envs = create_eval_envs(
        num_threads=NUM_THREADS,
        num_envs_per_thread=NUM_PAR_ENVS,
        env_seed=SEED,
        gym_id=gym_id,
        wrappers=[ImgObsWrapper],
        vecenv_cls=gym.vector.AsyncVectorEnv,  # gym.vector.SyncVectorEnv, sb3.DummyVecEnv, sb3.SubprocVecEnv
        # The following kwargs are passed to the gym.make function, which passes unknown args to the env
        max_episode_steps=TRUNCATE_LIMIT,
        max_steps=TRUNCATE_LIMIT,  # minigrid takes this as an argument
        agent_view_size=7,
        # render_mode='human'
    )

    # create the agent
    tenv = gym.make(  # to register the env and avoid errors when loading the agent
        gym_id,
        max_episode_steps=TRUNCATE_LIMIT,
        max_steps=TRUNCATE_LIMIT,
        agent_view_size=7,
    )
    tenv = ImgObsWrapper(tenv)

    agent = RecurrentPPO(policy=MinigridLSTMPolicy, env=tenv)
    state_dict = torch.load("recppo.pth", map_location=torch.device("cpu"))
    agent.policy.load_state_dict(state_dict)

    # initialize the evaluator
    log_dir = sys.argv[1] if len(sys.argv) > 1 else script_path / "logs"
    evaluator = Evaluator(env=envs, log_dir=log_dir)

    # create and register a predefined property
    properties = []
    properties.append(
        prop.create_predefined_property(
            property_id="return",
            name="returnGamma0.99",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            bounds=(-1, 1),
            sound=True,
        ),
    )
    properties.append(
        prop.create_predefined_property(
            property_id="return",
            name="returnUndiscounted",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            bounds=(-1, 1),
            sound=True,
            gamma=1.0,
        ),
    )
    properties.append(
        prop.create_predefined_property(
            property_id="episode_length",
            epsilon=0.025,
            kappa=0.05,
            relative_error=True,
            bounds=(0, TRUNCATE_LIMIT + 1),
        ),
    )
    properties.append(
        prop.create_predefined_property(
            property_id="goal_reaching_probability",
            epsilon=0.025,
            kappa=0.05,
            relative_error=False,
            goal_reward=1e-5,
        ),
    )

    evaluator.register_properties(properties)

    try:
        # evaluate the agent with respect to the registered properties
        results = evaluator.eval(
            agent=agent,
            episode_limit=None,
            save_every_n_episodes=128,
            num_initial_episodes=256,
            num_episodes_per_policy_run=128,
            save_full_results=False,
            stop_on_convergence=True,
            num_threads=NUM_THREADS,
            # Using non-deterministic policy might require setting a numpy seed for reproducibility
            # (this is for example the case with stable_baselines3 DQN) `np.random.seed(42)`
            deterministic=True,  # Fully deterministic minigrid environment, but procedurally gen'd
        )

    finally:
        for env in envs:
            env.close()
