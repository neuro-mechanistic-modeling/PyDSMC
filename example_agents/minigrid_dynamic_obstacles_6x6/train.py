#!python3


from collections import OrderedDict

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN

hyperparameters = OrderedDict(
    [
        ("batch_size", 512),
        ("learning_rate", 0.0001),
        ("learning_starts", 1000),
        ("train_freq", 10),
        ("policy", "MlpPolicy"),
    ],
)


env = gym.make("MiniGrid-Dynamic-Obstacles-Random-6x6-v0")
env = FlatObsWrapper(env)


agent = DQN(env=env, verbose=1, **hyperparameters)
agent.learn(total_timesteps=50000, log_interval=1000, progress_bar=True)

agent.save("dqn-minigrid-dynobs3.zip")
