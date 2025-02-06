#!python3

import pgtg
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit

env = gym.make('pgtg-v3', final_goal_bonus=150)
env = TimeLimit(env, max_episode_steps=100)
env = FlattenObservation(env)

agent = DQN("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=10000, log_interval=1000)

agent.save("dqn_pgtg-v3.zip")
