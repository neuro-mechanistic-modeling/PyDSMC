#!python3


import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN

if __name__ == "__main__":
    env = gym.make("MiniGrid-Dynamic-Obstacles-Random-6x6-v0")
    env = FlatObsWrapper(env)

    agent = DQN.load("example_agents/minigrid_dynamic_obstacles_6x6/dqn_agent.zip")

    import pydsmc.property as prop
    from pydsmc.evaluator import Evaluator

    # initialize the evaluator
    evaluator = Evaluator(env=env, log_dir="./example_logs")

    # create and register a predefined property
    return_property = prop.create_predefined_property(
        property_id="return",
        epsilon=0.025,
        kappa=0.05,
        relative_error=True,
        bounds=(-1, 1),
        sound=True,
    )
    evaluator.register_property(return_property)

    # Custom properties are also possible
    collision_property = prop.create_custom_property(
        name="obstacle_collision_prob",
        check_fn=lambda self, t: float(t[-1][2] == -1),
        epsilon=0.05,
        kappa=0.05,
        relative_error=True,
        bounds=(0, 1),
        binomial=True,
    )
    evaluator.register_property(collision_property)

    # evaluate the agent with respect to the registered properties
    results = evaluator.eval(
        agent=agent,
        save_every_n_episodes=1000,
        time_limit=2.5,
        stop_on_convergence=True,
    )
