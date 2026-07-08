# Examples

This page walks through a complete evaluation on a trained agent. For a version that runs entirely
in your browser with no installation, see [Try in Browser](try-in-browser.md).

## A full example (MiniGrid + Stable-Baselines3)

The snippet below mirrors [`simple_example.py`](https://github.com/neuro-mechanistic-modeling/PyDSMC/blob/main/simple_example.py)
in the repository. It loads a trained DQN agent on a MiniGrid environment and evaluates both a
predefined *return* property and a custom *collision-probability* property.

```python
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN

import pydsmc.property as prop
from pydsmc.evaluator import Evaluator

env = gym.make("MiniGrid-Dynamic-Obstacles-Random-6x6-v0")
env = FlatObsWrapper(env)

agent = DQN.load("example_agents/minigrid_dynamic_obstacles_6x6/dqn_agent.zip")

# Initialize the evaluator.
evaluator = Evaluator(env=env, log_dir="./example_logs")

# Create and register a predefined property.
return_property = prop.create_predefined_property(
    property_id="return",
    epsilon=0.025,
    kappa=0.05,
    relative_error=True,
    bounds=(-1, 1),
    sound=True,
)
evaluator.register_property(return_property)

# Custom properties are also possible.
collision_property = prop.create_custom_property(
    name="obstacle_collision_prob",
    check_fn=lambda self, t: float(t[-1][2] == -1),  # collision => final reward -1
    epsilon=0.05,
    kappa=0.05,
    relative_error=True,
    bounds=(0, 1),
    binomial=True,
)
evaluator.register_property(collision_property)

# Evaluate the agent with respect to the registered properties.
results = evaluator.eval(
    agent=agent,
    save_every_n_episodes=1000,
    time_limit=2.5,  # minutes
    stop_on_convergence=True,
)
```

To run it, install the example extras first:

```sh
pip install "pydsmc[examples]"
python simple_example.py
```

## More bundled agents

The repository's [`example_agents/`](https://github.com/neuro-mechanistic-modeling/PyDSMC/tree/main/example_agents)
directory contains ready-to-evaluate agents across a range of environments, including:

- Continuous control: `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5`, `HumanoidStandup-v5`,
  `MountainCarContinuous`
- Discrete/visual: `Breakout-v4`, MiniGrid (`Dynamic-Obstacles-6x6`, `MultiRoom-N6`)
- Custom: `pgtg-v3`

Each can be evaluated with the same pattern: load the agent, build the environment (optionally via
[`create_eval_envs`](usage.md#the-evaluator) for vectorized rollouts), register properties, and call
`eval`.

## Reading and aggregating results

Every run writes JSON-Lines logs under the evaluator's `log_dir`. Aggregate the latest run into a
pandas `DataFrame`:

```python
from pydsmc import jsons_to_df

# Aggregate the latest run; also writes an accumulated results file next to the logs.
df = jsons_to_df("example_logs")

# Include every intermediate result (the full time series), not just the final row:
df_ts = jsons_to_df("example_logs", include_timeseries=True, save=False)
```

The same functionality is available from the command line:

```sh
python -m pydsmc.json_parser --log-dir example_logs --verbose
```
