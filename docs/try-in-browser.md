# Try in Browser

The example below runs **PyDSMC entirely in your browser**.
It uses [marimo](https://marimo.io/) compiled to WebAssembly, so the Python code executes client-side via [Pyodide](https://pyodide.org/).

!!! info "What this demo does (and its limits)"
    Deep-learning frameworks such as PyTorch and Stable-Baselines3 are **not available** in
    WebAssembly, so this demo uses a tiny **hand-coded** CartPole controller instead of a trained
    neural agent. Everything else is the real PyDSMC: the same `Evaluator`, the same predefined
    `return` property, and the same statistical confidence interval you would get locally.

    Locally you would instead pass a trained `agent` (or any `predict_fn`) and typically use more
    parallel environments. See [Usage](usage.md) and [Examples](examples.md).

!!! note "First run takes a few seconds"
    The first cell downloads and installs PyDSMC and its dependencies into the in-browser Python
    runtime. Give it a moment; subsequent runs are instant.

## Live demo

The cell below is a **live, editable** notebook running entirely in your browser.
On load it installs PyDSMC and statistically checks a hand-coded CartPole policy.

!!! tip "Edit it and run it"
    Click into the code, change something (e.g. `epsilon` for a tighter/looser confidence interval)
    or the policy itself; then click **Run** to rerun the code and see the estimates change.

```python {marimo display_code=true}
import micropip

# Install dependencies
await micropip.install(["numpy", "scipy", "pandas"])
await micropip.install("gymnasium")
await micropip.install("pydsmc", deps=False)

import gymnasium as gym
import numpy as np
import marimo as mo

from pydsmc import Evaluator, create_predefined_property


def policy(obs, hidden_states=None, episode_starts=None, **kwargs):
    """
    A simple, hard-coded CartPole policy.
    In a real-scenario, this would be a trained neural agent.
    """
    # `obs` is a batch of observations (due to vectorized envs).
    obs = np.asarray(obs)
    # Push the cart toward the side the pole is falling
    return (obs[:, 2] + obs[:, 3] > 0).astype(np.int64), None  # (actions, hidden_states)


# 4 parallel CartPole environments. SyncVectorEnv avoids multiprocessing
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])

evaluator = Evaluator(env=envs, log_dir=None)  # log_dir=None => no file writes
evaluator.register_property(
    create_predefined_property(
        property_id="return", # predefined property
        name="return",        # custom name
        epsilon=0.1,          # requested (relative) half-width of the CI
        kappa=0.1,            # confidence level = 1 - kappa = 90%
        relative_error=True,  # relative epsilon
        bounds=(0, 500),      # CartPole-v1 caps episodes at 500 steps
        sound=False,          # do not choose a sound statistical method
        gamma=1.0,            # undiscounted return
    ),
)

# num_threads defaults to 1 (recommended)
prop = evaluator.eval(
    predict_fn=policy,
    stop_on_convergence=True,
    episode_limit=3000,
    num_initial_episodes=200,
    num_episodes_per_policy_run=100,
)[0]

converged, ci = prop.get_interval()
mo.md(f"""
**Result** — statistically checked a hand-coded CartPole policy:

| Metric                                            | Value                                |
| ------------------------------------------------- | ------------------------------------ |
| Episodes evaluated                                | **{prop.num_episodes}**              |
| Mean return                                       | **{prop.mean:.1f}** ± {prop.std:.1f} |
| {(1-prop.epsilon) * 100:.1f}% confidence interval | **[{ci[0]:.1f}, {ci[1]:.1f}]**       |
| Converged                                         | **{converged}**                      |
"""
)
```

## Run it locally

Prefer to run on a real trained agent? Install PyDSMC and follow the [Examples](examples.md):

```sh
pip install "pydsmc"
```
