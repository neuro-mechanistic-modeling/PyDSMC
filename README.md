# PyDSMC

**_Statistical Model Checking for Neural Agents Using the Gymnasium Interface_**

<!-- Badges -->

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/pydsmc)](https://pypi.org/project/pydsmc/)
![Downloads](https://img.shields.io/pepy/dt/pydsmc)
![Python](https://img.shields.io/pypi/pyversions/pydsmc)
[![Tests](https://github.com/neuro-mechanistic-modeling/PyDSMC/actions/workflows/tests.yml/badge.svg)](tests)
[![License](https://img.shields.io/github/license/neuro-mechanistic-modeling/PyDSMC)](LICENSE)

<!-- SHORT DESCRIPTION OF THE TOOL -->

PyDSMC is an open-source Python library for statistical model checking of neural agents on arbitrary [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments.
It is designed to be lightweight and easy-to-use while being based on [established statistical methods](#statistical-method-selection) to provide guarantees on the investigated agents' performances on various properties.
Implementing the Gymnasium interface, PyDSMC is widely applicable and fully agnostic to the environments underlying implementation.

PyDSMC is based on [Deep Statistical Model Checking](https://doi.org/10.1007/978-3-030-50086-3_6) and aims to facilitate greater adoption of statistical model checking by simplifying its usage.

Please consult the [accompanying paper](https://woshicado.eu/publications/2025/pydsmc/PyDSMC.pdf) for more details.

## Table of Contents

- [Deep Statistical Model Checking](#deep-statistical-model-checking)
- [Setup](#setup)
- [Usage](#usage)
  - [Properties](#properties)
  - [Evaluator](#evaluator)
  - [Full example](#full-example)
- [Parameters](#parameters)
- [License](#license)
- [Statistical Method Selection](#statistical-method-selection)
  - [Figure](#figure)
  - [Mermaid graph](#mermaid-graph)

## Deep Statistical Model Checking

The train and evaluation curves of neural agents differ greatly throughout training. Just inspecting the training curves does not suffice and leads to suboptimal policy extraction. We advocate to perform statistically-backed model evaluations periodically throughout training to ensure a good extraction point.

<img src="assets/motivation.svg" width="100%">

## Setup

PyDSMC can be installed from [PyPI](https://pypi.org/project/pydsmc/) using `pip install pydsmc`.

We recommend using a virtual environment and officially tested python versions 3.10, 3.11, and 3.12.

To **work on this project**, set up a virtual environment and install all necessary dependencies---for instance, using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) and executing:

```sh
mkvirtualenv --python=python3.11 pydsmc
pip install -r requirements.txt
```

## Usage

PyDSMC mainly exposes two functionalities: **Property** creation and the **Evaluator**.

### Properties

PyDSMC can analyze arbitrary properties. These can also be environment-specific. For ease-of-use, we provide ready-to-use implementations of commonly used, domain-independent properties that are parameterized and can, thus, be adjusted to each individual usecase. You can get a list of all available predefined properties by calling `pydsmc.get_predefined_properties()`.

Creating a predefined property is straightforward. For instance, a property analyzing the achieved return could be defined as follows:

```python
from pydsmc import create_predefined_property

return_property = create_predefined_property(
    property_id='return',   # Which predefined property to use
    name='returnGamma0.97', # Property's name, used for storing the evaluation results
    epsilon=0.025,          # Half-width of the requested confidence interval (CI)
    kappa=0.05,             # Probability that the true mean lies within the CI
    relative_error=True,    # Whether epsilon represents the relative or absolute error
    bounds=(0, 864),        # Bounds of the property, i.e., min and max possible values
    sound=True,             # Whether a sound statistical method should be used
    gamma=0.97              # Property specific attributes
)
```

Creating a custom property is equally simple. The only additional information required is a checking function `check_fn` that analyzes trajectory information and returns a float value as a sample result for the property:

```python
from pydsmc import create_custom_property

crash_property = create_custom_property(
    name='crash_prob',      # see above
    epsilon=0.025,          # see above
    kappa=0.05,             # see above
    relative_error=False,   # see above
    binomial=True,          # This property follows a binomial distribution
    bounds=(0, 1),          # see above
    sound=False,            # see above
    # The property's checking function, crash identified by last reward -100
    check_fn=lambda self, t: float(t[-1][2] == -100)
)
```

### Evaluator

We can then register the properties in an `Evaluator` which will do the work of evaluating all the properties until convergence is reached, or the given resource budget is exhausted.

In order to construct an `Evaluator`, it is first necessary to create the evaluation environments from the outside. For this, we provide a helper function `create_eval_envs` that creates a suitable vectorized environment according to the given configuration.

```python
from pydsmc import create_eval_envs, Evaluator

NUM_THREADS = 1   # Recommended: 1; Often better to use more parallel environments instead
NUM_PAR_ENVS = 8  # Number of parallel environments _per thread_

envs = create_eval_envs(
    num_threads=NUM_THREADS,
    num_envs_per_thread=NUM_PAR_ENVS,
    env_seed=42,  # Seeds are incremented for each environment, making them unique
    gym_id="HalfCheetah-v5",
    wrappers=[gym.wrappers.NormalizeObservation, Monitor],
    vecenv_cls=gym.vector.AsyncVectorEnv,  # gym.vector.SyncVectorEnv, sb3.DummyVecEnv, sb3.SubprocVecEnv
    # The following kwargs are passed to the gym.make function, which passes unknown args to the env
    max_episode_steps=1000,
)

evaluator = Evaluator(env=envs)
```

Having created the `Evaluator`, we can register the properties and start the evaluation:

```python
evaluator.register_property(return_property)
evaluator.register_property(crash_property)
### Or register multiple properties at once using:
# evaluator.register_properties([return_property, crash_property])

### Start the evaluation
evaluator.eval(
    agent=agent,  # or: predict_fn=agent.predict
    stop_on_convergence=True,  # Stop when all properties have converged
    episode_limit=10_000,  # Early stop evaluation after 10,000 episodes
    time_limit=60,  # Early stop evaluation after 1 hour
    num_episodes_per_policy_run=100,
    num_threads=NUM_THREADS,
    deterministic=True,
)
```

### Full example

A few full examples on a select set of environments can be found in [example_agents](example_agents/) to try out.

## Statistical Method Selection

For each property, PyDSMC automatically selects an appropriate statistical method based on the provided parameters, following decision tree below:

<img src="assets/sm_overview_gaps.svg" width="100%">

## Citing PyDSMC

If you use PyDSMC in your work, you can use the following BibTeX entry, citing the official Version of Record published at QEST2025.

```bibtex
@inproceedings{grosPyDSMC2025,
  title      = {{{PyDSMC}}: {{Statistical Model Checking}} for~{{Neural Agents Using}} the~{{Gymnasium Interface}}},
  shorttitle = {{{PyDSMC}}},
  booktitle  = {Quantitative {{Evaluation}} of {{Systems}} and {{Formal Modeling}} and {{Analysis}} of {{Timed Systems}}},
  author     = {Gros, Timo P. and Hartmanns, Arnd and Hoese, Ivo and Meyer, Joshua and M{\"u}ller, Nicola J. and Wolf, Verena},
  editor     = {Prabhakar, Pavithra and Vandin, Andrea},
  year       = {2025},
  pages      = {134--156},
  publisher  = {Springer Nature Switzerland},
  address    = {Cham},
  doi        = {10.1007/978-3-032-05792-1_8},
  isbn       = {978-3-032-05792-1},
}
```

## License

The code introduced by this project is licensed under the MIT license. Please consult the bundled LICENSE file for the full license text.
