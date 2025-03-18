# PyDSMC
***Statistical Model Checking for Neural Agents Using the Gymnasium Interface***

<!-- Badges -->
![Python](https://img.shields.io/pypi/pyversions/pydsmc)
![PyPI](https://img.shields.io/pypi/v/pydsmc)
![Downloads](https://img.shields.io/pepy/dt/pydsmc)
![Tests](https://github.com/neuro-mechanistic-modeling/PyDSMC/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/github/license/neuro-mechanistic-modeling/PyDSMC)

<!-- SHORT DESCRIPTION OF THE TOOL -->
PyDSMC is an open-source Python library for statistical model checking of neural agents on arbitrary [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments.
It is designed to be lightweight and easy-to-use while being based on established statistical methods to provide guarantees on the investigated properties' performances.
Implementing the Gymnasium interface, PyDSMC is widely appliacble and fully agnostic to the environments underlying implementation.

PyDSMC is based on [Deep Statistical Model Checking](https://doi.org/10.1007/978-3-030-50086-3_6) and aims to facilitate greater adoption of statistical model checking by simplifying its usage.

<!-- Putting a motivational plot somewhere, a visualization, might be pleasing and show what we can do -->

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
<!-- KEY FEATURES -->
<!-- More details either here, or later -->


## Setup
<!-- INSTALLATION -->

PyDSMC can be installed using `pip install pydsmc`.

We recommend using a virtual environment and officially tested python versions 3.10, 3.11, and 3.12.

To set up a virtual environment and install all necessary dependencies you can, for example, execute:
```sh
mkvirtualenv --python=python3.10 dsmc
pip install -r requirements.txt
```

## Usage
<!-- Usage -->

### Properties
<!-- Predefined properties; with a list of predefined properties? -->
PyDSMC can analyze arbitrary properties. These can also be environment specific. For ease-of-use, we provide ready-to-use implementations of commonly used, domain-independent properties that are parameterized and can, thus, be adjusted to each individual usecase.

Creating a predefined property is straightforward. For instance, a property analyzing the achieved return could be defined as follows:
```python
from pydsmc import create_predefined_property

return_property = create_predefined_property(
    property_id='return',   # Which predefined property to use
    name='returnGamma0.99', # Property's name, used for storing the evaluation results
    eps=0.025,              # Half-width of the requested confidence interval (CI)
    kappa=0.05,             # Probability that the true mean lies within the CI
    relative_error=True,    # Whether eps represents the relative or absolute error
    bounds=(0, 864),        # Bounds of the property, i.e., min and max possible values
    sound=True              # Whether a sound statistical method should be used
    gamma=0.97              # Property specific attributes
)
```

<!-- Custom properties -->
Creating a custom property is equally simple:
```python
from pydsmc import create_custom_property

crash_property = create_custom_property(
    name='crash_prob',      # see above
    eps=0.025,              # see above
    kappa=0.05,             # see above
    relative_error=False,   # see above
    binomial=True,          # This property follows a binomial distribution
    bounds=(0, 1),          # see above
    sound=False,            # see above
    # The property's checking function, crash identified by last reward -100
    check_fn=lambda self, t: float(t[-1][2] == -100)
)
```
<!-- blabla, only checking function differs. -->

### Evaluator
<!-- Having defined properties, an evaluator -->


### Full example
A few full examples on a select set of environments can be found in [example_agents](example_agents/) to try out.


## Parameters
<!-- Customization/Parameters exaplanation? -->

## License
The code introduced by this project is licensed under the MIT license. Please consult the bundled LICENSE file for the full license text.

## Statistical Method Selection
### Figure
<img src="assets/sm_overview_gaps.svg" width="100%">

### Mermaid graph
```mermaid
%%{ init: { 'flowchart': { 'curve': 'step' } } }%%
flowchart TD;
  classDef sm fill:#b2ffb2,color:#000000
  classDef dec fill:#ccccff,color:#000000

  A{"setting"} -->|fixed run| B["sound"]:::dec
  style A fill:#ffb266,color:#000
  A -->|sequential| C["sound"]:::dec

  B -->|no| D["property"]:::dec
  B -->|yes| E["property"]:::dec

  D -->|unbounded or bounded| F(["Student’s-t intervals"]):::sm
  D -->|binomial| G(["normal intervals"]):::sm

  E -->|bounded| I(["DKW"]):::sm
  E -->|binomial| J(["Wilson score w/cc"]):::sm

  C -->|no| K["property"]:::dec
  C -->|yes| L["property"]:::dec

  K -->|binomial| M["interval width"]:::dec
  K -->|bounded| N["interval width"]:::dec

  M -->|absolute| J
  M -->|relative| O(["EBStop"]):::sm

  N -->|absolute| Q(["Hoeffding’s inequality"]):::sm
  N -->|relative| O
  L -->|unbounded or bounded| R(["sequential Student’s-t"]):::sm
  L -->|binomial| S(["Chow-Robbins"]):::sm

  subgraph iw[" "]
    M
    N
  end

  subgraph prop[" "]
    D
    E
    K
    L
  end

  subgraph sound[" "]
    B
    C
  end

  subgraph StatMethod[" "]
    I
    J
    G
    F
    O
    Q
    R
    S
  end

  style StatMethod fill:none,stroke:none,heading:none;
  style sound fill:none,stroke:none,heading:none;
  style prop fill:none,stroke:none,heading:none;
  style iw fill:none,stroke:none,heading:none;
```
