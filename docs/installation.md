# Installation

## From PyPI

PyDSMC is published on [PyPI](https://pypi.org/project/pydsmc/):

```sh
pip install pydsmc
```

We recommend using a virtual environment and officially tested Python versions 3.10, 3.11, and 3.12.

Only the lightweight core dependencies (`numpy`, `scipy`, `gymnasium`, `pandas`, `packaging`) are installed by default.
In particular, PyDSMC does _not_ pull in a deep-learning framework, so you are free to bring your own (PyTorch, JAX, TensorFlow, …).

## Optional extras

PyDSMC ships a few optional dependency groups:

| Extra      | Install                          | Contents                                                                                                      |
| ---------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `examples` | `pip install "pydsmc[examples]"` | Environments/agents used by the bundled examples (`minigrid`, `ale-py`, `gymnasium-robotics`, `sb3-contrib`). |
| `dev`      | `pip install "pydsmc[dev]"`      | Test & tooling stack (`pytest`, `pre-commit`, `stable-baselines3`, `pgtg`, `ipython`).                        |
| `docs`     | `pip install "pydsmc[docs]"`     | Documentation stack (`mkdocs-material`, `mkdocstrings`, `mkdocs-marimo`).                                     |
| `all`      | `pip install "pydsmc[all]"`      | Everything above.                                                                                             |

## Development setup (with mise)

This project uses [mise](https://mise.jdx.dev/) to manage the Python toolchain and common tasks,
backed by [uv](https://github.com/astral-sh/uv) for fast installs. If you want to work on PyDSMC:

```sh
git clone https://github.com/neuro-mechanistic-modeling/PyDSMC.git
cd PyDSMC

# Installs the pinned Python + uv, creates the .venv, installs all extras, and sets up pre-commit.
mise run install
```

Handy tasks (`mise tasks` lists them all):

| Task                      | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `mise run install`        | Install all extras (`pydsmc[all]`) and pre-commit hooks. |
| `mise run test`           | Install dev deps and run the test suite.                 |
| `mise run docs [--serve]` | Build [Serve] the documentation.                         |
| `mise run info`           | Print project/environment information.                   |

Prefer a manual setup? A classic virtual environment works too:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
```

See [Contributing](contributing.md) for the full contributor workflow.
