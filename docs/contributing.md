# Contributing

Contributions are welcome and can consist of bug reports, feature requests, documentation improvements, and pull requests.
This page describes how to set up a development environment and the conventions we follow.

## Development setup

PyDSMC uses [mise](https://mise.jdx.dev/) (with [uv](https://github.com/astral-sh/uv)) to manage the toolchain and common tasks.

```sh
git clone https://github.com/neuro-mechanistic-modeling/PyDSMC.git
cd PyDSMC

# Installs the pinned Python + uv, creates .venv, installs all extras, sets up pre-commit hooks.
mise run install
```

If you prefer not to use mise, a plain virtual environment works too:

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
```

## Code style & linting

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting, enforced through [pre-commit](https://pre-commit.com/).
The hooks run automatically on commit once installed; you can also run them on demand:

```sh
pre-commit run --all-files
```

Please keep changes consistent with the surrounding code and ensure the hooks pass before opening a pull request.

## Running the tests

```sh
mise run test # installs dev deps, sets up pre-commit, and runs pytest
# or, directly:
pytest tests/
```

## Building the documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), with the API reference generated
by [mkdocstrings](https://mkdocstrings.github.io/).

```sh
mise run docs --serve # live-reloading preview at http://127.0.0.1:8000
mise run docs         # production build into site/ (fails on warnings)
```

The public API reference is generated from docstrings, so when you change a public function's signature or behavior, please update its docstring accordingly.
Documentation is deployed automatically to GitHub Pages on every push to `main`.

## Reporting issues

Please open issues on the [GitHub issue tracker](https://github.com/neuro-mechanistic-modeling/PyDSMC/issues).
For bugs, a minimal reproducible example (environment, property configuration, and the observed vs. expected behavior) helps a lot.
