from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import pydsmc.statistics as st
from pydsmc.utils import DSMCLogger, is_bounded, is_valid_bounds

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# TODO: ONLY ADD PROPERTIES WE ACTUALLY WANT TO KEEP
# NOTE: when changing these, keep the table in `create_predefined_property`'s docstring in sync.
__PRE_DEFINED_PROPERTIES = {
    # Records the accumulated discounted reward
    "return": {
        "check_fn": lambda self, t: np.sum(
            [t[i].reward * np.power(self.gamma, i) for i in range(len(t))],
        ),
        "binomial": False,
        "bounds": (-np.inf, np.inf),
        "gamma": 0.99,
    },
    # Records the length of each episode, including truncations!
    "episode_length": {
        "check_fn": lambda self, t: len(t),
        "bounds": (0.0, np.inf),  # Upper bound should be truncationlimit + 1
        "binomial": False,
    },
    # Records how often the goal was reached. Goal reached here means that the last reward is at least the goal reward
    # Alas, no general easy way to verify whether a 'goal' was reached via the gymnasium interface
    "goal_reaching_probability": {
        "check_fn": lambda self, t: t[-1].reward >= self.goal_reward,
        "bounds": (0.0, 1.0),
        "binomial": True,
        "goal_reward": 1,
    },
    # Records whether the episode was forcefully terminated after reaching the maximum episode length
    "truncation": {
        "check_fn": lambda self, t: t[-1].truncated,
        "bounds": (0.0, 1.0),
        "binomial": True,
    },
    # Records unsuccessful terminations, excluding truncations
    # Unsuccessful means that the the accumulated _undiscounted_ reward is non-positive
    "unsuccesful_termination": {
        "check_fn": lambda self, t: t[-1].terminated and np.sum([s.reward for s in t]) <= 0,
        "bounds": (0.0, 1.0),
        "binomial": True,
    },
}


def get_predefined_properties() -> list[str]:
    """List the IDs of all available predefined properties.

    The returned IDs (e.g. `"return"`, `"episode_length"`,  `"goal_reaching_probability"`,
    `"truncation"`, `"unsuccesful_termination"`) can be passed as `property_id` to
    [`create_predefined_property`][pydsmc.create_predefined_property].

    Returns:
        The list of predefined property IDs.
    """
    return list(__PRE_DEFINED_PROPERTIES.keys())


class StepWrapper:
    __slots__ = ("_t",)

    def __init__(self, t: tuple):
        self._t = t

    @property
    def state(self):
        return self._t[0]

    @property
    def action(self):
        return self._t[1]

    @property
    def reward(self):
        return self._t[2]

    @property
    def terminated(self):
        return self._t[3]

    @property
    def truncated(self):
        return self._t[4]

    @property
    def info(self):
        return self._t[5]

    def __getitem__(self, index):
        return self._t[index]


# base class for evaluation properties
class Property:
    """A trajectory-based quantity to estimate, together with its statistical method.

    A property maps each trajectory to a single `float` sample via its `check_fn`; PyDSMC then
    estimates the mean of that value with a confidence interval using the associated statistical
    method. Prefer the factory functions
    [`create_predefined_property`][pydsmc.create_predefined_property] and
    [`create_custom_property`][pydsmc.create_custom_property] over constructing this class directly.

    Parameters such as `epsilon`, `kappa`, `bounds`, `binomial`, and `relative_error` are exposed as
    read-only properties that delegate to the underlying statistical method.

    Attributes:
        mean: The mean of the property, estimated from the collected samples.
        variance: The variance of the property, estimated from the collected samples.
        std: The standard deviation of the property, estimated from the collected samples.
        num_episodes: The number of episodes (samples) collected for this property.
        epsilon: The properties requested half-width of the confidence interval.
        kappa: The properties confidence level.
        binomial: Whether the property is binomial (a 0/1 indicator, e.g. a probability).
        bounds: The properties bounds, i.e. the minimum and maximum possible values of the property.
        relative_error: Whether the property uses relative error for its confidence interval.
        fallbacked: If the property has been switched to a fallback statistical method, this is the previous method. Otherwise `None`.
            On setup, the property will reset to its original statistical method if it was previously fallbacked.
    """

    property_dir: Path | None
    save_path: Path | None
    full_results_path: Path | None
    fallbacked: st.StatisticalMethod | None

    def __init__(
        self,
        name: str,
        st_method: st.StatisticalMethod,
        check_fn: Callable[[Property, list[tuple[Any, Any, Any, bool, bool, dict]]], float],
        *,
        max_episodes_zero_variance: int = 1000,  # TODO: Good default?
        **kwargs,
    ):
        # Attributes defining the property
        self.name = name
        self.st_method = st_method
        self.fallbacked = None
        # TODO: If we are doing it like this. no need for the s argument in check_fn.
        # 'self' is already available via closure
        self.check_fn = lambda s, t: check_fn(s, [StepWrapper(step) for step in t])
        self.max_episodes_zero_variance = max_episodes_zero_variance

        # Internals
        self.set_property_id()
        self.save_full_results = False
        self.results_lock = threading.Lock()

        # Store all additional keyword arguments as attributes of our new custom property
        for key, value in kwargs.items():
            # Ignore existing attributes, like the ones defined by the st_method (e.g. epsilon, kappa, ...)
            if not hasattr(self, key):
                setattr(self, key, value)

    # we assume a trajectory is a list of tuples (observation, action, reward, terminated, truncated, info)
    def check(self, trajectory: list[tuple[Any, Any, Any, bool, bool, dict]]) -> float:
        return self.check_fn(self, trajectory)

    def add_samples(self, xs: Iterable[float]) -> None:
        # Since `add_sample` is overwritten by different statistical methods
        # we do not want to make adding multiple samples more efficient than looping one-by-one
        for x in xs:
            self.st_method.add_sample(x)

    def add_sample(self, x: float) -> None:
        self.st_method.add_sample(x)

        if self.save_full_results:
            self.__full_results.append(x)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        """Get the current confidence interval and whether it has converged.

        Returns:
            A tuple `(converged, interval)` where `converged` is a boolean indicating whether the
                confidence interval has converged, and `interval` is a tuple `(lower_bound, upper_bound)`
                representing the current confidence interval, or `None` if no interval can be constructed yet.
        """
        if (
            self.max_episodes_zero_variance > 0
            and self.st_method.count > self.max_episodes_zero_variance
            and self.st_method.mean == 0.0
            and self.st_method.variance == 0.0
        ):
            return True, (0.0, 0.0)

        return self.st_method.get_interval()

    def converged(self) -> bool:
        converged, conf_intv = self.get_interval()
        return converged

    def setup_eval(self, log_dir: Path | str | None, save_full_results: bool = False) -> None:
        if self.fallbacked is not None:
            self.st_method = self.fallbacked
            self.fallbacked = None

        # Reset all accumulated statistics (including method-specific state such as
        # DKW's stored samples or EBStop's running bounds) so the method can be reused.
        self.st_method.reset()

        # Setup internals
        self.save_full_results = save_full_results
        self.__full_results = []
        if log_dir is not None:
            self.property_dir = Path(log_dir) / f"{self.name}_{self.property_id}"
            self.save_path = self.property_dir / "results.jsonl"
            self.full_results_path = self.property_dir / "full_results.jsonl"
        else:
            self.property_dir = None
            self.save_path = None
            self.full_results_path = None

    def save_settings(
        self,
        log_dir: Path | None = None,
        exclude: list | None = None,
        *,
        additional_entries: dict[str, Any] | None = None,
    ) -> dict:
        if exclude is None:
            exclude = []

        settings = self.__dict__.copy()
        for key in [
            *exclude,
            "fallbacked",
            "st_method",
            "check_fn",
            "results_lock",
            "save_full_results",
            "property_dir",
            "_Property__full_results",
            "save_path",
            "full_results_path",
        ]:
            settings.pop(key, None)
        settings["st_method"] = self.st_method.__class__.__name__
        settings["property_dir"] = str(self.property_dir) if self.property_dir is not None else None
        settings = settings | (additional_entries or {})

        if log_dir is None and self.property_dir is None:
            return settings

        if log_dir is not None:
            property_dir = log_dir / f"{self.name}_{self.property_id}"
        else:
            property_dir = self.property_dir

        property_dir.mkdir(parents=True, exist_ok=True)
        with (property_dir / "settings.json").open("w") as f:
            json.dump(settings, f, indent=4)

        return settings

    def get_log_line(self) -> dict[str, Any]:
        """Get a dictionary containing the current state and information of the property.

        This dictionary has the same format as the one saved to results.jsonl.

        Returns:
            A dictionary containing the current state and information of the property (`name`, `property_id`, `total_episodes`, `mean`, `variance`, `std`, `confidence_interval`, `intv_converged`).
        """
        results: dict[str, Any] = {}
        results["name"] = self.name
        results["property_id"] = self.property_id
        results["total_episodes"] = self.num_episodes
        results["mean"] = self.mean
        results["variance"] = self.variance
        results["std"] = self.std
        converged, intv = self.get_interval()
        results["confidence_interval"] = intv
        results["intv_converged"] = bool(converged)

        return results

    # Saves the results to a json file
    def save_results(
        self,
        overwrite: bool = False,
        logging_fn: Callable = print,
    ) -> None:
        if self.save_path is None:
            return

        results = self.get_log_line()

        with self.save_path.open("w" if overwrite else "a") as f:
            f.write(json.dumps(results, indent=None) + "\n")
        logging_fn(f"Results for property {self.name} saved to {self.save_path}")

    def dump_buffer(self, overwrite: bool = False) -> None:
        if self.full_results_path is None:
            return

        with self.results_lock:
            if self.__full_results:
                with self.full_results_path.open("w" if overwrite else "a") as f:
                    for datapoint in self.__full_results:
                        f.write(f"{datapoint}\n")

                self.__full_results.clear()

    def set_property_id(self) -> None:
        current_time = str(time.time_ns())
        property_id = hashlib.sha256(current_time.encode()).hexdigest()[:6]
        self.property_id = property_id

    def fallback_st_method(self) -> st.StatisticalMethod | None:
        assert self.epsilon is not None, "Can only fallback in case of sequential setting"

        # We cannot replace by sound method if it is not binomial since we do not store all results
        # that are needed for the sound DKW method.
        fb_sound = self.st_method.is_sound() and self.binomial

        new_st_method = select_statistical_method(
            epsilon=None,  # Fallback means fixed run setting -> no epsilon
            kappa=self.kappa,
            bounds=self.bounds,
            relative_error=False,  # No relative error for fixed run setting
            binomial=self.binomial,
            sound=fb_sound,
        )
        if new_st_method.__class__ == self.st_method.__class__:
            return None

        prev_mean = self.mean
        prev_variance = self.variance
        prev_std = self.std
        prev_count = self.num_episodes
        prev_m2 = self.st_method._m2

        self.fallbacked = self.st_method
        self.st_method = new_st_method
        self.st_method.count = prev_count
        self.st_method.mean = prev_mean
        self.st_method.variance = prev_variance
        self.st_method.stddev = prev_std
        self.st_method._m2 = prev_m2

        return self.fallbacked

    # Refer to statistical method, to avoid errors due to duplicated variables
    @property
    def kappa(self) -> float:
        return self.st_method.kappa

    @property
    def epsilon(self) -> float | None:
        return self.st_method.epsilon

    @property
    def binomial(self) -> bool:
        return self.st_method.binomial

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        return (self.st_method.a, self.st_method.b)

    @property
    def relative_error(self) -> bool:
        return self.st_method.relative_error

    @property
    def num_episodes(self) -> int:
        return self.st_method.count

    @property
    def mean(self) -> float:
        return self.st_method.mean

    @property
    def variance(self) -> float:
        return self.st_method.variance

    @property
    def std(self) -> float:
        return self.st_method.stddev


# select the appropriate statistical method based on the given parameters
def select_statistical_method(
    epsilon: float | None = 0.1,
    kappa: float = 0.05,
    bounds: tuple[float | None, float | None] = (-np.inf, np.inf),
    min_samples: int | None = None,
    *,
    relative_error: bool = False,
    binomial: bool = False,
    sound: bool = False,
    **kwargs,  # Ignore additional arguments; unknown args will be used by the Property constructor
) -> st.StatisticalMethod:
    if not is_valid_bounds(bounds):
        raise ValueError(f"Invalid bounds: {bounds}")
    if epsilon is None and relative_error:
        raise ValueError("Relative error requires epsilon to be set")
    if sound and not is_bounded(bounds):
        raise ValueError("Soundness only supported for bounded properties")

    if sound:
        DSMCLogger.get_logger().warning(
            "Selected a statistics method with soundness guarantees. Converging will probably take a very long time.",
        )

    args = (
        epsilon,
        kappa,
        relative_error,
        bounds,
        binomial,
        *(() if min_samples is None else (min_samples,)),
    )

    if epsilon is not None:  # Sequential setting; runs until confidence level is reached
        if binomial:
            if not sound:
                return st.NormalIntervalMethod(*args)

            if relative_error:
                return st.EBStopMethod(*args)
            return st.ClopperPearsonIntervalMethod(*args)

        if is_bounded(bounds) and sound:
            if relative_error:
                return st.EBStopMethod(*args)
            return st.HoeffdingMethod(*args)

        # for unbounded or unsound properties
        return st.StudentsTMethod(*args)

    if binomial:
        if sound:
            return st.ClopperPearsonIntervalMethod(*args)
        return st.NormalIntervalMethod(*args)

    if is_bounded(bounds) and sound:
        return st.DKWMethod(*args)

    # for unbounded or unsound properties
    return st.StudentsTMethod(*args)


# create a pre-defined property that can be registered with the evaluator
def create_predefined_property(
    property_id: str,
    st_method: st.StatisticalMethod | None = None,
    epsilon: float | None = 0.1,
    kappa: float = 0.05,
    min_samples: int | None = None,
    name: str | None = None,
    *,
    relative_error: bool = False,
    sound: bool = False,
    **kwargs: Any,
) -> Property:
    """Create a ready-made, parameterized property.

    Unless an explicit `st_method` is given, a suitable statistical method is selected automatically
    from the provided parameters. The available `property_id` values (also returned by
    [`get_predefined_properties`][pydsmc.get_predefined_properties]) are:

    | `property_id`                 | Measures (per episode)                        | Binomial | Default `bounds` | Property-specific `**kwargs` |
    | ----------------------------- | --------------------------------------------- | -------- | ---------------- | ---------------------------- |
    | `"return"`                    | Acc. discounted reward                        | No       | `(-inf, inf)`    | `gamma` (default `0.99`)     |
    | `"episode_length"`            | #Steps, incl. truncation                      | No       | `(0, inf)`       | —                            |
    | `"goal_reaching_probability"` | Perc. of runs with return &geq; `goal_reward` | Yes      | `(0, 1)`         | `goal_reward` (default `1`)  |
    | `"truncation"`                | Perc. of truncated runs                       | Yes      | `(0, 1)`         | —                            |
    | `"unsuccesful_termination"`   | Perc. terminated with non-positive return     | Yes      | `(0, 1)`         | —                            |

    Any of the defaults above (`bounds`, `gamma`, `goal_reward`, ...) can be overridden via `**kwargs`.

    Args:
        property_id: ID of the predefined property (e.g. `"return"`, `"episode_length"`).
        st_method: An explicit statistical method. If `None`, one is selected automatically.
        epsilon: Half-width of the requested confidence interval. With `relative_error=True` it is
            relative to the estimate, otherwise absolute. Pass `None` to run a fixed number of
            episodes without a convergence target.
        kappa: Error probability; the true mean lies within the CI with prob. `1 - kappa`.
        min_samples: Minimum number of samples before convergence may be declared.
        name: Name used when storing results. Defaults to `property_id`.
        relative_error: Whether `epsilon` is a relative (`True`) or absolute (`False`) error.
        sound: Whether a sound statistical method (guaranteed coverage) must be used.
        **kwargs: Property-specific attributes (e.g. `bounds`, `gamma`, `goal_reward`) that override
            the predefined defaults.

    Returns:
        The configured [`Property`][pydsmc.Property].

    Raises:
        ValueError: If `property_id` is not a known predefined property.
    """
    if name is None:
        name = property_id

    if property_id not in __PRE_DEFINED_PROPERTIES:
        raise ValueError(f"Predefined property '{property_id}' not found")

    property_parameters = __PRE_DEFINED_PROPERTIES[property_id] | kwargs
    # determine which statistical method should be used to compute the confidence interval
    if st_method is None:
        st_method = select_statistical_method(
            epsilon=epsilon,
            kappa=kappa,
            relative_error=relative_error,
            min_samples=min_samples,
            sound=sound,
            **property_parameters,
        )
        DSMCLogger.get_logger().debug(
            f"Automatically selected statistical method {st_method.__class__.__name__} for predefined property {name}",
        )

    return Property(
        name=name,
        st_method=st_method,
        **property_parameters,
    )


# create a new property that can be registered with the evaluator
def create_custom_property(
    name: str,
    check_fn: Callable[[Property, list[tuple[Any, Any, Any, bool, bool, dict]]], float],
    st_method: st.StatisticalMethod | None = None,
    epsilon: float | None = 0.1,
    kappa: float = 0.05,
    bounds: tuple[float | None, float | None] = (-np.inf, np.inf),
    min_samples: int | None = None,
    *,
    relative_error: bool = False,
    binomial: bool = False,
    sound: bool = False,
    **kwargs: Any,
) -> Property:
    """Create a custom property from a user-provided checking function.

    The `check_fn` receives the property instance and a trajectory — a list of steps, each a tuple
    `(observation, action, reward, terminated, truncated, info)` — and returns a `float` sample.
    Unless an explicit `st_method` is given, a suitable statistical method is selected automatically.

    Args:
        name: Name used when storing results.
        check_fn: Callable `check_fn(property, trajectory) -> float` producing one sample per
            trajectory. `trajectory[-1]` is the final step; `trajectory[-1][2]` is the last reward.
        st_method: An explicit statistical method. If `None`, one is selected automatically.
        epsilon: Half-width of the requested confidence interval (relative if `relative_error`,
            otherwise absolute). Pass `None` for a fixed-run setting without a convergence target.
        kappa: Error probability; the true mean lies within the CI with prob. `1 - kappa`.
        bounds: The property's `(min, max)` possible values. Required for sound/bounded methods.
        min_samples: Minimum number of samples before convergence may be declared.
        relative_error: Whether `epsilon` is a relative (`True`) or absolute (`False`) error.
        binomial: Whether the property is binomial (a 0/1 indicator, e.g. a probability).
        sound: Whether a sound statistical method (guaranteed coverage) must be used.
        **kwargs: Extra attributes stored on the resulting property.

    Returns:
        The configured [`Property`][pydsmc.Property].
    """
    # determine which statistical method should be used to compute the confidence interval
    if st_method is None:
        st_method = select_statistical_method(
            epsilon=epsilon,
            kappa=kappa,
            relative_error=relative_error,
            bounds=bounds,
            binomial=binomial,
            min_samples=min_samples,
            sound=sound,
        )
        DSMCLogger.get_logger().debug(
            f"Automatically selected statistical method {st_method.__class__.__name__} for custom property {name}",
        )

    return Property(
        name=name,
        st_method=st_method,
        check_fn=check_fn,
        **kwargs,
    )
