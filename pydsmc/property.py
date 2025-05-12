from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

import pydsmc.statistics as st
from pydsmc.utils import DSMCLogger, is_bounded, is_valid_bounds

SelfProperty = TypeVar("SelfProperty", bound="Property")

# TODO: ONLY ADD PROPERTIES WE ACTUALLY WANT TO KEEP
__PRE_DEFINED_PROPERTIES = {
    # Records the accumulated discounted reward
    "return": {
        "check_fn": lambda self, t: np.sum(
            [t[i][2] * np.power(self.gamma, i) for i in range(len(t))],
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
        "check_fn": lambda self, t: t[-1][2] >= self.goal_reward,
        "bounds": (0.0, 1.0),
        "binomial": True,
        "goal_reward": 1,
    },
    # Records whether the episode was forcefully terminated after reaching the maximum episode length
    "truncation": {
        "check_fn": lambda self, t: t[-1][4],
        "bounds": (0.0, 1.0),
        "binomial": True,
    },
    # Records unsuccesful terminations, excluding truncations
    # Unsuccesful means that the the accumulated _undiscounted_ reward is non-positive
    "unsuccesful_termination": {
        "check_fn": lambda self, t: t[-1][3] and np.sum([s[2] for s in t]) <= 0,
        "bounds": (0.0, 1.0),
        "binomial": True,
    },
}
# TODO: Legacy, remove ig.
__PRE_DEFINED_PROPERTIES["goal_reaching_prob"] = __PRE_DEFINED_PROPERTIES[
    "goal_reaching_probability"
]


# base class for evaluation properties
class Property:
    def __init__(
        self,
        name: str,
        st_method: st.StatisticalMethod,
        check_fn: Callable[[SelfProperty, list[tuple[Any, Any, Any, bool, bool, Any]]], float]
        | None = None,
        epsilon: float | None = 0.05,
        kappa: float = 0.05,
        bounds: tuple[float | None, float | None] = (None, None),
        *,
        relative_error: bool = False,
        binomial: bool = False,
        max_episodes_zero_variance: int = 1000,  # TODO: Good default?
        **kwargs,
    ):
        # Attributes defining the property
        self.name = name
        self.st_method = st_method
        self.check_fn = check_fn
        self.epsilon = epsilon
        self.kappa = kappa
        self.relative_error = relative_error
        self.bounds = bounds
        self.binomial = binomial
        self.max_episodes_zero_variance = max_episodes_zero_variance

        # Internals
        self.set_property_id()
        self.save_full_results = False
        self.results_lock = threading.Lock()

        if not is_valid_bounds(bounds):
            raise ValueError(f"Invalid bounds: {bounds}")
        if epsilon is None and relative_error:
            raise ValueError("Relative error requires epsilon to be set")

        # Store all additional keyword arguments as attributes of our new custom property
        for key, value in kwargs.items():
            setattr(self, key, value)

    # we assume a trajectory is a list of tuples (observation, action, reward, terminated, truncated, info)
    def check(self, trajectory: list[tuple[Any, Any, Any, bool, bool, Any]]) -> float:
        return self.check_fn(self, trajectory)

    def add_samples(self, xs: np.array) -> None:
        # Since `add_sample` is overwritten by different statistical methods
        # we do not want to make adding multiple samples more efficient than looping one-by-one
        for x in xs:
            self.st_method.add_sample(x)

    def add_sample(self, x) -> None:
        self.st_method.add_sample(x)

        if self.save_full_results:
            self.__full_results.append(x)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
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

    def setup_eval(self, log_dir: Path | str, save_full_results: bool = False) -> None:
        self.st_method.count = 0
        self.st_method.mean = 0.0
        self.st_method._m2 = 0.0
        self.st_method.variance = 0.0
        self.st_method.stddev = 0.0

        # Setup internals
        self.save_full_results = save_full_results
        self.__full_results = []
        self.property_dir = Path(log_dir) / f"{self.name}_{self.property_id}"
        self.save_path = self.property_dir / "results.jsonl"
        self.full_results_path = self.property_dir / "full_results.jsonl"

    def save_settings(self, log_dir: Path, exclude: list | None = None) -> None:
        if exclude is None:
            exclude = []
        settings = self.__dict__.copy()
        for key in [
            *exclude,
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
        settings["property_dir"] = str(self.property_dir)

        property_dir = log_dir / f"{self.name}_{self.property_id}"
        property_dir.mkdir(parents=True, exist_ok=True)
        with (property_dir / "settings.json").open("w") as f:
            json.dump(settings, f, indent=4)

    # Saves the results to a json file
    def save_results(
        self,
        overwrite: bool = False,
        logging_fn: Callable = print,
    ):
        results = {}
        results["name"] = self.name
        results["property_id"] = self.property_id
        results["total_episodes"] = self.num_episodes
        results["mean"] = self.mean
        results["variance"] = self.variance
        results["std"] = self.std
        converged, intv = self.get_interval()
        results["confidence_interval"] = intv
        results["intv_converged"] = bool(converged)

        with self.save_path.open("w" if overwrite else "a") as f:
            f.write(json.dumps(results, indent=None) + "\n")
        logging_fn(f"Results for property {self.name} saved to {self.save_path}")

    def dump_buffer(self, overwrite: bool = False):
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
    **kwargs,
) -> Property:
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
        epsilon=epsilon,
        kappa=kappa,
        relative_error=relative_error,
        **property_parameters,
    )


# create a new property that can be registered with the evaluator
def create_custom_property(
    name: str,
    check_fn: Callable[[Property, list[tuple[Any, Any, Any, bool, bool, Any]]], float],
    st_method: st.StatisticalMethod = None,
    epsilon: float | None = 0.1,
    kappa: float = 0.05,
    bounds: tuple[float | None, float | None] = (-np.inf, np.inf),
    min_samples: int | None = None,
    *,
    relative_error: bool = False,
    binomial: bool = False,
    sound: bool = False,
    **kwargs,
) -> Property:
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
        epsilon=epsilon,
        kappa=kappa,
        relative_error=relative_error,
        bounds=bounds,
        binomial=binomial,
        **kwargs,
    )
