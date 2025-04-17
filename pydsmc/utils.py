from __future__ import annotations

import json
import logging
from typing import ClassVar

import gymnasium as gym
import numpy as np


def is_bounded(bounds: tuple[float | None, float | None]) -> bool:
    return (
        bounds[0] is not None
        and bounds[1] is not None
        and not np.isinf(bounds[0])
        and not np.isinf(bounds[1])
    )


def is_valid_bounds(bounds: tuple[float | None, float | None]) -> bool:
    return bounds[0] is None or bounds[1] is None or bounds[0] < bounds[1]


def create_eval_envs(
    num_threads: int,
    num_envs_per_thread: int,
    env_seed: int,
    gym_id: str,
    wrappers: list[gym.Wrapper] | None = None,
    vecenv_cls=gym.vector.AsyncVectorEnv,  # Tested: gym.VectorEnv, sb3.VecEnv. gym VecEnvs recommended
    **gym_kwargs,
) -> list[gym.Env]:
    if wrappers is None:
        wrappers = []

    def make_env(env_id, wrap_list=None):
        if wrap_list is None:
            wrap_list = []

        def _thunk():
            env = gym.make(gym_id, **gym_kwargs)
            for w in wrap_list:
                env = w(env)
            env.reset(
                seed=env_seed + env_id,
            )  # Seeding only necessary for SB3 I think, doesn't hurt either way
            return env

        return _thunk

    # create the environment
    envs = []
    for i in range(num_threads):
        env_fns = [
            make_env(i * num_envs_per_thread + j, wrappers) for j in range(num_envs_per_thread)
        ]
        env = vecenv_cls(
            env_fns,
        )  # Eval seems to be faster with gym vector envs in preliminary tests
        envs.append(env)

    return envs


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        return super().default(obj)


class DSMCLogger:
    _logger = None
    _LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
    _DATE_FORMAT = "%H:%M:%S"

    class ColorfulFormatter(logging.Formatter):
        NC = "\033[0m"
        BOLD = "\033[1m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GRAY = "\033[90m"

        FORMATS: ClassVar = {
            logging.DEBUG: GRAY + "[%(asctime)s] [%(levelname)s] %(message)s" + NC,
            logging.INFO: CYAN + "[%(asctime)s] [%(levelname)s] %(message)s" + NC,
            logging.WARNING: YELLOW + BOLD + "[%(asctime)s] [%(levelname)s] %(message)s" + NC,
            logging.ERROR: RED + BOLD + "[%(asctime)s] [%(levelname)s] %(message)s" + NC,
            logging.CRITICAL: RED + BOLD + "[%(asctime)s] [%(levelname)s] %(message)s" + NC,
        }

        def format(self, record):
            log_fmt = DSMCLogger.ColorfulFormatter.FORMATS.get(
                record.levelno,
                DSMCLogger.ColorfulFormatter.GRAY
                + DSMCLogger._LOGGING_FORMAT
                + DSMCLogger.ColorfulFormatter.NC,
            )
            formatter = logging.Formatter(log_fmt, datefmt=DSMCLogger._DATE_FORMAT)
            return formatter.format(record)

    @staticmethod
    def get_logger():
        if DSMCLogger._logger is None:
            DSMCLogger._logger = logging.getLogger(__name__)
            DSMCLogger._logger.setLevel(logging.INFO)
            stdout_handler = logging.StreamHandler()

            DSMCLogger._logger.addHandler(stdout_handler)
        return DSMCLogger._logger

    @staticmethod
    def set_colorize(colorize: bool = False):
        new_formatter = (
            DSMCLogger.ColorfulFormatter()
            if colorize
            else logging.Formatter(DSMCLogger._LOGGING_FORMAT, datefmt=DSMCLogger._DATE_FORMAT)
        )
        DSMCLogger._logger.handlers[0].setFormatter(new_formatter)
