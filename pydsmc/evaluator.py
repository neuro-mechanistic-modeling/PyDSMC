from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import Env as GymEnv
from packaging.version import Version

from pydsmc.utils import DSMCLogger, NumpyEncoder

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from pydsmc.property import Property


# Main evaluator class
class Evaluator:
    def __init__(
        self,
        env: GymEnv | list[gym.vector.VectorEnv],
        log_dir: Path | str | os.PathLike | bytes = "logs",
        log_subdir: str = "eval",
        log_level: int = logging.INFO,
        *,
        colorize_logs: bool = False,
    ) -> None:
        self.envs = env
        self.log_base = Path(log_dir)
        self.log_base.mkdir(exist_ok=True, parents=True)
        self.log_subdir = log_subdir
        self.properties = []
        self.total_episodes = 0
        self.next_log_time = 0

        # For parallel episode execution
        self.lock = threading.Lock()
        self.thread_local = threading.local()
        self.next_free_env = 0

        self.logger = DSMCLogger.get_logger()
        DSMCLogger.set_colorize(colorize=colorize_logs)
        self.logger.setLevel(log_level)

        if not isinstance(env, list):
            # Single (vectorized) environment
            if hasattr(env, "num_envs") or hasattr(env, "n_envs"):
                self.envs = [env]  # Already vectorized, just not a list
            else:
                self.envs = [gym.vector.AsyncVectorEnv([lambda: env])]  # Not vectorized, nor list

        elif not (hasattr(env[0], "num_envs") or hasattr(env[0], "n_envs")):
            self.envs = [
                gym.vector.AsyncVectorEnv([lambda e=e: e]) for e in env
            ]  # List but not vectorized

        # earlier versions used n_envs. So we'd enforce a more recent version here otherwise
        # Support _some_ backwards compatibility at least
        if hasattr(self.envs[0], "n_envs"):
            for e in self.envs:
                e.num_envs = e.n_env

        if not hasattr(self.envs[0], "num_envs"):
            raise ValueError(
                "Environment must be a vectorized gymnasium or stable_baselines3 environment.",
            )

        self.gym_vecenv = isinstance(self.envs[0], gym.vector.VectorEnv)
        self.gym_version_lt_1 = Version(gym.__version__) < Version("1.0.0")

        if self.gym_vecenv and hasattr(self.envs[0], "autoreset_mode"):
            from gymnasium.vector.vector_env import AutoresetMode

            if any(env.autoreset_mode != AutoresetMode.NEXT_STEP for env in self.envs):
                raise ValueError("Only the default `NextStep` autoreset mode is supported.")

    @property
    def num_envs(self) -> int:
        return self.envs[0].num_envs

    def register_property(self, property_: Property) -> None:
        property_.set_property_id()
        self.properties.append(property_)

    def register_properties(self, properties: Iterable[Property]) -> None:
        for prop in properties:
            self.register_property(prop)

    def eval(
        self,
        agent: Any = None,  # Allow None, since agent is _ONLY_ necessary if predict_fn is None
        predict_fn: Callable | None = None,
        episode_limit: int | None = None,
        time_limit: float | None = None,
        num_initial_episodes: int = 100,
        num_episodes_per_policy_run: int = 50,
        save_every_n_episodes: int | None = None,
        *,
        stop_on_convergence: bool = True,
        save_full_results: bool = False,
        save_full_trajectory: bool = False,
        num_threads: int = 1,
        extra_log_info: Any = None,
        **predict_kwargs,
    ) -> list[Property]:
        if num_initial_episodes < 1 or num_episodes_per_policy_run < 1:
            raise ValueError("Number of initial episodes, and per policy run, must be at least 1")
        if num_threads < 1:
            raise ValueError("Number of threads must be at least 1")
        if (
            episode_limit is None
            and time_limit is None
            and not (stop_on_convergence and all(prop.epsilon is not None for prop in self.properties))
        ):
            raise ValueError(
                "At least one stopping criterion must be set: episode_limit, time_limit, or stop_on_convergence. "
                "If only stop_on_convergence is set, all properties must have an epsilon value set.",
            )

        if save_full_trajectory:
            self.logger.warning(
                "SAVING FULL TRAJECTORIES ENABLED. "
                "This is usually not recommended as it will slow down evaluation as well as consume a lot of disk space.",
            )

        eval_params = predict_kwargs | {
            "num_initial_episodes": num_initial_episodes,
            "episode_limit": episode_limit,
            "time_limit": time_limit,
        }

        predict_fn = self.__setup_eval(
            agent=agent,
            predict_fn=predict_fn,
            num_episodes_per_policy_run=num_episodes_per_policy_run,
            save_every_n_episodes=save_every_n_episodes,
            save_full_results=save_full_results,
            eval_params=eval_params,
            num_threads=num_threads,
            extra_log_info=extra_log_info,
        )

        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive.")
            # convert time limit from minutes to seconds, time limit is a float, 2.5 hours are 2 hours and 30 minutes
            time_limit_seconds = time_limit * 60

        if save_full_results:
            stop_event = threading.Event()
            threading.Thread(
                target=Evaluator.__save_full_results_daemon,
                args=(stop_event, self.properties),
                daemon=True,
            ).start()

        ### run the policy until all properties have converged
        # So sadly, a ProcessPoolExecutor does not work here because
        # (1) VecEnvs are not picklable which could be circumvented
        # (2) The predict_fn is not picklable, which _is not_ circumventable
        # Therefore, still with ThreadPoolExecutor, and suggest the user to use AsyncVectorEnv or SubprocVecEnv
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.perf_counter()

            eval_string = "Starting evaluation"
            if episode_limit is not None and time_limit is not None:
                eval_string += f" with episode limit of {episode_limit} episodes and time limit of {time_limit} minutes"
            elif episode_limit is not None:
                eval_string += f" with episode limit of {episode_limit} episodes"
            elif time_limit is not None:
                eval_string += f" with time limit of {time_limit} minutes"
            self.logger.info(eval_string)

            self.logger.info("The agent will be evaluated according to the following properties:")
            for property_ in self.properties:
                property_string = (
                    f"\t{property_.name} using {property_.st_method.__class__.__name__}"
                )
                if property_.epsilon is None:
                    property_string += f" in the fixed run setting with min_samples={property_.st_method.min_samples}"
                else:
                    property_string += f" in the sequential setting with epsilon ={property_.epsilon }"
                self.logger.info(property_string)

            while True:
                self.__run_policy(
                    predict_fn=predict_fn,
                    executor=executor,
                    num_episodes=(
                        num_initial_episodes
                        if self.total_episodes == 0
                        else num_episodes_per_policy_run
                    ),
                    num_threads=num_threads,
                    save_full_trajectory=save_full_trajectory,
                    **predict_kwargs,
                )

                time_passed = time.perf_counter() - start_time
                if stop_on_convergence and all(prop.converged() for prop in self.properties):
                    self.logger.info("All properties converged!")
                    break

                if (time_limit is not None) and (time_passed >= time_limit_seconds):
                    self.logger.info("Time limit reached!")
                    break

                if (episode_limit is not None) and (self.total_episodes >= episode_limit):
                    self.logger.info("Episode limit reached!")
                    break

                if save_every_n_episodes and self.total_episodes >= self.next_log_time:
                    overwrite = self.next_log_time == save_every_n_episodes
                    for property_ in self.properties:
                        property_.save_results(overwrite=overwrite, logging_fn=self.logger.debug)

                    save_path = self.log_dir / "resources.jsonl"
                    with save_path.open("w" if overwrite else "a") as f:
                        f.write(
                            json.dumps(
                                {"total_episodes": self.total_episodes, "time_passed": time_passed},
                            )
                            + "\n",
                        )

                    self.next_log_time = self.total_episodes + save_every_n_episodes

        # Save resources at the end again
        save_path = self.log_dir / "resources.jsonl"
        with save_path.open("a") as f:
            f.write(
                json.dumps({"total_episodes": self.total_episodes, "time_passed": time_passed})
                + "\n",
            )

        hours, rem = divmod(time.perf_counter() - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        self.logger.info(
            f"Evaluation finished after {self.total_episodes} episodes, which took"  # noqa: G004
            f" {f'{hours:.0f} hours, ' if hours > 0 else ''}"
            f" {f'{minutes:.0f} minutes, ' if minutes > 0 else ''}"
            f" {seconds:.0f} seconds.",
        )

        if save_full_results:
            stop_event.set()

        self.__end_eval(save_full_results)
        return self.properties

    def clear_properties(self) -> None:
        self.properties = []

    def set_log_dir(self, log_dir: Path | str | os.PathLike | bytes = "logs") -> None:
        self.log_base = Path(log_dir)
        self.log_base.mkdir(exist_ok=True, parents=True)

    def __get_thread_env(self) -> bool:
        try:
            return self.thread_local.env
        except AttributeError:
            with self.lock:
                self.thread_local.env = self.envs[self.next_free_env]
                self.next_free_env += 1
        return self.thread_local.env

    def __run_episodes(
        self,
        predict_fn: Callable,
        num_episodes: int,
        *,
        save_full_trajectory: bool,
        **predict_kwargs,
    ) -> None:
        env = self.__get_thread_env()

        # Distribute episodes evenly to available parallel environments
        num_episodes_per_venv = np.array(
            [(num_episodes + i) // self.num_envs for i in range(self.num_envs)],
            dtype="int",
        )
        episodes_done_per_venv = np.zeros(self.num_envs, dtype="int")
        episode_starts = np.zeros(self.num_envs, dtype="bool")

        reset_data = env.reset()
        if self.gym_vecenv:
            state, info = reset_data
        else:
            state = reset_data
            info = env.reset_infos

        trajectories = [[[] for _ in range(num_episodes)] for num_episodes in num_episodes_per_venv]
        while (episodes_done_per_venv < num_episodes_per_venv).any():
            actions, states = predict_fn(state, **predict_kwargs)
            step_data = env.step(actions)
            processed_infos = [{} for _ in range(self.num_envs)]

            if self.gym_vecenv:
                next_states, rewards, terminateds, truncateds, infos = step_data

                # Remove RecordEpisodeStatisticsWrapper's episode info, since we don't need it (gym 1.0.0)
                infos.pop("episode", None)
                infos.pop("_episode", None)

                # Invert dictionary and list order
                if len(infos) > 0:
                    for key, array in infos.items():
                        for i in range(self.num_envs):
                            processed_infos[i][key] = array[i]

            else:
                # SB3 VecEnv [sadly still merges terminated and truncated]
                next_states, rewards, dones, infos = step_data
                if len(infos) > 0:
                    processed_infos = infos
                    truncateds = [infos[i]["TimeLimit.truncated"] for i in range(self.num_envs)]
                    terminateds = [(a and not b) for a, b in zip(dones, truncateds)]
                else:
                    terminateds = dones
                    truncateds = [False for _ in range(self.num_envs)]

            for i in range(self.num_envs):
                if episodes_done_per_venv[i] >= num_episodes_per_venv[i]:
                    continue

                reward = rewards[i]
                action = actions[i]
                terminated = terminateds[i]
                truncated = truncateds[i]
                info = processed_infos[i]
                done = terminated or truncated

                if self.gym_vecenv:  # Gymnasium VecEnv
                    if self.gym_version_lt_1:  # Autoreset_Mode == SameStep
                        if done:
                            info = info["final_info"]
                        elif "_final_observation" in info:  # Remove data that we dont need
                            info.pop("_final_observation")
                            info.pop("_final_info")
                            info.pop("final_observation")
                            info.pop("final_info")
                        trajectories[i][episodes_done_per_venv[i]].append(
                            (state[i], action, reward, terminated, truncated, info),
                        )

                    else:  # Autoreset_Mode == NextStep
                        if not episode_starts[i]:
                            trajectories[i][episodes_done_per_venv[i]].append(
                                (state[i], action, reward, terminated, truncated, info),
                            )
                        episode_starts[i] = done
                    # TODO: We could support SAME_STEP as well.
                    #       This stores the final observation in "final_obs", should be handled similar to first case

                else:  # SB3 VecEnv, Autoreset_Mode == SameStep
                    if "terminal_observation" in info:  # Remove data that we dont need
                        info.pop("terminal_observation")
                    trajectories[i][episodes_done_per_venv[i]].append(
                        (state[i], action, reward, terminated, truncated, info),
                    )

                if done:
                    episodes_done_per_venv[i] += 1

            state = next_states

        trajectories = [t for trajectory in trajectories for t in trajectory]

        assert len(trajectories) == num_episodes, (
            f"Expected {num_episodes} trajectories, got {len(trajectories)}"
        )

        for trajectory in trajectories:
            for property_ in self.properties:
                prop_check = property_.check(trajectory)
                property_.add_sample(prop_check)

        if save_full_trajectory:
            with (self.log_dir / "trajectories.jsonl").open("a") as f:
                for trajectory in trajectories:
                    f.write(json.dumps(trajectory, cls=NumpyEncoder) + "\n")

        self.total_episodes += int(num_episodes)

    def __run_policy(
        self,
        predict_fn: Callable,
        executor: Executor,
        num_episodes: int = 50,
        num_threads: int = 1,
        *,
        save_full_trajectory: bool = False,
        **predict_kwargs,
    ):
        # Distribute episodes evenly to available threads
        num_episodes_per_thread = np.array(
            [(num_episodes + i) // num_threads for i in range(num_threads)],
            dtype="int",
        )
        num_episodes_before = [0, *np.cumsum(num_episodes_per_thread).tolist()]

        # Temporarily store the results in numpy arrays of fixed size
        futures = {
            executor.submit(
                self.__run_episodes,
                predict_fn=predict_fn,
                num_episodes=num_episodes,
                save_full_trajectory=save_full_trajectory,
                **predict_kwargs,
            ): (num_before, num_episodes)
            for num_before, num_episodes in zip(num_episodes_before, num_episodes_per_thread)
        }
        for future in as_completed(futures):
            _result = future.result()  # Wait for completion

    def __save_eval_params(self, eval_settings: dict):
        save_path = self.log_dir / "settings.json"
        with save_path.open("w") as f:
            json.dump(eval_settings, f, indent=4, cls=NumpyEncoder)
        self.logger.info(f"Evaluation settings saved to {save_path}")

    def __setup_eval(
        self,
        agent: Any,
        predict_fn: Callable | None,
        num_episodes_per_policy_run: int,
        save_every_n_episodes: int | None,
        eval_params: dict[str, Any],
        num_threads: int,
        *,
        extra_log_info: Any = None,
        save_full_results: bool,
    ):
        if len(self.envs) < num_threads:
            raise ValueError(
                f"Number of environments must be at least the same as number of threads. Envs: {len(self.envs)}, threads: {num_threads}. "
                f"There is a helper function to create environments in the correct format in `pydsmc.utils` called `create_eval_envs`.",
            )

        if predict_fn is None and agent is not None:
            predict_fn = agent.predict

        if not callable(predict_fn):
            raise TypeError("No callable predict function or agent given.")

        if len(self.properties) == 0:
            raise ValueError(
                "No properties registered. Use `register_property` to register properties to evaluate.",
            )

        self.log_dir = (
            self.log_base
            / f"{self.log_subdir}_{Evaluator.__get_next_run_id(self.log_base, self.log_subdir)}"
        )

        for property_ in self.properties:
            property_.setup_eval(self.log_dir, save_full_results=save_full_results)
            property_.save_settings(self.log_dir)

        if self.gym_version_lt_1:
            self.logger.warning(
                "gymnasium before version 1.0.0 does not allow to inspect the environments' seeds. Storing 'None' instead.",
            )
            env_seeds = [[None] * vec_env.num_envs for vec_env in self.envs]
        elif self.gym_vecenv:
            env_seeds = [vec_env.np_random_seed for vec_env in self.envs]
        else:
            env_seeds = [[e.np_random_seed for e in vec_env.envs] for vec_env in self.envs]

        eval_params = eval_params | {
            "num_episodes_per_policy_run": num_episodes_per_policy_run,
            "num_threads": num_threads,
            "env_seeds": env_seeds,
            "property_ids": [property_.property_id for property_ in self.properties],
            "extra_log_info": extra_log_info,
        }

        self.__save_eval_params(eval_params)

        self.total_episodes = 0
        self.next_free_env = 0
        self.next_log_time = (
            num_episodes_per_policy_run if save_every_n_episodes is None else save_every_n_episodes
        )

        return predict_fn

    def __end_eval(
        self,
        save_full_results: bool,
    ) -> None:
        Evaluator.__save_full_results_daemon(save_full_results, self.properties)
        for property_ in self.properties:
            property_.save_results(logging_fn=self.logger.info)

    @staticmethod
    def __get_next_run_id(log_path: Path, log_subdir: str = "") -> int:
        """
        Inspired from stable_baselines3.common.utils > get_latest_run_id.
        """
        max_run_id = -1
        for path in log_path.glob(f"{log_subdir}_[0-9]*"):
            file_name = Path(path).name
            ext = file_name.split("_")[-1]
            if (
                log_subdir == "_".join(file_name.split("_")[:-1])
                and ext.isdigit()
                and int(ext) > max_run_id
            ):
                max_run_id = int(ext)
        return max_run_id + 1

    @staticmethod
    def __save_full_results_daemon(save_full_results, properties, stop_event=None) -> None:
        if not save_full_results:
            return

        if stop_event is None:
            for p in properties:
                p.dump_buffer(overwrite=False)

        else:
            first = True
            while not stop_event.is_set():
                for p in properties:
                    p.dump_buffer(overwrite=first)

                first = False
                time.sleep(60)
