
import math
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import pgtg
import pytest
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import pydsmc.property as prop
from pydsmc.evaluator import Evaluator
from pydsmc.json_translator import jsons_to_df
from pydsmc.statistics import *
from pydsmc.utils import create_eval_envs


@pytest.fixture(scope='module')
def setup_module():
    FIXED_LOG_DIR = Path('./tests_fixed_logs')
    SEQ_LOG_DIR = Path('./tests_seq_logs')

    NUM_THREADS = 5
    NUM_PAR_ENVS = 5
    SEED = 42
    TRUNCATE_LIMIT = 100

    # Intialize evaluation environment
    envs = create_eval_envs(
        num_threads=NUM_THREADS,
        num_envs_per_thread=NUM_PAR_ENVS,
        env_seed=SEED,
        gym_id='pgtg-v3',
        wrappers=[FlattenObservation, Monitor],
        vecenv_cls=gym.vector.SyncVectorEnv,
        max_episode_steps=TRUNCATE_LIMIT,
    )

    # Setup properties
    properties: list[prop.Property] = {}

    properties['binomial_abs_sound'] = prop.create_custom_property(
                                        name='goal_reaching_prob_binomial_abs_sound',
                                        binomial=True,
                                        sound=True,
                                        eps=0.1,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                        goal_reward=100)
    properties['binomial_rel_sound'] = prop.create_custom_property(
                                        name='goal_reaching_prob_binomial_rel_sound',
                                        binomial=True,
                                        sound=True,
                                        eps=0.2,
                                        kappa=0.25,
                                        relative_error=True,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                        goal_reward=100)
    properties['binomial_abs_unsound'] = prop.create_custom_property(
                                        name='goal_reaching_prob_binomial_abs_unsound',
                                        binomial=True,
                                        sound=False,
                                        eps=0.1,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                        goal_reward=100)
    properties['binomial_rel_unsound'] = prop.create_custom_property(
                                        name='goal_reaching_prob_binomial_rel_unsound',
                                        binomial=True,
                                        sound=False,
                                        eps=0.2,
                                        kappa=0.25,
                                        relative_error=True,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                        goal_reward=100)

    properties['bounded_abs_sound'] = prop.create_predefined_property(
                                        property_id='return',
                                        name='return_bounded_abs_sound',
                                        sound=True,
                                        eps=5,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(-100, 100))
    properties['bounded_rel_sound'] = prop.create_predefined_property(
                                        property_id='return',
                                        name='return_bounded_rel_sound',
                                        sound=True,
                                        eps=0.2,
                                        kappa=0.25,
                                        relative_error=True,
                                        bounds=(-100, 100))
    properties['bounded_abs_unsound'] = prop.create_predefined_property(
                                        property_id='return',
                                        name='return_bounded_abs_unsound',
                                        sound=False,
                                        eps=5,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(-100, 100))
    properties['bounded_rel_unsound'] = prop.create_predefined_property(
                                        property_id='return',
                                        name='return_bounded_rel_unsound',
                                        sound=False,
                                        eps=0.2,
                                        kappa=0.25,
                                        relative_error=True,
                                        bounds=(-100, 100))

    properties['unbounded_rel'] = prop.create_predefined_property( # one-sided; effectively bounded by truncation
                                        property_id='episode_length',
                                        name='episode_length_unbounded_rel',
                                        eps=0.2,
                                        kappa=0.25,
                                        relative_error=True,
                                        bounds=(0, np.inf))
    properties['unbounded_abs'] = prop.create_predefined_property( # one-sided; effectively bounded by truncation
                                        property_id='episode_length',
                                        name='episode_length_unbounded_abs',
                                        eps=5,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(0, np.inf))

    properties['fixed_binomial_sound'] = prop.create_custom_property(
                                        name='goal_reaching_prob_fixed_binomial_sound',
                                        binomial=True,
                                        sound=True,
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                        goal_reward=100)
    properties['fixed_binomial_unsound'] = prop.create_custom_property(
                                        name='goal_reaching_prob_fixed_binomial_unsound',
                                        binomial=True,
                                        sound=False,
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(0, 1),
                                        check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                        goal_reward=100)


    properties['fixed_bounded_sound'] = prop.create_predefined_property(
                                        property_id='return',
                                        name='return_fixed_bounded_sound',
                                        sound=True,
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(-100, 100))
    properties['fixed_bounded_unsound'] = prop.create_predefined_property(
                                        property_id='return',
                                        name='return_fixed_bounded_unsound',
                                        sound=False,
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(-100, 100))

    properties['fixed_unbounded'] = prop.create_predefined_property( # one-sided; effectively bounded by truncation
                                        property_id='episode_length',
                                        name='episode_length_fixed_unbounded',
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(0, np.inf))

    for property in properties.values():
        assert (property.eps is None and 'fixed' in property.name) or \
               (property.eps is not None and 'fixed' not in property.name), \
                'There is some error in naming or property value assignment.'

    # initialize the evaluator
    shutil.rmtree(FIXED_LOG_DIR, ignore_errors=True)
    evaluator = Evaluator(env=envs, log_dir=FIXED_LOG_DIR)
    agent = DQN.load('example_agents/pgtg-v3/dqn_agent')

    # Run evaluator with fixed properties
    for property in properties.values():
        if 'fixed' in property.name:
            evaluator.register_property(property)

    evaluator.eval(agent=agent,
                predict_fn=agent.predict,
                episode_limit=2000,
                save_every_n_episodes=100,
                num_initial_episodes=300,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=False,
                num_threads=NUM_THREADS,
                deterministic=True)

    # Set up evaluator for sequential runs
    evaluator.clear_properties()
    shutil.rmtree(SEQ_LOG_DIR, ignore_errors=True)
    evaluator.set_log_dir(SEQ_LOG_DIR)

    # Run evaluator with sequential properties
    for property in properties.values():
        if 'fixed' not in property.name:
            evaluator.register_property(property)

    evaluator.eval(agent=agent,
                predict_fn=agent.predict,
                episode_limit=None,
                save_every_n_episodes=100,
                num_initial_episodes=300,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=NUM_THREADS,
                deterministic=True)


    fixed_df = jsons_to_df(log_dir=FIXED_LOG_DIR / 'eval_0', include_timeseries=True, save=False)
    seq_df = jsons_to_df(log_dir=SEQ_LOG_DIR / 'eval_0', include_timeseries=True, save=False)

    # Perform common setup tasks
    data = { # Shared resource
        'evaluator': evaluator,
        'agent': agent,
        'properties': properties,
        'fixed_df': fixed_df,
        'seq_df': seq_df
    }
    yield data
    # Perform teardown tasks after all tests in the file

    for env in envs:
        env.close()

    print('\nDeleting test evaluation results...')
    shutil.rmtree(SEQ_LOG_DIR / 'eval_0', ignore_errors=True)
    shutil.rmtree(FIXED_LOG_DIR / 'eval_0', ignore_errors=True)


def assert_convergence(
    property: prop.Property,
    v_total_episodes: int | None = None,
    v_mean: float | None = None,
    v_variance: float | None = None,
    v_std: float | None = None,
    v_intv: tuple[float, float] | None = None,
):
    REL_TOLERANCE = 1e-1
    converged, intv = property.get_interval()
    assert converged, 'Property should have converged'
    assert intv is not None, 'A converged interval cannot be None'

    assert v_total_episodes is None or math.isclose(property.num_episodes, v_total_episodes, rel_tol=REL_TOLERANCE)
    assert v_mean is None or math.isclose(property.mean, v_mean, rel_tol=REL_TOLERANCE)
    assert v_variance is None or math.isclose(property.variance, v_variance, rel_tol=REL_TOLERANCE)
    assert v_std is None or math.isclose(property.std, v_std, rel_tol=REL_TOLERANCE)
    assert v_intv is None or math.isclose(intv[0], v_intv[0], rel_tol=REL_TOLERANCE) and \
              math.isclose(intv[1], v_intv[1], rel_tol=REL_TOLERANCE)

    intv_size = intv[1] - intv[0]
    assert intv_size >= 0, f'Upper interval bound should be greater than lower bound; Interval: ({float(intv[0]):3f},{float(intv[1]):3f})'
    if property.eps is not None:
        eps = property.eps * (property.mean if property.relative_error else 1)
        assert intv_size <= 2 * eps, f'Interval is too large; len(({float(intv[0]):3f},{float(intv[1]):3f})) = {intv_size:3f} > {2*eps:3f}'


def assert_statistics_selection(property, st_method_cls):
    assert type(property.st_method) == st_method_cls, \
        'Wrong automatic statistical method selection for given parameters. '\
        f'Expected: {st_method_cls}, Actual: {type(property.st_method)}'



### --- BINOMIAL --- ###
@pytest.mark.parametrize(
    "test_name, method, total_episodes, mean, variance, std, intv",
    [
        # Binomial Tests
        ("binomial_abs_sound",   WilsonScoreIntervalMethod, 1700, 0.836470588235294, 0.13678754325259496, 0.36984800020088654, (0.8259106504626834, 0.8471428557612839)),
        ("binomial_rel_sound",   EBStopMethod,              1700, 0.836470588235294, 0.13678754325259496, 0.36984800020088654, (0.6835047683613917, 1.0252571525420875)),
        ("binomial_abs_unsound", NormalIntervalMethod,      1700, 0.836470588235294, 0.13678754325259496, 0.36984800020088654, (0.8261482740776852, 0.8467929023929027)),
        ("binomial_rel_unsound", NormalIntervalMethod,      1700, 0.836470588235294, 0.13678754325259496, 0.36984800020088654, (0.8261482740776852, 0.8467929023929027)),

        # Bounded Tests
        ("bounded_abs_sound",   HoeffdingMethod, 1700, 71.98752776044076, 1443.2400932417527, 37.98999991105229, (66.98752776044076, 76.98752776044076)),
        ("bounded_rel_sound",   EBStopMethod,    1700, 71.98752776044076, 1443.2400932417527, 37.98999991105229, (58.16942104253408, 87.25413156380111)),
        ("bounded_abs_unsound", StudentsTMethod, 1700, 71.98752776044076, 1443.2400932417527, 37.98999991105229, (70.92724160581396, 73.04781391506755)),
        ("bounded_rel_unsound", StudentsTMethod, 1700, 71.98752776044076, 1443.2400932417527, 37.98999991105229, (70.92724160581396, 73.04781391506755)),

        # Unbounded Tests
        ("unbounded_abs", StudentsTMethod, 1700, 31.212352941176462, 349.72988096885786, 20.331898157795365, (30.644897553851735, 31.77980832850119)),
        ("unbounded_rel", StudentsTMethod, 1700, 31.212352941176462, 349.72988096885786, 20.331898157795365, (30.644897553851735, 31.77980832850119)),
    ],
)
def test_seq_properties(
    setup_module,
    test_name,
    method,
    total_episodes,
    mean,
    variance,
    std,
    intv
):
    evaluator = setup_module["evaluator"]
    agent = setup_module["agent"]
    properties: dict[str, prop.Property] = setup_module["properties"]

    tested_property = properties[test_name]
    assert_statistics_selection(tested_property, method)
    assert_convergence(
        property=tested_property,
        v_total_episodes=total_episodes,
        v_mean=mean,
        v_variance=variance,
        v_std=std,
        v_intv=intv
    )


### --- FIXED RUNS --- ###
@pytest.mark.parametrize(
    "test_name, method, total_episodes, mean, variance, std, intv",
    [
        # Fixed Binomial Tests
        ("fixed_binomial_sound",   WilsonScoreIntervalMethod, 2000, 0.8495, 0.12784975, 0.3575608339849318, (0.8400932839491071, 0.8589937026426394)),
        ("fixed_binomial_unsound", NormalIntervalMethod,      2000, 0.8495, 0.12784975, 0.3575608339849318, (0.8400932839491071, 0.8589937026426394)),

        # Fixed Bounded Tests
        ("fixed_bounded_sound",   DKWMethod,       2000, 74.12348035016905, 1184.4999046689964, 34.41656439374791, (69.80995029922501, 77.56629654990643)),
        ("fixed_bounded_unsound", StudentsTMethod, 2000, 74.12348035016905, 1184.4999046689964, 34.41656439374791, (73.23793973076012, 75.00902096957797)),

        # Fixed Unbounded Tests
        ("fixed_unbounded", StudentsTMethod, 2000, 31.688, 365.391096, 20.391607489357, (31.163322506910344, 32.21267749308958)),
    ],
)
def test_fixed_properties(
    setup_module,
    test_name,
    method,
    total_episodes,
    mean,
    variance,
    std,
    intv
):
    evaluator = setup_module["evaluator"]
    agent = setup_module["agent"]
    properties: dict[str, prop.Property] = setup_module["properties"]

    tested_property = properties[test_name]
    assert_statistics_selection(tested_property, method)
    assert_convergence(
        property=tested_property,
        v_total_episodes=total_episodes,
        v_mean=mean,
        v_variance=variance,
        v_std=std,
        v_intv=intv
    )
