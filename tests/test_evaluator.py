import shutil

import gymnasium as gym
import numpy as np
import pgtg
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import pydsmc.property as prop
from pydsmc.evaluator import Evaluator
from pydsmc.utils import create_eval_envs


@pytest.fixture(scope='module')
def setup_module():
    PROP_LOG_DIR = 'test_prop_logs'
    shutil.rmtree(PROP_LOG_DIR, ignore_errors=True)

    envs = create_eval_envs(
        num_threads=1,
        num_envs_per_thread=1,
        env_seed=1,
        gym_id='pgtg-v3',
        wrappers=[gym.wrappers.FlattenObservation],
        vecenv_cls=gym.vector.SyncVectorEnv,
        max_episode_steps=100,
    )

    agent = DQN.load('example_agents/pgtg-v3/dqn_agent')
    property = prop.create_custom_property(
            name='goal_reaching_prob_binomial_abs_sound',
            binomial=True,
            sound=True,
            eps=0.1,
            kappa=0.25,
            relative_error=False,
            bounds=(0, 1),
            check_fn=lambda self, t: np.sum(np.fromiter((s[2] for s in t), dtype=np.float32)) >= self.goal_reward - 1e-8,
            goal_reward=100
        )

    data = {
        'PROP_LOG_DIR': PROP_LOG_DIR,

        'envs': envs,
        'agent': agent,
        'property': property,
    }

    yield data

    print('\nDeleting test evaluation results...')
    shutil.rmtree(PROP_LOG_DIR, ignore_errors=True)


def test_no_properties(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])

    with pytest.raises(ValueError, match='No properties registered'):
        evaluator.eval(agent=agent,
                predict_fn=agent.predict,
                episode_limit=None,
                save_every_n_episodes=100,
                num_initial_episodes=300,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )


def test_no_agent_predict(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])

    evaluator.register_property(setup_module['property'])

    evaluator.eval(predict_fn=agent.predict,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )

    evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )

    with pytest.raises(ValueError, match='No callable predict function or agent given'):
        evaluator.eval(agent=None,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )


def test_stable_baselines_vecenv(setup_module):
    envs = create_eval_envs(
        num_threads=1,
        num_envs_per_thread=1,
        env_seed=1,
        gym_id='pgtg-v3',
        wrappers=[gym.wrappers.FlattenObservation],
        vecenv_cls=DummyVecEnv,
        max_episode_steps=100,
    )

    agent = setup_module['agent']
    evaluator = Evaluator(env=envs, log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )



def test_single_env(setup_module):
    envs = create_eval_envs(
        num_threads=1,
        num_envs_per_thread=1,
        env_seed=1,
        gym_id='pgtg-v3',
        wrappers=[gym.wrappers.FlattenObservation],
        vecenv_cls=DummyVecEnv,
        max_episode_steps=100,
    )

    agent = setup_module['agent']

    vecenv_but_no_list = envs[0]
    neither_vecenv_nor_list = vecenv_but_no_list.envs[0]
    list_but_no_vecenv = [neither_vecenv_nor_list]

    # 1
    evaluator = Evaluator(env=vecenv_but_no_list, log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )

    # 2
    evaluator = Evaluator(env=neither_vecenv_nor_list, log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )

    # 3
    evaluator = Evaluator(env=list_but_no_vecenv, log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )




def test_not_enough_envs(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    with pytest.raises(ValueError, match='Number of environments must be at least the same as number of threads'):
        evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=2,
                deterministic=True
            )

    with pytest.raises(IndexError):
        evaluator = Evaluator(env=[], log_dir=setup_module['PROP_LOG_DIR'])


def test_no_stopping_criterion(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    with pytest.raises(ValueError, match='At least one stopping criterion must be set'):
        evaluator.eval(agent=agent,
                episode_limit=None,
                time_limit=None,
                stop_on_convergence=False,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                num_threads=1,
                deterministic=True
            )


def test_negative_timelimit(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    with pytest.raises(ValueError, match='Time limit must be positive'):
        evaluator.eval(agent=agent,
                episode_limit=None,
                time_limit=-1,
                stop_on_convergence=False,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                num_threads=1,
                deterministic=True
            )


def test_no_threads(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    with pytest.raises(ValueError, match='Number of threads must be at least 1'):
        evaluator.eval(agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=0,
                deterministic=True
            )


def test_policy_runs_per_iter(setup_module):
    agent = setup_module['agent']
    evaluator = Evaluator(env=setup_module['envs'], log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    with pytest.raises(ValueError, match='Number of initial episodes, and per policy run, must be at least 1'):
        evaluator.eval(
                agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=0,
                num_episodes_per_policy_run=50,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )

    with pytest.raises(ValueError, match='Number of initial episodes, and per policy run, must be at least 1'):
        evaluator.eval(
                agent=agent,
                episode_limit=100,
                save_every_n_episodes=100,
                num_initial_episodes=100,
                num_episodes_per_policy_run=0,
                save_full_results=False,
                stop_on_convergence=True,
                num_threads=1,
                deterministic=True
            )


@pytest.mark.parametrize(
    "vecenv_cls",
    [
        # Binomial Tests
        (gym.vector.SyncVectorEnv),
        (DummyVecEnv),
    ],
)
def test_empty_info(setup_module, vecenv_cls):
    class EmptyInfoWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)

        def step(self, action):
            a, r, tm, tr, i = self.env.step(action)
            return a, r, tm, tr, {}

    envs = create_eval_envs(
        num_threads=1,
        num_envs_per_thread=1,
        env_seed=1,
        gym_id='pgtg-v3',
        wrappers=[gym.wrappers.FlattenObservation, EmptyInfoWrapper],
        vecenv_cls=vecenv_cls,
        max_episode_steps=100,
    )

    agent = setup_module['agent']
    evaluator = Evaluator(env=envs, log_dir=setup_module['PROP_LOG_DIR'])
    evaluator.register_property(setup_module['property'])

    evaluator.eval(agent=agent,
            episode_limit=100,
            save_every_n_episodes=100,
            num_initial_episodes=100,
            num_episodes_per_policy_run=50,
            save_full_results=False,
            stop_on_convergence=True,
            num_threads=1,
            deterministic=True
        )
