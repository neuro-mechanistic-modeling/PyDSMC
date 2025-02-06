import numpy as np
import pytest

import pydsmc.property as prop


@pytest.fixture(scope='module')
def setup_module():
    SOUNDNESS_UNBOUND_ERROR_STR = 'Soundness only supported for bounded properties'
    FIXED_RELAIVE_ERROR_STR = 'Relative error requires epsilon to be set'

    data = {
        'SOUNDNESS_UNBOUND_ERROR_STR': SOUNDNESS_UNBOUND_ERROR_STR,
        'FIXED_RELAIVE_ERROR_STR': FIXED_RELAIVE_ERROR_STR,
    }

    yield data


@pytest.mark.parametrize(
    "lower, upper",
    [
        (None, None),
        (None, 0),
        (0, None),
        (-np.inf, np.inf),
        (-np.inf, 0),
        (0, np.inf),
        (None, np.inf),
        (np.inf, None),
    ],
)
def test_selection_sound_unbound(setup_module, lower, upper):
    with pytest.raises(ValueError, match=setup_module['SOUNDNESS_UNBOUND_ERROR_STR']):
        prop.select_statistical_method(sound=True, bounds=(lower, upper))

    with pytest.raises(ValueError, match=setup_module['SOUNDNESS_UNBOUND_ERROR_STR']):
        prop.create_custom_property(name='goal_reaching_prob_binomial_rel_sound',
                                    binomial=True,
                                    sound=True,
                                    eps=0.2,
                                    kappa=0.25,
                                    relative_error=True,
                                    bounds=(lower, upper),
                                    check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                    goal_reward=100)

    with pytest.raises(ValueError, match=setup_module['SOUNDNESS_UNBOUND_ERROR_STR']):
        prop.create_predefined_property(property_id='return',
                                        name='return_bounded_abs_unsound',
                                        sound=True,
                                        eps=5,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(lower, upper))

@pytest.mark.parametrize(
    "lower, upper",
    [
        (1, 0),
        (1, -np.inf),
        (np.inf, -np.inf),
        (1, 1),
        (0, 0),
    ],
)
def test_invalid_bounds(setup_module, lower, upper):
    with pytest.raises(ValueError, match='Invalid bounds'):
        prop.select_statistical_method(sound=True, bounds=(lower, upper))

    with pytest.raises(ValueError, match='Invalid bounds'):
        prop.create_custom_property(name='goal_reaching_prob_binomial_rel_sound',
                                    binomial=True,
                                    sound=True,
                                    eps=0.2,
                                    kappa=0.25,
                                    relative_error=True,
                                    bounds=(lower, upper),
                                    check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                    goal_reward=100)

    with pytest.raises(ValueError, match='Invalid bounds'):
        prop.create_predefined_property(property_id='return',
                                        name='return_bounded_abs_unsound',
                                        sound=True,
                                        eps=5,
                                        kappa=0.25,
                                        relative_error=False,
                                        bounds=(lower, upper))


def test_selection_fixed_relative(setup_module):
    # Exception thrown in statistical selection
    with pytest.raises(ValueError, match=setup_module['FIXED_RELAIVE_ERROR_STR']):
        prop.select_statistical_method(relative_error=True, eps=None)

    with pytest.raises(ValueError, match=setup_module['FIXED_RELAIVE_ERROR_STR']):
        prop.create_custom_property(name='goal_reaching_prob_binomial_rel_sound',
                                    binomial=True,
                                    sound=True,
                                    eps=None,
                                    kappa=0.25,
                                    relative_error=True,
                                    bounds=(0, 1),
                                    check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                    goal_reward=100)

    with pytest.raises(ValueError, match=setup_module['FIXED_RELAIVE_ERROR_STR']):
        prop.create_predefined_property(property_id='return',
                                        name='return_bounded_abs_unsound',
                                        sound=True,
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=True,
                                        bounds=(-100, 100))


    # Exception thrown outside of statistical selection, since a valid statistical method is passed
    st_method = prop.select_statistical_method(relative_error=True, eps=0.2)
    with pytest.raises(ValueError, match=setup_module['FIXED_RELAIVE_ERROR_STR']):
        prop.create_predefined_property(property_id='return',
                                        name='return_bounded_abs_unsound',
                                        sound=True,
                                        eps=None,
                                        kappa=0.25,
                                        relative_error=True,
                                        st_method=st_method,
                                        bounds=(-100, 100))

    with pytest.raises(ValueError, match=setup_module['FIXED_RELAIVE_ERROR_STR']):
        prop.create_custom_property(name='goal_reaching_prob_binomial_rel_sound',
                                    binomial=True,
                                    sound=True,
                                    eps=None,
                                    kappa=0.25,
                                    relative_error=True,
                                    bounds=(0, 1),
                                    st_method=st_method,
                                    check_fn=lambda self, t: np.sum(s[2] for s in t) >= self.goal_reward - 1e-8,
                                    goal_reward=100)


def test_selection_predefined_unknown(setup_module):
    with pytest.raises(ValueError, match='not found'):
        prop.create_predefined_property(property_id='unknown')
