
import random

import pytest

from pydsmc.statistics import *


@pytest.fixture(scope='module')
def setup_module():
    data = {
        'sample_binomial': lambda: 1.0 if random.random() < 0.8 else 0.0,
        'sample_uniform': lambda: random.uniform(5.0, 15.0),
        'sample_normal1': lambda: max(90.0, min(110.0, random.normalvariate(100.0, 2.0))),
        'sample_normal2': lambda: max(50.0, min(150.0, random.normalvariate(100.0, 2.0)))
    }
    yield data


def check_method_sequential(
    method: StatisticalMethod,
    sample,
    correct_samples = None,
    repeatable = False
):
    if repeatable:
        random.seed(368808248)
    i = 0
    while True:
        i += 1
        method.add_sample(sample())
        converged, interval = method.get_interval()
        if converged:
            assert correct_samples is None or correct_samples[0] <= i <= correct_samples[1], \
                f"Number of samples {i} not in correct range {correct_samples}"
            break
    return interval



def check_method_fixed(
    method: StatisticalMethod,
    sample,
    correct_eps = None,
    num_samples = 1000,
    repeatable = False,
):
    if repeatable:
        random.seed(368808248)
    for _ in range(num_samples):
        method.add_sample(sample())
    converged, interval = method.get_interval()
    assert interval is not None
    actual_eps = (interval[1] - interval[0]) / 2.0
    assert correct_eps is None or abs(actual_eps - correct_eps) < 5e-3 # must never fail as this is based on a deterministic precalculation
    return interval


# TODO: Student's t tests?


### --- BINOMIAL --- ###
def test_seq_WilsonScoreIntervalMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_sequential(
        method=WilsonScoreIntervalMethod(eps = 0.1, binomial = True),
        sample=sample,
        correct_samples=(106, 106),
        repeatable=True
    )

def test_seq_HoeffdingMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_sequential(
        method=HoeffdingMethod(eps=0.1, bounds = (0.0, 1.0)),
        sample=sample,
        correct_samples=(185, 185),
        repeatable=True
    )

def test_seq_EBStopMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_sequential(
        method=EBStopMethod(relative_error = True, eps = 0.1 / 0.8, bounds = (0.0, 1.0)),
        sample=sample,
        repeatable=True
    )


def test_fixed_WilsonScoreIntervalMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_fixed(
        method=WilsonScoreIntervalMethod(eps = 0.1, binomial = True),
        sample=sample,
        correct_eps=0.0250,
        repeatable=True
    )

def test_fixed_NormalIntervalMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_fixed(
        method=NormalIntervalMethod(eps = None),
        sample=sample,
        correct_eps=0.0245,
        repeatable=True
    )

def test_fixed_DKWMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_fixed(
        method=DKWMethod(eps = None, bounds = (0.0, 1.0)),
        sample=sample,
        correct_eps=0.042946940834673764,
        repeatable=True
    )

def test_fixed_HoeffdingMethod(setup_module):
    sample = setup_module['sample_binomial']
    check_method_fixed(
        method=HoeffdingMethod(eps = None, bounds = (0.0, 1.0)),
        sample=sample,
        correct_eps=0.042946940834673764,
        repeatable=True
    )


### --- UNIFORM --- ###
def test_seq_HoeffdingMethod_uniform(setup_module):
    sample = setup_module['sample_uniform']
    check_method_sequential(
        method=HoeffdingMethod(eps=0.1, bounds = (5.0, 15.0)),
        sample=sample,
        correct_samples=(18445, 18445),
        repeatable=True
    )

def test_seq_EBStopMethod_uniform(setup_module):
    sample = setup_module['sample_uniform']
    check_method_sequential(
        method=EBStopMethod(relative_error = True, eps = 0.1 / 10.0, bounds = (5.0, 15.0)),
        sample=sample,
        repeatable=True
    )

def test_fixed_NormalIntervalMethod_uniform(setup_module):
    sample = setup_module['sample_uniform']
    check_method_fixed(
        method=NormalIntervalMethod(eps = None),
        sample=sample,
        correct_eps=0.181,
        repeatable=True
    )

def test_fixed_DKWMethod_uniform(setup_module):
    sample = setup_module['sample_uniform']
    check_method_fixed(
        method=DKWMethod(eps = None, bounds = (5.0, 15.0)),
        sample=sample,
        correct_eps=0.418,
        repeatable=True
    )

def test_fixed_HoeffdingMethod_uniform(setup_module):
    sample = setup_module['sample_uniform']
    check_method_fixed(
        method=HoeffdingMethod(eps = None, bounds = (5.0, 15.0)),
        sample=sample,
        correct_eps=0.42946940834673697,
        repeatable=True
    )


### --- NORMAL1 --- ###
def test_seq_HoeffdingMethod_normal1(setup_module):
    sample = setup_module['sample_normal1']
    check_method_sequential(
        method=HoeffdingMethod(eps=0.1, bounds = (90.0, 110.0)),
        sample=sample,
        repeatable=True
    )

def test_seq_EBStopMethod_normal1(setup_module):
    sample = setup_module['sample_normal1']
    check_method_sequential(
        method=EBStopMethod(relative_error = True, eps = 0.1 / 100.0, bounds = (90.0, 110.0)),
        sample=sample,
        repeatable=True
    )

def test_fixed_NormalIntervalMethod_normal1(setup_module):
    sample = setup_module['sample_normal1']
    check_method_fixed(
        method=NormalIntervalMethod(eps = None),
        sample=sample,
        repeatable = True
    )

def test_fixed_DKWMethod_normal1(setup_module):
    sample = setup_module['sample_normal1']
    check_method_fixed(
        method=DKWMethod(eps = None, bounds = (90.0, 110.0)),
        sample=sample,
        repeatable=True
    )

def test_fixed_HoeffdingMethod_normal1(setup_module):
    sample = setup_module['sample_normal1']
    check_method_fixed(
        method=HoeffdingMethod(eps = None, bounds = (90.0, 110.0)),
        sample=sample,
        repeatable=True
    )

### --- NORMAL2 --- ###
def test_seq_HoeffdingMethod_normal2(setup_module):
    sample = setup_module['sample_normal2']
    check_method_sequential(
        method=HoeffdingMethod(eps=0.1, bounds = (50.0, 150.0)),
        sample=sample,
        repeatable=True
    )

def test_seq_EBStopMethod_normal2(setup_module):
    sample = setup_module['sample_normal2']
    check_method_sequential(
        method=EBStopMethod(relative_error = True, eps = 0.1 / 100.0, bounds = (50.0, 150.0)),
        sample=sample,
        repeatable=True
    )

def test_fixed_NormalIntervalMethod_normal2(setup_module):
    sample = setup_module['sample_normal2']
    check_method_fixed(
        method=NormalIntervalMethod(eps = None),
        sample=sample,
        repeatable=True
    )

def test_fixed_DKWMethod_normal2(setup_module):
    sample = setup_module['sample_normal2']
    check_method_fixed(
        method=DKWMethod(eps = None, bounds = (50.0, 150.0)),
        sample=sample,
        repeatable=True
    )

def test_fixed_HoeffdingMethod_normal2(setup_module):
    sample = setup_module['sample_normal2']
    check_method_fixed(
        method=HoeffdingMethod(eps = None, bounds = (50.0, 150.0)),
        sample=sample,
        repeatable=True
    )



### CREATED FROM THESE TESTS:

# # ...on binomial p=0.8
# sample = BINOMIAL
# # check_method_sequential(ChowRobbinsMethod(), sample, "p=0.8", "Chow-Robbins", (30, 100), repeatable = True)
# # check_method_sequential(HoeffdingMethod(bounds = (0.0, 1.0)), sample, "p=0.8", "Hoeffding   ", (185, 185), repeatable = True)
# print()
# check_method_sequential(EBStopMethod(relative_error = True, eps = 0.1 / 0.8, bounds = (0.0, 1.0)), sample, "p=0.8", "EBStop      ", repeatable = True)
# print()
# check_method_fixed(WilsonScoreIntervalMethod(binomial = True), sample, "p=0.8", "Wilson-CC   ", 0.0250, repeatable = True)
# # check_method_fixed(NormalIntervalMethod(eps = None), sample, "p=0.8", "Normal int. ", 0.0245, repeatable = True)
# check_method_fixed(DKWMethod(eps = None, bounds = (0.0, 1.0)), sample, "p=0.8", "DKW         ", 0.042946940834673764, repeatable = True)
# check_method_fixed(HoeffdingMethod(eps = None, bounds = (0.0, 1.0)), sample, "p=0.8", "Hoeffding   ", 0.042946940834673764, repeatable = True)
# print()

# ...on uniform over [5, 15]
# print("****")
# print()
# sample = UNIFORM
# check_method_sequential(ChowRobbinsMethod(), sample, "Uni(5, 15)", "Chow-Robbins", (3000, 3300), repeatable = True)
# check_method_sequential(HoeffdingMethod(bounds = (5.0, 15.0)), sample, "Uni(5, 15)", "Hoeffding   ", (18445, 18445), repeatable = True)
# print()
# check_method_sequential(EBStopMethod(relative_error = True, eps = 0.1 / 10.0, bounds = (5.0, 15.0)), sample, "Uni(5, 15)", "EBStop      ", repeatable = True)
# print()
# check_method_fixed(NormalIntervalMethod(eps = None), sample, "Uni(5, 15)", "Normal int. ", 0.181, repeatable = True)
# check_method_fixed(DKWMethod(eps = None, bounds = (5.0, 15.0)), sample, "Uni(5, 15)", "DKW         ", 0.418, repeatable = True)
# check_method_fixed(HoeffdingMethod(eps = None, bounds = (5.0, 15.0)), sample, "Uni(5, 15)", "Hoeffding   ", 0.42946940834673697, repeatable = True)
# print()

# # ...on normal(100.0, 2.0) ∩ [90, 110]
# print("****")
# print()
# sample = NORMAL1
# check_method_sequential(ChowRobbinsMethod(), sample, "Normal(100.0, 2.0) ∩ [90, 110]", "Chow-Robbins", repeatable = True)
# check_method_sequential(HoeffdingMethod(bounds = (90.0, 110.0)), sample, "Normal(100.0, 2.0) ∩ [90, 110]", "Hoeffding   ", repeatable = True)
# print()
# check_method_sequential(EBStopMethod(relative_error = True, eps = 0.1 / 100.0, bounds = (90.0, 110.0)), sample, "Normal(100.0, 2.0) ∩ [90, 110]", "EBStop      ", repeatable = True)
# print()
# check_method_fixed(NormalIntervalMethod(eps = None), sample, "Normal(100.0, 2.0) ∩ [90, 110]", "Normal int. ", repeatable = True)
# check_method_fixed(DKWMethod(eps = None, bounds = (90.0, 110.0)), sample, "Normal(100.0, 2.0) ∩ [90, 110]", "DKW         ", repeatable = True)
# check_method_fixed(HoeffdingMethod(eps = None, bounds = (90.0, 110.0)), sample, "Normal(100.0, 2.0) ∩ [90, 110]", "Hoeffding   ", repeatable = True)
# print()

# ...on normal(100.0, 2.0) ∩ [50, 150]
# print("****")
# print()
# sample = NORMAL2
# check_method_sequential(ChowRobbinsMethod(), sample, "Normal(100.0, 2.0) ∩ [50, 150]", "Chow-Robbins", repeatable = True)
# check_method_sequential(HoeffdingMethod(bounds = (50.0, 150.0)), sample, "Normal(100.0, 2.0) ∩ [50, 150]", "Hoeffding   ", repeatable = True)
# print()
# check_method_sequential(EBStopMethod(relative_error = True, eps = 0.1 / 100.0, bounds = (50.0, 150.0)), sample, "Normal(100.0, 2.0) ∩ [50, 150]", "EBStop      ", repeatable = True)
# print()
# check_method_fixed(NormalIntervalMethod(eps = None), sample, "Normal(100.0, 2.0) ∩ [50, 150]", "Normal int. ", repeatable = True)
# check_method_fixed(DKWMethod(eps = None, bounds = (50.0, 150.0)), sample, "Normal(100.0, 2.0) ∩ [50, 150]", "DKW         ", repeatable = True)
# check_method_fixed(HoeffdingMethod(eps = None, bounds = (50.0, 150.0)), sample, "Normal(100.0, 2.0) ∩ [50, 150]", "Hoeffding   ", repeatable = True)
# print()
