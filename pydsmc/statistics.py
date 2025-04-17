from __future__ import annotations

import abc
import bisect

import numpy as np
import numpy.typing
from scipy.stats import beta, norm, t

from pydsmc.utils import DSMCLogger, is_bounded

# Default methods:
# sequential
# + binomial and sound:
#   + absolute error:     ClopperPearsonIntervalMethod
#   + relative error:     EBStopMethod
# + binomial and unsound:
#   + absolute error:     NormalMethod (a.k.a. Chow-Robbins)
#   + relative error:     NormalMethod (a.k.a. Chow-Robbins)
# + bounded and sound:
#   + absolute error:     HoeffdingMethod (may need a large number of samples)
#   + relative error:     EBStopMethod
# + unbounded or unsound:
#   + absolute error:     StudentsTMethod
#   + relative error:     StudentsTMethod
# fixed runs
# + binomial:             ClopperPearsonIntervalMethod
# + bounded and sound:    DKWMethod
# + unbounded or unsound: StudentsTMethod


class StatisticalMethod:
    def __init__(
        self,
        name: str,
        epsilon: float | None = 0.1,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float, float] = (None, None),
        binomial: bool = False,
        min_samples: int = 2,
    ):
        self.name = name
        self.epsilon = epsilon
        self.kappa = kappa
        self.relative_error = relative_error
        self.a = 0.0 if binomial else -numpy.inf if bounds[0] is None else bounds[0]
        self.b = 1.0 if binomial else numpy.inf if bounds[1] is None else bounds[1]
        self.binomial = binomial
        self.min_samples = min_samples

        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.variance = 0.0
        self.stddev = 0.0

    def _check_intv_2epsilon(self, intv: tuple[float, float]) -> float:
        if self.relative_error:
            center = 0.5 * abs(intv[0] + intv[1])
            return intv[1] - intv[0] <= 2 * self.epsilon * center
        return intv[1] - intv[0] <= 2 * self.epsilon

    def add_sample(self, x: float):
        assert not numpy.isinf(x) and self.a <= x <= self.b, (
            f"Sample {x} out of bounds [{self.a}, {self.b}]"
        )
        assert not self.binomial or x in {0.0, 1.0}
        self.count += 1

        # Welford's algorithm for mean and variance
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2
        self.variance = self._m2 / self.count
        self.stddev = numpy.sqrt(self.variance)

    @abc.abstractmethod
    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        return False, None


# The Clopper-Pearson interval
#
# Applies to binomial proportions, i.e. to samples from a Bernoulli distribution.
#
# Given the samples collected so far (via add_sample),
# optionally the desired absolute interval half-width (epsilon),
# and one minus the desired confidence level (kappa),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# If the half-width is omitted, then whenever there are at least min_samples samples, the return value will indicate that no more samples are needed,
# and provide an interval based on the provided samples (fixed number of runs setting).
#
# This method is sound.
#
class ClopperPearsonIntervalMethod(StatisticalMethod):
    def __init__(
        self,
        epsilon: float | None = None,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 2,
    ):
        super().__init__(
            "Clopper-Pearson interval",
            epsilon,
            kappa,
            relative_error,
            bounds,
            binomial,
            min_samples,
        )

        # Only for binomial proportions, and requires absolute error in the sequential setting
        if not binomial:
            raise Exception(f"The {self.name} only applies to binomial proportions.")

        if epsilon is not None and relative_error:
            raise Exception(f"The {self.name} only supports absolute interval half-width.")

        self.worst_case_samples = (
            min_samples  # for fixed runs, we are confident from min_samples onward
        )

        # Sequential setting: half-width is given, pre-calculate number of samples needed based on worst-case of p=0.5
        # via exponential and binary search for smallest interval of half-width at least epsilon assuming p=0.5
        if self.epsilon is not None:

            def need_more_runs(n):
                (l, u) = self._get_interval(0.5, n)[1]
                return (u - l) * 0.5 > self.epsilon

            upper_runs = 2
            while need_more_runs(upper_runs):
                upper_runs *= 2
            lower_runs = upper_runs // 2
            while lower_runs + 1 < upper_runs:
                runs = (lower_runs + upper_runs) // 2
                if need_more_runs(runs):
                    lower_runs = runs  # interval too wide, need more runs
                else:
                    upper_runs = runs  # interval small enough, but can perhaps do with fewer runs
                assert not need_more_runs(upper_runs)  # loop invariant
            self.worst_case_samples = upper_runs

    def _get_interval(self, mean, count):
        lower = 0.0
        upper = 1.0
        confidence = 1.0 - self.kappa
        if mean == 0.0:
            upper = 1.0 - (0.5 - 0.5 * confidence) ** (1.0 / count)
        elif mean == 1.0:
            lower = (0.5 - 0.5 * confidence) ** (1.0 / count)
        else:
            successes = mean * count
            lower = beta.ppf(0.5 - 0.5 * confidence, successes, count - successes + 1)
            upper = beta.ppf(0.5 + 0.5 * confidence, successes + 1, count - successes)
        return self.count >= self.worst_case_samples, (lower, upper)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Check minimum number of samples
        if self.count < self.min_samples:
            return False, None

        # Return interval
        return self._get_interval(self.mean, self.count)


# The Wilson score interval with continuity correction
#
# Applies to binomial proportions, i.e. to samples from a Bernoulli distribution.
#
# Given the samples collected so far (via add_sample),
# optionally the desired absolute interval half-width (epsilon),
# and one minus the desired confidence level (kappa),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# If the half-width is omitted, then whenever there are at least min_samples samples, the return value will indicate that no more samples are needed,
# and provide an interval based on the provided samples (fixed number of runs setting).
#
# This method is not sound: The requested confidence level may not be achieved.
#
class WilsonScoreIntervalMethod(StatisticalMethod):
    def __init__(
        self,
        epsilon: float | None = None,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 2,
    ):
        super().__init__(
            "Wilson score interval with continuity correction",
            epsilon,
            kappa,
            relative_error,
            bounds,
            binomial,
            min_samples,
        )

        # Only for binomial proportions, and requires absolute error in the sequential setting
        if not binomial:
            raise Exception(f"The {self.name} only applies to binomial proportions.")

        if epsilon is not None and relative_error:
            raise Exception(f"The {self.name} only supports absolute interval half-width.")

        # Precompute quantile
        self.z = norm.ppf(
            1.0 - 0.5 * kappa,
        )  # (1 - 0.5 * kappa = 0.5 + 0.5 * confidence)-quantile of standard normal
        self.z_squared = self.z * self.z

        self.worst_case_samples = (
            min_samples  # for fixed runs, we are confident from min_samples onward
        )

        # Sequential setting: half-width is given, pre-calculate number of samples needed based on worst-case of p=0.5
        # via exponential and binary search for smallest interval of half-width at least epsilon assuming p=0.5
        if self.epsilon is not None:

            def need_more_runs(n):
                (l, u) = self._get_interval(0.5, n)[1]
                return (u - l) * 0.5 > self.epsilon

            upper_runs = 2
            while need_more_runs(upper_runs):
                upper_runs *= 2
            lower_runs = upper_runs // 2
            while lower_runs + 1 < upper_runs:
                runs = (lower_runs + upper_runs) // 2
                if need_more_runs(runs):
                    lower_runs = runs  # interval too wide, need more runs
                else:
                    upper_runs = runs  # interval small enough, but can perhaps do with fewer runs
                assert not need_more_runs(upper_runs)  # loop invariant
            self.worst_case_samples = upper_runs

    def _get_interval(self, mean, count):
        lower = (
            0.0
            if mean == 0.0
            else max(
                0.0,
                (
                    2 * count * mean
                    + self.z_squared
                    - (
                        self.z
                        * numpy.sqrt(
                            self.z_squared
                            - 1 / count
                            + 4 * count * mean * (1 - mean)
                            + (4 * mean - 2),
                        )
                        + 1
                    )
                )
                / (2 * count + self.z_squared),
            )
        )
        upper = (
            1.0
            if mean == 1.0
            else min(
                1.0,
                (
                    2 * count * mean
                    + self.z_squared
                    + (
                        self.z
                        * numpy.sqrt(
                            self.z_squared
                            - 1 / count
                            + 4 * count * mean * (1 - mean)
                            - (4 * mean - 2),
                        )
                        + 1
                    )
                )
                / (2 * count + self.z_squared),
            )
        )
        return self.count >= self.worst_case_samples, (lower, upper)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Check minimum number of samples
        if self.count < self.min_samples:
            return False, None

        # Return interval
        return self._get_interval(self.mean, self.count)


# The normal approximation based confidence interval, for binomial proportions also known as the Wald interval
#
# Applies to samples from any distribution. Based on the central limit theorem, assumes a "large" number of samples.
#
# Given the samples collected so far (via add_sample)
# and one minus the desired confidence level (kappa),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# Whenever there are at least min_samples samples, the return value will indicate that no more samples are needed,
# and provide an interval based on the provided samples (fixed number of runs setting).
#
# This method is not sound: The requested confidence level may not be achieved.
#
class NormalIntervalMethod(StatisticalMethod):
    def __init__(
        self,
        epsilon: float | None = None,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 30,
    ):
        super().__init__(
            "normal approximation confidence interval",
            epsilon,
            kappa,
            relative_error,
            bounds,
            binomial,
            min_samples,
        )

        if min_samples < 30:
            DSMCLogger.get_logger().warning(
                f"Typical guideline for the normal interval method is min_samples >= 30. Otherwise, results might be inaccurate.",
            )

        # Precompute quantile
        self.z = norm.ppf(1.0 - 0.5 * self.kappa)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Check minimum number of samples (typical guideline is min_samples >= 30 for CLT and large number of samples assumption)
        if self.count < self.min_samples:
            return False, None

        # Return interval
        epsilon = self.z * self.stddev / np.sqrt(self.count)
        intv = (self.mean - epsilon, self.mean + epsilon)
        converged = self.epsilon is None or self._check_intv_2epsilon(intv)
        return converged, intv


# The Student's t approximation based confidence interval
#
# Applies to samples from any distribution. Based on the central limit theorem.
#
# Given the samples collected so far (via add_sample)
# and one minus the desired confidence level (kappa),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# Whenever there are at least min_samples samples, the return value will indicate that no more samples are needed,
# and provide an interval based on the provided samples (fixed number of runs setting).
#
# This method is not sound: The requested confidence level may not be achieved.
#
class StudentsTMethod(NormalIntervalMethod):
    def __init__(
        self,
        epsilon: float | None = None,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 30,
    ):
        super().__init__(epsilon, kappa, relative_error, bounds, binomial, min_samples)

        self.name = "Student's t"

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Return interval
        self.z = t(df=self.count - 1).ppf(1.0 - 0.5 * self.kappa)
        return super().get_interval()


# Hoeffding's inequality, as described in
# W. Hoeffding:
# Probability inequalities for sums of bounded random variables.
# Journal of the American Statistical Association (1963).
#
# Applies to samples from distributions with bounded support.
#
# Given the samples collected so far (via add_sample),
# optionally the desired absolute interval half-width (epsilon),
# one minus the desired confidence level (kappa),
# and the bounds of the underlying distribution's support (bounds),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# If the half-width is omitted, the return value will indicate that no more samples are needed,
# and provide an interval based on the provided samples (fixed number of runs setting).
#
# This method is sound.
#
class HoeffdingMethod(StatisticalMethod):
    def __init__(
        self,
        epsilon: float | None = 0.1,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 2,
    ):
        super().__init__(
            "Hoeffding's inequality",
            epsilon,
            kappa,
            relative_error,
            bounds,
            binomial,
            min_samples,
        )

        # Requires absolute error and bounded support
        if relative_error:
            raise Exception(f"{self.name} only supports absolute interval half-width.")
        if not is_bounded(bounds):
            raise Exception(
                f"{self.name} requires samples from a distribution with bounded support.",
            )

        self.worst_case_samples = min_samples

        # Sequential setting: half-width is given, pre-calculate number of runs needed
        if self.epsilon is not None:
            # n = ln(2/kappa) / 2*(epsilon/(b-a))^2
            scaled_epsilon = self.epsilon / (self.b - self.a)
            self.worst_case_samples = numpy.ceil(
                numpy.log(2.0 / self.kappa) / (scaled_epsilon * scaled_epsilon * 2.0),
            )

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Check minimum number of samples
        if self.count < self.min_samples:
            return False, None

        # Fixed number of runs setting: Calculate half-width
        epsilon = self.epsilon
        if epsilon is None or self.count < self.worst_case_samples:
            # epsilon = (b - a) * sqrt(ln(2/kappa) / 2n)
            epsilon = (self.b - self.a) * numpy.sqrt(numpy.log(2.0 / self.kappa) / (2.0 * self.count))

        # Return interval
        return self.count >= self.worst_case_samples, (self.mean - epsilon, self.mean + epsilon)


# Confidence interval around the mean using the
# Dvoretzky-Kiefer-Wolfowitz(-Massart) inequality (DKW).
#
# Applies to samples from distributions with bounded support.
#
# Given the samples collected so far (via add_sample),
# one minus the desired confidence level (kappa),
# and the bounds of the underlying distribution's support (bounds),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# Whenever there are at least min_samples samples, the return value will indicate that no more samples are needed,
# and provide an interval based on the provided samples (fixed number of runs setting).
#
# This method is sound.
#
class DKWMethod(StatisticalMethod):
    def __init__(
        self,
        epsilon: float | None = None,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 2,
    ):
        super().__init__(
            "Dvoretzky-Kiefer-Wolfowitz-Massart inequality",
            epsilon,
            kappa,
            relative_error,
            bounds,
            binomial,
            min_samples,
        )
        self.samples = []

        # Fixed number of runs setting only; requires bounded support
        if epsilon is not None:
            raise Exception("The DKW method only supports the fixed number of runs setting.")
        if not is_bounded(bounds):
            raise Exception(
                "The DKW method requires samples from a distribution with bounded support.",
            )

    def add_sample(self, x: float):
        super().add_sample(x)

        # Keep sorted list of samples
        bisect.insort(self.samples, x)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Check minimum number of samples
        if self.count < self.min_samples:
            return False, None

        # Calculate epsilon (same formula as for Hoeffding)
        epsilon = numpy.sqrt(numpy.log(2.0 / self.kappa) / (2.0 * self.count))

        # Determine how many observations to count fully
        obs_prob = 1.0 / self.count
        obs_to_consider_fully = int(self.count - epsilon / obs_prob)
        assert obs_to_consider_fully <= self.count

        # Calculate interval
        l = self.a * epsilon  # initial step of probability epsilon for lower bound a
        u = self.b * epsilon  # final step of probability epsilon for upper bound b
        for i in range(obs_to_consider_fully):  # observations that count fully
            l += self.samples[i] * obs_prob
            u += self.samples[self.count - i - 1] * obs_prob
        if (
            obs_to_consider_fully < self.count
        ):  # last observation we consider may only be accounted for partially
            remainingProbability = obs_prob - epsilon % obs_prob
            assert remainingProbability > 0.0
            l += self.samples[obs_to_consider_fully] * remainingProbability
            u += self.samples[self.count - obs_to_consider_fully - 1] * remainingProbability
        return True, (l, u)


# The EBStop algorithm as described in
# V. Mnih, C. Szepesvári, J.-Y. Audibert:
# "Empirical Bernstein stopping"
# ICML 2008.
#
# Applies to samples from any distribution.
#
# Given the samples collected so far (via add_sample),
# the desired relative interval half-width (epsilon),
# and one minus the desired confidence level (kappa),
# a call to get_interval returns None if more samples are needed and the current interval otherwise.
#
# This method is sound.
#
class EBStopMethod(StatisticalMethod):
    def __init__(
        self,
        epsilon: float | None = 0.1,
        kappa: float = 0.05,
        relative_error: bool = False,
        bounds: tuple[float | None, float | None] = (-numpy.inf, numpy.inf),
        binomial: bool = False,
        min_samples: int = 2,
    ):
        super().__init__(
            "EBStop",
            epsilon,
            kappa,
            relative_error,
            bounds,
            binomial,
            max(2, min_samples),
        )
        self.lb = 0.0
        self.ub = numpy.inf

        assert self.min_samples >= 2, "The EBStop method requires at least two samples."

        # Sequential setting with relative error only
        if epsilon is None:
            raise Exception(f"{self.name} only supports the sequential setting.")
        if not relative_error:
            raise Exception(f"{self.name} only supports relative interval half-width.")

        # Compensate ε so that we get an interval of width relative to its center (due to the calculation in the lines marked "relative interval" in get_interval below):
        # We return the interval [l, u] = [v / (1.0 + ε), v / (1.0 - ε)] (assuming v >= 0 w.l.o.g.),
        # so its width is v / (1.0 - ε) - v / (1.0 + ε), which can be > 2ε * (l + u)/2.
        # To compensate, we solve 1/(1 - x) - 1/(1 + x) = 2ε (with 0 < x < 1, 0 < epsilon < 1) for x, giving us the formula below for the necessary compensated ε:
        self.compensated_epsilon = (numpy.sqrt(1.0 / (self.epsilon**2) + 4) * self.epsilon - 1) / (2 * self.epsilon)

    def _d(self, t):
        return 1 / (t * (t + 1))

    def add_sample(self, x: np.array):
        super().add_sample(x)

        # Process sample
        log3dt = numpy.log(3.0 / self._d(self.count))
        c_t = (
            self.stddev * numpy.sqrt(2.0 * log3dt / self.count)
            + 3.0 * (self.b - self.a) * log3dt / self.count
        )
        self.lb = max(self.lb, abs(self.mean) - c_t)
        self.ub = min(self.ub, abs(self.mean) + c_t)

    def get_interval(self) -> tuple[bool, tuple[float, float] | None]:
        # Check minimum number of samples
        if self.count < self.min_samples:
            return False, None

        # Must have a non-zero sample mean
        if self.mean == 0.0:
            return False, None

        # Check stopping criterion...
        stopping_criterion = (1.0 + self.compensated_epsilon) * self.lb >= (
            1.0 - self.compensated_epsilon
        ) * self.ub
        rel_center = (
            numpy.sign(self.mean)
            * 0.5
            * ((1.0 + self.compensated_epsilon) * self.lb + (1.0 - self.compensated_epsilon) * self.ub)
        )
        # ...and return interval:
        # EBStop guarantees that, with the given confidence, if we stop, then |rel_center - true value| <= ε * true value,
        # so true value = rel_center / (1.0 + ε) and true value = rel_center / (1.0 - ε) are the extremal cases that the interval needs to cover.
        l = rel_center / (1.0 + self.compensated_epsilon)  # relative interval lower
        u = rel_center / (1.0 - self.compensated_epsilon)  # relative interval upper
        if u < l:
            (l, u) = (u, l)  # in case the sign is negative

        if stopping_criterion and not self._check_intv_2epsilon((l, u)):
            if self.relative_error:
                center = 0.5 * (l + u)
                req_intv_size = 2 * self.epsilon * center
            else:
                req_intv_size = 2 * self.epsilon
            DSMCLogger.get_logger().warning(
                f"EBS Stopping criterion reached, but len({(l, u)})={u - l} > {req_intv_size}",
            )

        return stopping_criterion, (l, u)
