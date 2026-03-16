from __future__ import annotations

from random import (
    uniform,
    gauss,
    triangular,
    betavariate,
    expovariate,
    gammavariate,
    lognormvariate,
    normalvariate,
    vonmisesvariate,
    paretovariate,
    weibullvariate,
)

from numpy.random import (
    poisson,
    binomial,
    geometric,
    exponential,
    chisquare,
    gamma,
    beta,
    normal,
    laplace,
    logistic,
    f,
    wald,
    rayleigh,
    pareto,
    zipf,
)

from typing import Iterator

from .des import DES, des_run_simulations
from .dst import DST, dst_run_simulations
from .time_units import TimeUnit

__all__ = [
    "DES",
    "DST",
    "TimeUnit",
    "run_simulations",
    "des_run_simulations",
    "dst_run_simulations",
    # random distributions re-exported for convenience
    "uniform",
    "gauss",
    "triangular",
    "betavariate",
    "expovariate",
    "gammavariate",
    "lognormvariate",
    "normalvariate",
    "vonmisesvariate",
    "paretovariate",
    "weibullvariate",
    "poisson",
    "binomial",
    "geometric",
    "exponential",
    "chisquare",
    "gamma",
    "beta",
    "normal",
    "laplace",
    "logistic",
    "f",
    "wald",
    "rayleigh",
    "pareto",
    "zipf",
]


def run_simulations(
    simulation: DES | DST,
    n_times: int,
    **kwargs: object,
) -> Iterator[DES] | Iterator[DST]:
    """Dispatch to the appropriate runner based on simulation type."""
    if isinstance(simulation, DES):
        return des_run_simulations(simulation, n_times)
    if isinstance(simulation, DST):
        return dst_run_simulations(simulation, n_times, **kwargs)

    raise TypeError(f"{simulation} is not a DES or DST instance")
