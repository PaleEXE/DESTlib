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

from .des import *
from .dst import *
from .time_units import *

__all__ = [
    "des",
    "dst",
    "time_units",
    "DES",
    "DST",
    "run_simulations",
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
    "TimeUnit",
]


def run_simulations(
    simulation: Union[DES, DST],
    n_times: int,
    **kwargs,  # noqa: F405
) -> Iterator[Union[DES, DST]]:  # noqa: F405
    if isinstance(simulation, DES):  # noqa: F405
        return des_run_simulations(simulation, n_times)  # noqa: F405
    if isinstance(simulation, DST):  # noqa: F405
        return dst_run_simulations(simulation, n_times, **kwargs)  # noqa: F405

    raise TypeError(f"{simulation} is not a des_lib or DST")
