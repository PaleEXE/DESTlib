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
    "TimeUnit",
]


def run_simulations(
    simulation: Union[DES, DST], n_times: int, **kwargs
) -> Iterator[Union[DES, DST]]:
    if isinstance(simulation, DES):
        return des_run_simulations(simulation, n_times)
    if isinstance(simulation, DST):
        return dst_run_simulations(simulation, n_times, **kwargs)

    raise TypeError(f"{simulation} is not a des_lib or DST")
