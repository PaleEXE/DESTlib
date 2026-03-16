from __future__ import annotations

import copy
import json
import os
import random
import heapq
from typing import Callable, Iterator, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .time_units import TimeUnit


distro_mapping: dict[str, Callable[..., float]] = {
    "uniform": random.uniform,
    "triangular": random.triangular,
    "betavariate": random.betavariate,
    "expovariate": random.expovariate,
    "gammavariate": random.gammavariate,
    "gauss": random.gauss,
    "lognormvariate": random.lognormvariate,
    "normalvariate": random.normalvariate,
    "vonmisesvariate": random.vonmisesvariate,
    "paretovariate": random.paretovariate,
    "weibullvariate": random.weibullvariate,
    "poisson": np.random.poisson,
    "binomial": np.random.binomial,
    "geometric": np.random.geometric,
    "exponential": np.random.exponential,
    "chisquare": np.random.chisquare,
    "gamma": np.random.gamma,
    "beta": np.random.beta,
    "normal": np.random.normal,
    "laplace": np.random.laplace,
    "logistic": np.random.logistic,
    "f": np.random.f,
    "wald": np.random.wald,
    "rayleigh": np.random.rayleigh,
    "pareto": np.random.pareto,
    "zipf": np.random.zipf,
}


# ---------------------------------------------------------------------------
# Timing calculation helpers
# ---------------------------------------------------------------------------

def _calculate_times(time_between: np.ndarray, service_time: np.ndarray) -> np.ndarray:
    """
    Calculate arrival, start, end, wait, system, and idle times for a single-server queue.
    Returns a stacked numpy array with all timing information.
    """
    arrival_time: np.ndarray = time_between.cumsum()
    start_time: np.ndarray = np.zeros_like(time_between)
    end_time: np.ndarray = np.zeros_like(time_between)
    wait_time: np.ndarray = np.zeros_like(time_between)
    system_time: np.ndarray = np.zeros_like(time_between)
    idle_time: np.ndarray = np.zeros_like(time_between)

    end_time[0] = service_time[0]
    system_time[0] = service_time[0]

    for i in range(1, time_between.shape[0]):
        start_time[i] = max(arrival_time[i], end_time[i - 1])
        end_time[i] = start_time[i] + service_time[i]
        wait_time[i] = start_time[i] - arrival_time[i]
        system_time[i] = end_time[i] - arrival_time[i]
        idle_time[i] = start_time[i] - end_time[i - 1]

    return np.stack(
        (time_between, service_time, arrival_time,
         start_time, end_time, wait_time, system_time, idle_time),
        axis=1,
    )


def _calculate_times_mms(
    time_between: np.ndarray, service_time: np.ndarray, num_servers: int = 1
) -> np.ndarray:
    """
    Calculate timing information for a multi-server (M/M/s) queue.
    Returns a stacked numpy array with all timing and server assignment information.
    """
    arrival_time: np.ndarray = time_between.cumsum()
    start_time: np.ndarray = np.zeros_like(arrival_time)
    end_time: np.ndarray = np.zeros_like(arrival_time)
    wait_time: np.ndarray = np.zeros_like(arrival_time)
    system_time: np.ndarray = np.zeros_like(arrival_time)
    idle_time: np.ndarray = np.zeros_like(arrival_time)
    server: np.ndarray = np.zeros_like(arrival_time)

    server_available_at: np.ndarray = np.zeros(num_servers)

    for i in range(len(arrival_time)):
        chosen_server: int = int(np.argmin(server_available_at))
        server[i] = chosen_server

        start_time[i] = max(arrival_time[i], server_available_at[chosen_server])
        end_time[i] = start_time[i] + service_time[i]
        wait_time[i] = start_time[i] - arrival_time[i]
        system_time[i] = end_time[i] - arrival_time[i]

        if i > 0 and server_available_at[chosen_server] > arrival_time[i]:
            idle_time[i] = 0
        else:
            idle_time[i] = max(0, start_time[i] - server_available_at[chosen_server])

        server_available_at[chosen_server] = end_time[i]

    return np.stack(
        (time_between, service_time, arrival_time,
         start_time, end_time, wait_time, system_time, idle_time, server),
        axis=1,
    )


def _calculate_times_mms_levels(
    time_between: np.ndarray,
    service_time: np.ndarray,
    level: np.ndarray,
    num_servers: int = 1,
) -> np.ndarray:
    """
    Calculate timing for a multi-server queue with priority levels.
    Returns a stacked numpy array with all timing, level, and server assignment information.
    """
    arrival_time: np.ndarray = time_between.cumsum()
    start_time: np.ndarray = np.zeros_like(arrival_time)
    end_time: np.ndarray = np.zeros_like(arrival_time)
    wait_time: np.ndarray = np.zeros_like(arrival_time)
    system_time: np.ndarray = np.zeros_like(arrival_time)
    idle_time: np.ndarray = np.zeros_like(arrival_time)
    server: np.ndarray = np.zeros_like(arrival_time)

    worker_queues: list[list[tuple[Any, Any, int]]] = [[] for _ in range(num_servers)]
    worker_available_time: list[float] = [0.0] * num_servers

    current_time: int = 0
    completed_tasks: int = 0
    total_tasks: int = len(arrival_time)
    enqueued: set[int] = set()

    while completed_tasks < total_tasks:
        for i in range(total_tasks):
            if arrival_time[i] == current_time and i not in enqueued:
                worker_idx: int = int(np.argmin(worker_available_time))
                heapq.heappush(
                    worker_queues[worker_idx], (level[i], arrival_time[i], i)
                )
                enqueued.add(i)

        for worker_idx in range(num_servers):
            if (
                worker_queues[worker_idx]
                and worker_available_time[worker_idx] <= current_time
            ):
                _, _, i = heapq.heappop(worker_queues[worker_idx])

                start_time[i] = float(max(current_time, arrival_time[i]))
                end_time[i] = start_time[i] + service_time[i]
                wait_time[i] = start_time[i] - arrival_time[i]
                system_time[i] = end_time[i] - arrival_time[i]
                idle_time[i] = float(max(0, start_time[i] - worker_available_time[worker_idx]))
                worker_available_time[worker_idx] = float(end_time[i])
                server[i] = worker_idx

                completed_tasks += 1

        current_time += 1

    return np.stack(
        (time_between, service_time, arrival_time, level,
         start_time, end_time, wait_time, system_time, idle_time, server),
        axis=1,
    )


def _system_state(arrival_time: np.ndarray, end_time: np.ndarray) -> np.ndarray:
    """
    Calculate the number of entities in the system at each time unit.
    Returns a numpy array representing the system state over time.
    """
    state: np.ndarray = np.zeros(end_time[-1] - arrival_time[0], dtype=int)

    for i in range(arrival_time.shape[0]):
        state[arrival_time[i] : end_time[i]] += 1

    return state


# ---------------------------------------------------------------------------
# Column name constants (eliminates duplication in run() branches)
# ---------------------------------------------------------------------------

_BASE_COLUMNS: list[str] = [
    "time_between", "service_time", "arrival_time",
    "start_time", "end_time", "wait_time", "system_time", "idle_time",
]

_COLUMNS_WITH_SERVER: list[str] = [*_BASE_COLUMNS, "server"]

_COLUMNS_WITH_LEVELS: list[str] = [
    *_BASE_COLUMNS[:3], "level", *_BASE_COLUMNS[3:], "server",
]


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _derive_path(path: str, suffix: str) -> str:
    """Replace the file extension of *path* with *suffix* (e.g. '_metadata.json')."""
    root, _ = os.path.splitext(path)
    return f"{root}{suffix}"


# ---------------------------------------------------------------------------
# DES simulation class
# ---------------------------------------------------------------------------

class DES:
    # Vectorised calculation helpers (class-level, immutable)
    vec_calculate_times: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.vectorize(
        _calculate_times, signature="(n),(n) -> (n,m)"
    )
    vec_calculate_times_mss: Callable[[np.ndarray, np.ndarray, int], np.ndarray] = (
        np.vectorize(_calculate_times_mms, signature="(n),(n),()->(n,m)")
    )
    vec_calculate_times_mms_levels: Callable[
        [np.ndarray, np.ndarray, np.ndarray, int], np.ndarray
    ] = np.vectorize(_calculate_times_mms_levels, signature="(n),(n),(n),()->(n,m)")

    vec_calculate_state: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.vectorize(
        _system_state, signature="(n),(n) -> (m)"
    )

    def __init__(
        self,
        sample_size: int | None = None,
        time_between_distro: Callable[..., float] = random.uniform,
        time_between_params: dict[str, int | float] | None = None,
        service_time_distro: Callable[..., float] = random.uniform,
        service_time_params: dict[str, int | float] | None = None,
        num_servers: int = 1,
        levels: list[str] | None = None,
        levels_prob: list[float] | None = None,
        entity_name: str = "Entity",
        system_name: str = "System",
        sim_number: int = 1,
        time_unit: str = TimeUnit.Sec,
    ) -> None:
        """Initialize the DES simulation object with all configuration parameters."""
        self._random_seed: int | None = None
        self._sample_size: int | None = sample_size

        self._time_between_distro: Callable[..., float] = time_between_distro
        self._time_between_params: dict[str, int | float] = (
            time_between_params if time_between_params is not None else {"a": 0, "b": 1}
        )

        self._service_time_distro: Callable[..., float] = service_time_distro
        self._service_time_params: dict[str, int | float] = (
            service_time_params if service_time_params is not None else {"a": 0, "b": 1}
        )

        self._num_servers: int = num_servers

        self._levels: list[str] | None = levels
        self._levels_prob: list[float] | None = levels_prob

        self._entity_name: str = entity_name
        self._system_name: str = system_name
        self._sim_number: int = sim_number
        self._time_unit: str = time_unit

        self._df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def set_time_between_distro(
        self, distro: Callable[..., float], **params: int | float
    ) -> DES:
        """Set the distribution and parameters for time between arrivals."""
        self._time_between_distro = distro
        self._time_between_params = params
        return self

    def set_service_time_distro(
        self, distro: Callable[..., float], **params: int | float
    ) -> DES:
        """Set the distribution and parameters for service times."""
        self._service_time_distro = distro
        self._service_time_params = params
        return self

    def set_seed(self, seed: int) -> DES:
        """Set the random seed for reproducibility."""
        self._random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        return self

    def set_sample_size(self, sample_size: int) -> DES:
        """
        Set the sample size (number of entities to simulate).

        :param sample_size: must be >= 0
        """
        if sample_size < 0:
            raise ValueError(
                f"sample_size must be greater than or equal to 0: {sample_size}"
            )
        self._sample_size = sample_size
        return self

    def set_entity_name(self, entity_name: str) -> DES:
        """Set the name of the entity being simulated."""
        self._entity_name = entity_name
        return self

    def set_sim_number(self, sim_number: int) -> DES:
        """Set the simulation number (useful for batch runs)."""
        self._sim_number = sim_number
        return self

    def set_time_unit(self, time_unit: str) -> DES:
        """Set the time unit for reporting (e.g., seconds, minutes)."""
        if not TimeUnit.is_valid_unit(time_unit):
            raise ValueError(f"time_unit must be one of: {TimeUnit.all_units()}")
        self._time_unit = time_unit
        return self

    def set_system_name(self, system_name: str) -> DES:
        """Set the name of the system being simulated."""
        self._system_name = system_name
        return self

    def set_levels(
        self,
        levels: list[str],
        levels_prob: list[float] | None = None,
    ) -> DES:
        """Set the levels, optional level names, and optional level probabilities."""
        self._levels = levels
        self._levels_prob = levels_prob
        return self

    def set_levels_prob(self, levels_prob: list[float]) -> DES:
        """Set the level probabilities (weights)."""
        self._levels_prob = levels_prob
        return self

    def set_num_servers(self, num_servers: int) -> DES:
        """Set the number of servers in the system."""
        self._num_servers = num_servers
        return self

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_levels(self) -> list[str] | None:
        """Get the current levels."""
        return self._levels

    def get_levels_prob(self) -> list[float] | None:
        """Get the current level probabilities (weights)."""
        return self._levels_prob

    def get_num_servers(self) -> int:
        """Get the current number of servers."""
        return self._num_servers

    def get_sim_number(self) -> int:
        """Get the simulation number."""
        return self._sim_number

    def get_seed(self) -> int | None:
        """Get the current random seed."""
        return self._random_seed

    @property
    def df(self) -> pd.DataFrame:
        """Get the simulation results DataFrame."""
        return self._df

    # ------------------------------------------------------------------
    # Internal sample generation
    # ------------------------------------------------------------------

    def _generate_array(
        self, distro: Callable[..., float], params: dict[str, int | float]
    ) -> np.ndarray:
        """Generate an array of random values using the specified distribution and parameters."""
        if self._sample_size is None:
            raise ValueError("Sample size must be defined")

        return np.array(
            [distro(**params) for _ in range(self._sample_size)], dtype=np.int64
        )

    def _generate_time_between_array(self) -> np.ndarray:
        """Generate the array of time between arrivals."""
        return self._generate_array(
            self._time_between_distro, self._time_between_params
        )

    def _generate_service_time_array(self) -> np.ndarray:
        """Generate the array of service times."""
        return self._generate_array(
            self._service_time_distro, self._service_time_params
        )

    def _generate_levels_array(self) -> np.ndarray:
        """Generate the array of levels for each entity based on probabilities."""
        if self._levels is None:
            raise ValueError("Levels must be defined")
        if self._levels_prob is None:
            raise ValueError("Levels prob must be defined")

        return np.random.choice(
            list(range(len(self._levels))),
            size=self._sample_size,
            p=self._levels_prob,
        )

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the simulation and populate the results DataFrame."""
        time_between: np.ndarray = self._generate_time_between_array()
        time_between[0] = 0
        service_time: np.ndarray = self._generate_service_time_array()

        if self._levels is not None:
            levels: np.ndarray = self._generate_levels_array()
            data = DES.vec_calculate_times_mms_levels(
                time_between, service_time, levels, self._num_servers,
            )
            self._df = pd.DataFrame(data, columns=_COLUMNS_WITH_LEVELS)
            self._df["level"] = self._df["level"].apply(
                lambda x: self._levels[int(x)]  # type: ignore[index]
            )
            return

        if self._num_servers > 1:
            data = DES.vec_calculate_times_mss(
                time_between, service_time, self._num_servers,
            )
            self._df = pd.DataFrame(data, columns=_COLUMNS_WITH_SERVER)
            return

        data = DES.vec_calculate_times(time_between, service_time)
        self._df = pd.DataFrame(data, columns=_BASE_COLUMNS)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _ensure_has_data(self) -> None:
        """Raise if the simulation hasn't been run yet."""
        if self._df.empty:
            raise ValueError(
                "Simulation data is empty. Run the simulation before calculating statistics."
            )

    def compute_statistics(self) -> dict[str, float]:
        """Compute and return various statistics for the simulation."""
        self._ensure_has_data()

        return {
            "average_waiting_time": self._df["wait_time"].mean(),
            "average_service_time": self._df["service_time"].mean(),
            "average_time_between_arrivals": self._df["time_between"].mean(),
            "idle_time_percentage": (
                self._df["idle_time"].sum() / self._df["end_time"].values[-1]
            ) * 100,
            "std_dev_waiting_time": self._df["wait_time"].std(),
        }

    def compare_with_expected(
        self, expected_service_time: float, expected_time_between: float
    ) -> dict[str, float]:
        """
        Compare computed statistics with expected values.
        Returns a dictionary containing the comparison results.
        """
        stats = self.compute_statistics()
        return {
            "average_service_time_vs_expected": (
                stats["average_service_time"] - expected_service_time
            ),
            "average_time_between_arrivals_vs_expected": (
                stats["average_time_between_arrivals"] - expected_time_between
            ),
        }

    # ------------------------------------------------------------------
    # Display & plotting
    # ------------------------------------------------------------------

    def show(self, n: int = 5) -> None:
        """Print the first n rows and the sum of the DataFrame."""
        print(self._df[:n].to_markdown())
        print(self._df.sum().to_markdown())

    def plot(
        self,
        v_lines: bool = False,
        entity_color: str = "purple",
        arrival_color: str = "blue",
        departure_color: str = "pink",
    ) -> None:
        """Plot the system state over time, optionally with arrival and departure lines."""
        state: np.ndarray = DES.vec_calculate_state(
            self.df["arrival_time"].values,
            self.df["end_time"].values,
        )

        time_intervals = np.arange(len(state))

        plt.figure(figsize=(10, 6))
        plt.step(
            time_intervals,
            state,
            where="post",
            color=entity_color,
            label=f"{self._entity_name} in System",
        )

        if v_lines:
            plt.vlines(
                self.df["arrival_time"],
                ymin=0,
                ymax=state.max(),
                color=arrival_color,
                linestyle=":",
                label="Arrival",
            )
            plt.vlines(
                self.df["end_time"],
                ymin=0,
                ymax=state.max(),
                color=departure_color,
                linestyle=":",
                label="Departure",
            )

        plt.xlabel(self._time_unit)
        plt.ylabel(self._entity_name)
        plt.title(f"{self._system_name} State Over Time ({self._sim_number})")
        plt.legend()
        plt.show()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to(
        self,
        file_type: str = "csv",
        save_metadata: bool = True,
        save_statistics: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the simulation results to a file (csv, json, or xlsx), with optional metadata and statistics."""
        folder_name: str = self._system_name
        os.makedirs(folder_name, exist_ok=True)

        sim_folder: str = os.path.join(
            folder_name, f"{self._entity_name}{self._sim_number}"
        )
        os.makedirs(sim_folder, exist_ok=True)

        file_path: str = os.path.join(sim_folder, f"{self._entity_name}.{file_type}")

        _SAVE_DISPATCH: dict[str, Callable[..., None]] = {
            "csv": self._df.to_csv,
            "json": self._df.to_json,
            "xlsx": self._df.to_excel,
        }

        saver = _SAVE_DISPATCH.get(file_type)
        if saver is None:
            raise ValueError("File type must be 'csv', 'json', or 'xlsx'")
        saver(file_path, **kwargs)

        if save_metadata:
            self.save_metadata(file_path)

        if save_statistics:
            self.save_statistics(file_path)

    def save_metadata(self, path: str) -> None:
        """Save simulation metadata to a JSON file."""
        metadata_path = _derive_path(path, "_metadata.json")

        metadata: dict[str, Any] = {
            "sample_size": self._sample_size,
            "random_seed": self._random_seed,
            "time_between_distro": self._time_between_distro.__name__,
            "time_between_params": self._time_between_params,
            "service_time_distro": self._service_time_distro.__name__,
            "service_time_params": self._service_time_params,
            "system_name": self._system_name,
            "entity_name": self._entity_name,
            "time_unit": self._time_unit,
            "num_servers": self._num_servers,
            "levels": self._levels,
            "levels_prob": self._levels_prob,
        }
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

    def save_statistics(self, path: str) -> None:
        """Save simulation statistics to a JSON file."""
        statistics_path = _derive_path(path, "_statistics.json")
        self._ensure_has_data()

        statistics: dict[str, float | dict[str, float]] = {
            "mean_time_between": round(self._df["time_between"].mean(), 3),
            "mean_service_time": round(self._df["service_time"].mean(), 3),
            "mean_idle_time": round(self._df["idle_time"].mean(), 3),
            f"mean_waiting_time for each {self._entity_name}": round(
                self._df["wait_time"].mean(), 3
            ),
        }

        if "level" in self._df.columns:
            statistics["mean_waiting_time_per_level"] = {
                level: round(mean, 3)
                for level, mean in self._df.groupby("level")["wait_time"].mean().items()
            }

        with open(statistics_path, "w") as stats_file:
            json.dump(statistics, stats_file, indent=4)

    @classmethod
    def load_from(cls, metadata_path: str) -> DES:
        """Load a DES simulation object from a metadata JSON file."""
        with open(metadata_path, "r") as metadata_file:
            metadata: dict[str, Any] = json.load(metadata_file)

        sample_size = metadata.get("sample_size")
        random_seed = metadata.get("random_seed")
        system_name = metadata.get("system_name", "System")
        entity_name = metadata.get("entity_name", "Entity")
        time_unit = metadata.get("time_unit", TimeUnit.Sec)

        des_instance = cls()

        if sample_size is not None:
            des_instance.set_sample_size(sample_size)
        if random_seed is not None:
            des_instance.set_seed(random_seed)

        des_instance.set_system_name(system_name)
        des_instance.set_entity_name(entity_name)
        des_instance.set_time_unit(time_unit)

        # Set number of servers if present
        if "num_servers" in metadata:
            des_instance.set_num_servers(metadata["num_servers"])

        # Set levels and probabilities if present
        if "levels" in metadata:
            des_instance.set_levels(metadata["levels"], metadata.get("levels_prob"))

        # Set time_between distribution and params
        if metadata.get("time_between_distro") in distro_mapping:
            des_instance.set_time_between_distro(
                distro_mapping[metadata["time_between_distro"]],
                **metadata.get("time_between_params", {}),
            )

        # Set service_time distribution and params
        if metadata.get("service_time_distro") in distro_mapping:
            des_instance.set_service_time_distro(
                distro_mapping[metadata["service_time_distro"]],
                **metadata.get("service_time_params", {}),
            )

        return des_instance

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the simulation results as dictionaries."""
        yield from self._df.to_dict("records")

    def __len__(self) -> int:
        """Get the sample size (number of entities)."""
        return self._sample_size if self._sample_size is not None else 0


def des_run_simulations(simulation: DES, n_times: int) -> Iterator[DES]:
    """Run a simulation multiple times, yielding a new DES object for each run."""
    for i in range(n_times):
        new_des: DES = copy.deepcopy(simulation)
        new_des.set_sim_number(new_des.get_sim_number() + i)
        new_des.set_seed(new_des.get_seed() or random.randint(0, 123456798) + i)
        new_des.run()
        yield new_des
