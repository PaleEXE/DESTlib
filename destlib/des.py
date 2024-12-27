import copy
import json
import os
import random
from typing import Callable, Union, Dict, Iterator, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .time_units import TimeUnit


def _calculate_times(time_between: np.ndarray, service_time: np.ndarray) -> np.ndarray:
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

    result = np.stack(
            (
                time_between,
                service_time,
                arrival_time,
                start_time,
                end_time,
                wait_time,
                system_time,
                idle_time
            ), axis=1
    )
    return result


def _system_state(arrival_time: np.ndarray, end_time: np.ndarray) -> np.ndarray:
    state: np.ndarray = np.zeros(end_time[-1] - arrival_time[0])

    for i in range(arrival_time.shape[0]):
        state[arrival_time[i]:end_time[i]] += 1

    return state


class DES:
    _random_seed: Optional[int] = None
    _sample_size: Optional[int] = None

    _time_between_distro: Callable = random.uniform
    _time_between_params: Dict[str, Union[int, float]] = {"a": 0, "b": 1}

    _service_time_distro: Callable = random.uniform
    _service_time_params: Dict[str, Union[int, float]] = {"a": 0, "b": 1}

    _entity_name: str = "Entity"
    _system_name: str = "System"
    _sim_number: int = 1
    _time_unit: str = TimeUnit.Sec

    _df: pd.DataFrame = pd.DataFrame()

    vec_calculate_times = np.vectorize(_calculate_times, signature="(n),(n) -> (n,m)")
    vec_calculate_state = np.vectorize(_system_state, signature="(n),(n) -> (m)")

    def __init__(
        self,
        sample_size: Optional[int] = None,
        time_between_distro: Callable = random.uniform,
        time_between_params: Optional[Dict[str, Union[int, float]]] = None,
        service_time_distro: Callable = random.uniform,
        service_time_params: Optional[Dict[str, Union[int, float]]] = None,
        entity_name: str = "Entity",
        system_name: str = "System",
        sim_number: int = 1,
        time_unit: TimeUnit = TimeUnit.Sec,
    ) -> None:
        if time_between_params is None:
            time_between_params = {"a": 0, "b": 1}
        if service_time_params is None:
            service_time_params = {"a": 0, "b": 1}

        self._sample_size = sample_size
        self._time_between_distro = time_between_distro
        self._time_between_params = time_between_params
        self._service_time_distro = service_time_distro
        self._service_time_params = service_time_params
        self._entity_name = entity_name
        self._system_name = system_name
        self._sim_number = sim_number
        self._time_unit = time_unit

    def set_time_between_distro(self, distro: Callable, **params) -> 'DES':
        self._time_between_distro = distro
        self._time_between_params = params
        return self

    def set_service_time_distro(self, distro: Callable, **params) -> 'DES':
        self._service_time_distro = distro
        self._service_time_params = params
        return self

    def set_seed(self, seed: int) -> 'DES':
        self._random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        return self

    def set_sample_size(self, sample_size: int) -> 'DES':
        """
        set sample size
        :param sample_size: int can't be less than 1
        :return: the same object but set sample size
        """
        if sample_size < 0:
            raise ValueError(f"sample_size must be greater than or equal to 0: {sample_size}")
        self._sample_size = sample_size
        return self

    def set_entity_name(self, entity_name: str) -> 'DES':
        self._entity_name = entity_name
        return self

    def set_sim_number(self, sim_number: int) -> 'DES':
        """

        :param sim_number: int
        :return: Self
        """
        self._sim_number = sim_number
        return self

    def set_time_unit(self, time_unit: str) -> 'DES':
        if not TimeUnit.is_valid_unit(time_unit):
            raise ValueError(f"time_unit must be one of: {TimeUnit.all_units()}")
        self._time_unit = time_unit
        return self

    def set_system_name(self, system_name: str) -> 'DES':
        self._system_name = system_name
        return self

    def _generate_array(self, distro: Callable, params: dict) -> np.ndarray:
        if self._sample_size is None:
            raise Exception(f"Sample size must be defined")

        return np.array([distro(**params) for _ in range(self._sample_size)], dtype=np.int64)

    def get_sim_number(self) -> int:
        return self._sim_number

    def get_seed(self) -> Union[int, None]:
        return self._random_seed

    def _generate_time_between_array(self) -> np.ndarray:
        return self._generate_array(self._time_between_distro, self._time_between_params)

    def _generate_service_time_array(self) -> np.ndarray:
        return self._generate_array(self._service_time_distro, self._service_time_params)

    def run(self):
        time_between: np.ndarray = self._generate_time_between_array()
        time_between[0] = 0
        service_time: np.ndarray = self._generate_service_time_array()
        self._df = pd.DataFrame(
                DES.vec_calculate_times(time_between, service_time),
                columns=[
                    "time_between",
                    "service_time",
                    "arrival_time",
                    "start_time",
                    "end_time",
                    "wait_time",
                    "system_time",
                    "idle_time",
                ]
        )

    def show(self, n: int = 5) -> None:
        print(self._df[:n].to_markdown())
        print(self._df.sum().to_markdown())

    def plot(
        self,
        v_lines: bool = False,
        entity_color: str ='purple',
        arrival_color: str ='blue',
        departure_color: str ='pink',
    ) -> None:
        state: np.ndarray = DES.vec_calculate_state(
                self.df["arrival_time"].values,
                self.df["end_time"].values
        )

        time_intervals = np.arange(len(state))

        # Plot the system state using step function for visual clarity

        plt.figure(figsize=(10, 6))
        plt.step(time_intervals, state, where='post', color=entity_color, label=f'{self._entity_name} in System')

        if v_lines:
            plt.vlines(
                self.df["arrival_time"].values, ymin=0, ymax=state.max(), color=arrival_color, linestyle=':', label='Arrival'
                )
            plt.vlines(
                self.df["end_time"].values, ymin=0, ymax=state.max(), color=departure_color, linestyle=':', label='Departure'
                )

        plt.xlabel(self._time_unit)
        plt.ylabel(self._entity_name)
        plt.title(f'{self._system_name} State Over Time ({self._sim_number})')
        plt.legend()
        plt.show()

    def save_to(
        self,
        file_type: str = "csv",
        save_metadata: bool = True,
        save_statistics: bool = True,
        **kwargs
    ) -> None:
        folder_name = self._system_name
        os.makedirs(folder_name, exist_ok=True)

        sim_folder = os.path.join(folder_name, f"{self._entity_name}{self._sim_number}")
        os.makedirs(sim_folder, exist_ok=True)

        file_path = os.path.join(sim_folder, f"{self._entity_name}.{file_type}")

        # Save DataFrame
        if file_type == "csv":
            self._df.to_csv(file_path, **kwargs)
        elif file_type == "json":
            self._df.to_json(file_path, **kwargs)
        elif file_type == "xlsx":
            self._df.to_excel(file_path, **kwargs)
        else:
            raise ValueError("File type must be 'csv', 'json', or 'xlsx'")

        if save_metadata:
            self.save_metadata(file_path)

        if save_statistics:
            self.save_statistics(file_path)

    def save_metadata(self, path: str) -> None:
        metadata_path = path.replace(".csv", "_metadata.json").replace(".json", "_metadata.json")

        self._metadata = {
            "sample_size":         self._sample_size,
            "random_seed":         self._random_seed,
            "time_between_distro": self._time_between_distro.__name__,
            "time_between_params": self._time_between_params,
            "service_time_distro": self._service_time_distro.__name__,
            "service_time_params": self._service_time_params,
            "system_name":         self._system_name,
            "entity_name":         self._entity_name,
            "time_unit":           self._time_unit,
        }
        with open(metadata_path, "w") as metadata_file:
            json.dump(self._metadata, metadata_file, indent=4)

    def save_statistics(self, path: str) -> None:
        statistics_path = path.replace(".csv", "_statistics.json").replace(".json", "_statistics.json")

        if self._df.empty:
            raise ValueError("Simulation data is empty. Run the simulation before calculating statistics.")

        system_time: int = self._df["end_time"].values[-1] - self._df["arrival_time"].values[0]

        statistics = {
            "mean_time_between":                               round(self._df["time_between"].mean(), 3),
            "mean_service_time":                               round(self._df["service_time"].mean(), 3),
            "mean_idle_time":                                  round(self._df["idle_time"].mean(), 3),
            "mean_waiting_time":                               round(self._df["wait_time"].sum() / system_time, 3),
            f"mean_waiting_time for each {self._entity_name}": round(self._df["wait_time"].mean(), 3),
        }

        with open(statistics_path, "w") as stats_file:
            json.dump(statistics, stats_file, indent=4)

    @classmethod
    def load_from(cls, metadata_path: str) -> "DES":
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        distro_mapping = {
            "uniform":         random.uniform,
            "triangular":      random.triangular,
            "betavariate":     random.betavariate,
            "expovariate":     random.expovariate,
            "gammavariate":    random.gammavariate,
            "gauss":           random.gauss,
            "lognormvariate":  random.lognormvariate,
            "normalvariate":   random.normalvariate,
            "vonmisesvariate": random.vonmisesvariate,
            "paretovariate":   random.paretovariate,
            "weibullvariate":  random.weibullvariate,
        }

        des_instance = cls() \
            .set_sample_size(metadata["sample_size"]) \
            .set_seed(metadata["random_seed"]) \
            .set_system_name(metadata["system_name"]) \
            .set_entity_name(metadata["entity_name"]) \
            .set_time_unit(metadata["time_unit"])

        if metadata["time_between_distro"] in distro_mapping:
            des_instance.set_time_between_distro(
                    distro_mapping[metadata["time_between_distro"]],
                    **metadata["time_between_params"]
            )

        if metadata["service_time_distro"] in distro_mapping:
            des_instance.set_service_time_distro(
                    distro_mapping[metadata["service_time_distro"]],
                    **metadata["service_time_params"]
            )

        return des_instance

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self._df.to_dict("records"):
            yield row

    def __len__(self) -> int:
        return self._sample_size


def des_run_simulations(simulation: DES, n_times: int) -> Iterator[DES]:
    for i in range(n_times):
        new_des = copy.deepcopy(simulation)
        new_des.set_sim_number(new_des.get_sim_number() + i)
        if new_des.get_seed():
            new_des.set_seed(new_des.get_seed() + i)
        else:
            new_des.set_seed(random.randint(0, 123456798))
        new_des.run()
        yield new_des
