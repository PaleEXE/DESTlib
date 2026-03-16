from __future__ import annotations

import copy
import functools
import inspect
import random
import re
from typing import Callable, Any, Iterator

type_pattern = functools.partial(re.findall, pattern=r"'[A-Za-z]*'")


class DST:
    def __init__(
        self,
        sim_class: type | None = None,
        behaviors: list[Callable[..., Any]] | None = None,
        behaviors_calls: int | None = None,
        args_ranges: dict[str, tuple[int, int]] | None = None,
        weights: list[int] | None = None,
        random_seed: int | None = None,
        instant_number: int = 1,
    ) -> None:
        self._sim_class: type | None = sim_class
        self._sim_class_signature: inspect.Signature | None = None
        self._sim_class_parameters: dict[str, tuple[str, Any]] | None = None

        if sim_class is not None:
            self._sim_class_signature = inspect.signature(sim_class.__init__)  # type: ignore[misc]
            self._sim_class_parameters = DST.function_signature_to_dict(
                sim_class.__init__  # type: ignore[misc]
            )

        self._behaviors: list[Callable[..., Any]] = behaviors if behaviors is not None else []
        self._behaviors_calls: int | None = behaviors_calls
        self._args_ranges: dict[str, tuple[int, int]] = args_ranges if args_ranges is not None else {}
        self._weights: list[int] | None = weights
        self._random_seed: int | None = random_seed
        self._instant_number: int = instant_number

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def set_sim_class(self, sim_class: type) -> DST:
        self._sim_class = sim_class
        self._sim_class_signature = inspect.signature(sim_class.__init__)  # type: ignore[misc]
        self._sim_class_parameters = DST.function_signature_to_dict(sim_class.__init__)  # type: ignore[misc]
        return self

    def set_behaviors(self, behaviors: list[Callable[..., Any]]) -> DST:
        self._behaviors = behaviors
        return self

    def set_behaviors_calls(self, behaviors_calls: int) -> DST:
        self._behaviors_calls = behaviors_calls
        return self

    def set_weights(self, weights: list[int]) -> DST:
        self._weights = weights
        return self

    def set_args_ranges(self, **args_ranges: tuple[int, int]) -> DST:
        self._args_ranges = args_ranges
        return self

    def set_seed(self, random_seed: int) -> DST:
        self._random_seed = random_seed
        return self

    def set_instant_number(self, instant_number: int) -> DST:
        self._instant_number = instant_number
        return self

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_sim_class(self) -> type | None:
        return self._sim_class

    def get_sim_class_signature(self) -> inspect.Signature | None:
        return self._sim_class_signature

    def get_sim_class_parameters(self) -> dict[str, tuple[str, Any]] | None:
        return self._sim_class_parameters

    def get_behaviors(self) -> list[Callable[..., Any]]:
        return self._behaviors

    def get_behaviors_calls(self) -> int | None:
        return self._behaviors_calls

    def get_args_ranges(self) -> dict[str, tuple[int, int]]:
        return self._args_ranges

    def get_weights(self) -> list[int] | None:
        return self._weights

    def get_seed(self) -> int | None:
        return self._random_seed

    def get_instant_number(self) -> int:
        return self._instant_number

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def function_signature_to_dict(func: Callable[..., Any]) -> dict[str, tuple[str, Any]]:
        """Extract parameter names, type annotations, and defaults from a callable."""
        sig = inspect.signature(func)
        param_dict: dict[str, tuple[str, Any]] = {}

        for name, param in sig.parameters.items():
            annotation = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else "Any"
            )
            default = (
                param.default if param.default != inspect.Parameter.empty else None
            )

            annotation_name = type_pattern(string=str(annotation))
            annotation_name = annotation if not annotation_name else annotation_name[0][1:-1]

            param_dict[name] = (annotation_name, default)

        return param_dict

    def get_sim_args(self, with_defaults: bool) -> dict[str, Any]:
        """Build a dict of constructor arguments for the simulation class."""
        if self._sim_class_parameters is None:
            raise ValueError("Simulation class must be set before generating args")

        args: dict[str, Any] = {}
        for arg, (arg_type, default) in self._sim_class_parameters.items():
            if arg in self._args_ranges:
                lower, upper = self._args_ranges[arg]
                if arg_type == "str":
                    args[arg] = str(random.randint(lower, upper))
                elif arg_type == "float":
                    args[arg] = random.uniform(lower, upper)
                else:
                    args[arg] = random.randint(lower, upper)
            elif with_defaults and default is not None:
                args[arg] = default
            elif arg == "self":
                continue
            else:
                raise ValueError(
                    f"{arg} values range is not specified and no default was specified"
                )
        return args

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def run(self, with_defaults: bool = True, state_history: bool = False) -> Any:
        """
        Run the simulation.

        If *state_history* is ``True`` this becomes a generator that yields
        deep copies of the instance before each behaviour step.  Otherwise
        it returns the final instance directly.
        """
        if self._random_seed is not None:
            random.seed(self._random_seed)

        if self._sim_class is None:
            raise ValueError("Simulation class must be set before running")
        if self._behaviors_calls is None:
            raise ValueError("behaviors_calls must be set before running")

        args = self.get_sim_args(with_defaults)
        instance = self._sim_class(**args)

        weights = self._weights
        if weights is not None and len(weights) != len(self._behaviors):
            raise ValueError("Weights and behaviors must have the same length")
        if weights is None:
            weights = [1] * len(self._behaviors)

        for _ in range(self._behaviors_calls):
            if state_history:
                yield copy.deepcopy(instance)

            behavior = random.choices(self._behaviors, weights=weights, k=1)[0]
            behavior(instance)

        return instance


def dst_run_simulations(simulation: DST, n_times: int, **kwargs: Any) -> Iterator[DST]:
    """Run a simulation multiple times, yielding a new DST object for each run."""
    for i in range(n_times):
        new_dst = copy.deepcopy(simulation)
        new_dst.set_instant_number(new_dst.get_instant_number() + i)
        seed = new_dst.get_seed()
        if seed is not None:
            new_dst.set_seed(seed + i)
        else:
            new_dst.set_seed(random.randint(0, 123456798))
        new_dst.run(**kwargs)
        yield new_dst
