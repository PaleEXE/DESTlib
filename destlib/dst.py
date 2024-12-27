import copy
from typing import Type, Callable, List, Dict, Tuple, Any, Optional, Iterator
import functools
import inspect
import random
import re

type_pattern = functools.partial(re.findall, pattern=r"'[A-Za-z]*'")


class DST:
    _random_seed: Optional[int] = None
    _sim_class: Optional[Type] = None
    _sim_class_signature: Optional[inspect.Signature] = None
    _sim_class_parameters: Optional[Dict[str, Tuple[str, Any]]] = None

    _behaviors: Optional[List[Callable]] = None
    _weights: Optional[List[int]] = None
    _behaviors_calls: Optional[int] = None

    _args_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    _instant_number: int = 1

    def __init__(
        self,
        sim_class: Optional[Type] = None,
        behaviors: Optional[List[Callable]] = None,
        behaviors_calls: Optional[int] = None,
        args_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        weights: Optional[List[int]] = None,
        random_seed: Optional[int] = None,
        instant_number: int = 1,
    ) -> None:
        if behaviors is None:
            behaviors = []
        if args_ranges is None:
            args_ranges = {}

        self._sim_class = sim_class
        self._sim_class_signature: inspect.Signature = inspect.signature(sim_class.__init__)
        self._sim_class_parameters: Dict[str, Tuple[str, Any]] = DST.function_signature_to_dict(sim_class.__init__)
        self._behaviors: List[Callable] = behaviors
        self._behaviors_calls: Optional[int] = behaviors_calls
        self._args_ranges: Optional[Dict[str, Tuple[int, int]]] = args_ranges
        self._weights: Optional[List[int]] = weights
        self._random_seed: Optional[int] = random_seed
        self._instant_number: int = instant_number

    def set_sim_class(self, sim_class: Type) -> 'DST':
        self._sim_class = sim_class
        self._sim_class_signature = inspect.signature(sim_class.__init__)
        self._sim_class_parameters = DST.function_signature_to_dict(sim_class.__init__)
        return self

    def set_behaviors(self, behaviors: List[Callable]) -> 'DST':
        self._behaviors = behaviors
        return self

    def set_behaviors_calls(self, behaviors_calls: int) -> 'DST':
        self._behaviors_calls = behaviors_calls
        return self

    def set_weights(self, weights: List[int]) -> 'DST':
        self._weights = weights
        return self

    def set_args_ranges(self, **args_ranges: Dict[str, Tuple[int, int]]) -> 'DST':
        self._args_ranges = args_ranges
        return self

    def set_seed(self, random_seed: int) -> 'DST':
        self._random_seed = random_seed
        return self

    def set_instant_number(self, instant_number: int) -> 'DST':
        self._instant_number = instant_number
        return self

    def get_sim_class(self) -> Type:
        return self._sim_class

    def get_sim_class_signature(self) -> inspect.Signature:
        return self._sim_class_signature

    def get_sim_class_parameters(self) -> Dict[str, Tuple[str, Any]]:
        return self._sim_class_parameters

    def get_behaviors(self) -> List[Callable]:
        return self._behaviors

    def get_behaviors_calls(self) -> int:
        return self._behaviors_calls

    def get_args_ranges(self) -> Dict[str, Tuple[int, int]]:
        return self._args_ranges

    def get_weights(self) -> List[int]:
        return self._weights

    def get_seed(self) -> int:
        return self._random_seed

    def get_instant_number(self) -> int:
        return self._instant_number

    @staticmethod
    def function_signature_to_dict(func) -> Dict[str, Tuple[str, Any]]:
        sig = inspect.signature(func)
        param_dict = {}

        for name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
            default = param.default if param.default != inspect.Parameter.empty else None

            rizz = type_pattern(string=str(annotation))
            rizz = annotation if not rizz else rizz[0][1:-1]

            param_dict[name] = (rizz, default)

        return param_dict

    def run(self, with_defaults: bool = True, state_history: bool = False) -> Type[_sim_class]:
        if self._random_seed is not None:
            random.seed(self._random_seed)

        args = self.get_sim_args(with_defaults)

        instance = self._sim_class(**args)

        if self._weights is not None and len(self._weights) != len(self._behaviors):
            raise Exception("Weights and behaviors do not match")

        if self._weights is None:
            self._weights = [1] * len(self._behaviors)

        for _ in range(self._behaviors_calls):
            if state_history:
                yield copy.deepcopy(instance)

            behavior = random.choices(self._behaviors, weights=self._weights, k=1)[0]
            behavior(instance)

        return instance

    def get_sim_args(self, with_defaults):
        args = {}
        for arg, (arg_type, default) in self._sim_class_parameters.items():
            if arg in self._args_ranges:
                lower, upper = self._args_ranges[arg]
                if arg_type == 'str':
                    args[arg] = str(random.randint(lower, upper))
                elif arg_type == 'float':
                    args[arg] = random.uniform(lower, upper)
                else:
                    args[arg] = random.randint(lower, upper)
            elif with_defaults and default is not None:
                args[arg] = default
            elif arg == "self":
                continue
            else:
                raise Exception(f"{arg} values range is not specified and no default was specified")
        return args


def dst_run_simulations(simulation: DST, n_times: int, **kwargs) -> Iterator[DST]:
    for i in range(n_times):
        new_dst = copy.deepcopy(simulation)
        new_dst.set_instant_number(new_dst.get_instant_number() + i)
        if new_dst.get_seed():
            new_dst.set_seed(new_dst.get_seed() + i)
        else:
            new_dst.set_seed(random.randint(0, 123456798))
        new_dst.run(**kwargs)
        yield new_dst
