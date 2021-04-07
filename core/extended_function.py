from typing import Union

import numpy as np


class Cached:
    def __init__(self, func):
        self.calls = 0
        self.reads = 0
        self.func = func
        self._cache = {}

    def __call__(self, x: np.ndarray):
        self.reads += 1

        key = tuple(x)
        if key not in self._cache:
            self._cache[key] = self.func(x)
            self.calls += 1

        return self._cache[key]


class ExtendedFunction:

    def __init__(self):
        self.apply_cache = Cached(self.apply)
        self.grad_cache = Cached(self.grad)

    def apply(self, x: np.ndarray) -> float:
        raise NotImplementedError("function not implemented")

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("function not implemented")

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.grad_cache(x)

    def __call__(self, x: Union[np.ndarray, float, int]) -> float:

        if isinstance(x, int) or isinstance(x, float):
            args = np.array([x])
        elif isinstance(x, np.ndarray):
            args = x
        else:
            raise RuntimeError(f"Unsupported type {type(x)} for func")

        return self.apply_cache(args)
