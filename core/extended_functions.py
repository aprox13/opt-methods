import math
from typing import Callable

import numpy as np

from core.extended_function import ExtendedFunction


class Paraboloid(ExtendedFunction):
    def apply(self, x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * x[0], 2 * x[1]])


class MinusSin1(ExtendedFunction):

    def apply(self, x: np.ndarray) -> float:
        return -math.sin(x[0])

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        return np.array([-math.cos(x[0])])


class DelegateFunction(ExtendedFunction):

    def apply(self, x: np.ndarray) -> float:
        return self._f(x)

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        return self._gf(x)

    def __init__(self, func: Callable[[np.ndarray], float] = None,
                 grad_func: Callable[[np.ndarray], np.ndarray] = None):
        super().__init__()
        self._f = func
        self._gf = grad_func
