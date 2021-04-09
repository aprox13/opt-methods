import math
from typing import Callable

import numpy as np

from core.extended_function import ExtendedFunction


class Paraboloid(ExtendedFunction):
    @property
    def name(self):
        return "x^2 + y^2"

    def apply(self, x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * x[0], 2 * x[1]])


class MinusSin1(ExtendedFunction):

    @property
    def name(self):
        return "-sin(x)"

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

    @property
    def name(self):
        return "delegate"


class PureQuadraticFunction(ExtendedFunction):
    def __init__(self, matrix: np.ndarray):
        super().__init__()
        self._f = lambda x: matrix.dot(x).dot(x)
        self._gf = lambda x: np.dot(matrix, x)

    def apply(self, x: np.ndarray) -> float:
        return self._f(x)

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        return self._gf(x)


class QuadraticFunction(ExtendedFunction):
    def __init__(self, a: np.ndarray, b: np.ndarray, c: float):
        super().__init__()
        self._f = lambda x: a.dot(x).dot(x) + b.dot(x) + c
        self._gf = lambda x: np.dot(a, x) + b

    def apply(self, x: np.ndarray) -> float:
        return self._f(x)

    def grad_apply(self, x: np.ndarray) -> np.ndarray:
        return self._gf(x)
