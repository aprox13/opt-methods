import numpy as np
from math import exp


class Function:
    def f(self, x: np.ndarray):
        pass

    def grad(self, x: np.ndarray):
        pass

    def hessian(self, x: np.ndarray):
        pass


class F1(Function):
    def f(self, x: np.ndarray):
        return 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2

    def grad(self, x: np.ndarray):
        return np.array([202 * x[0] - 200 * x[1] - 2, 200 * (x[1] - x[0])])

    def hessian(self, x: np.ndarray):
        return np.array([
            [202, -200],
            [-200, 200]
        ])


class Rosenbrock(Function):
    def f(self, x: np.ndarray):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def grad(self, x: np.ndarray):
        return np.array([2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)])

    def hessian(self, x: np.ndarray):
        return np.array([
            [-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])


class F2(Function):
    def f(self, x: np.ndarray):
        return 2 * exp(-((x[0] - 1) / 2) ** 2 - (x[1] - 1) ** 2) + 3 * exp(
            -((x[0] - 2) / 3) ** 2 - ((x[1] - 3) / 2) ** 2)

    def grad(self, x: np.ndarray):
        return np.array([
            -2 / 3 * (x[0] - 2) * exp(-1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) - (x[0] - 1) * exp(
                -0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            -1.5 * (x[1] - 3) * exp(-1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) - 4 * (x[1] - 1) * exp(
                -0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2)
        ])

    def hessian(self, x: np.ndarray):
        return np.array(
            [[4 / 27 * (x[0] - 2) ** 2 * exp(-1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) - 2 / 3 * exp(
                -1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) - exp(
                -0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2) + 0.5 * (x[0] - 1) ** 2 * exp(
                -0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
              1 / 3 * (x[0] - 2) * (x[1] - 3) * exp(-1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) + 2 * (
                      x[0] - 1) * (x[1] - 1) * exp(-0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2)],
             [1 / 3 * (x[0] - 2) * (x[1] - 3) * exp(-1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) + 2 * (
                     x[0] - 1) * (x[1] - 1) * exp(-0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
              0.75 * (x[1] - 3) ** 2 * exp(-1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) - 1.5 * exp(
                  -1 / 9 * (x[0] - 2) ** 2 - 0.25 * (x[1] - 3) ** 2) - 4 * exp(
                  -0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2) + 8 * (x[1] - 1) ** 2 * exp(
                  -0.25 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2)]])


class F2neg(Function):
    def __init__(self):
        self._base = F2()

    def f(self, x: np.ndarray):
        return -self._base.f(x)

    def grad(self, x: np.ndarray):
        return -self._base.grad(x)

    def hessian(self, x: np.ndarray):
        return -self._base.hessian(x)
