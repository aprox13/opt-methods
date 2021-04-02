import numpy as np
import math


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
        return [202 * x[0] - 200 * x[1] - 2, 200 * (x[1] - x[0])]

    def hessian(self, x: np.ndarray):
        return [
            [202, -200],
            [-200, 200]
        ]


class Rosenbrock(Function):
    def f(self, x: np.ndarray):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def grad(self, x: np.ndarray):
        return [2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)]

    def hessian(self, x: np.ndarray):
        return [
            [-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ]


class F2(Function):
    def f(self, x: np.ndarray):
        return 2 * math.exp(-((x[0] - 1) / 2) ** 2 - (x[1] - 1) ** 2) + 3 * math.exp(
            -((x[0] - 2) / 3) ** 2 - ((x[1] - 3) / 2) ** 2)

    def grad(self, x: np.ndarray):
        return [
            -2 / 3 * (x[0] - 2) * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2) - (x[0] - 1) * math.exp(
                -1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            -3 / 2 * (x[1] - 3) * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2) - 4 * (
                        x[1] - 1) * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2)
        ]

    def hessian(self, x: np.ndarray):
        pass
