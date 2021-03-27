from inspect import signature

import numpy as np

from one_dim_search import *


class FunctionWithGrad:
    def __init__(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray]):
        self.f = f
        self.grad_f = grad_f

        assert len(signature(f).parameters) == len(signature(grad_f).parameters) == 1

    def __call__(self, point: np.ndarray) -> float:
        return self.f(point)

    def grad(self, point: np.ndarray) -> np.ndarray:
        return self.grad_f(point)


class Paraboloid(FunctionWithGrad):
    def __init__(self):
        super(Paraboloid, self).__init__(
            f=lambda p: p[0] ** 2 + p[1] ** 2,
            grad_f=lambda p: np.array([2 * p[0], 2 * p[1]])
        )


class GradDescent:

    @staticmethod
    def search(f: FunctionWithGrad,
               start: np.ndarray,
               step_f: Callable[[FunctionWithGrad, np.ndarray, np.ndarray], float],
               stop_criterion: Callable[[FunctionWithGrad, np.ndarray, np.ndarray], bool]
               ) -> List[np.ndarray]:
        """

        :param f: - target function with grad support
        :param start: - starting point
        :param step_f: - calculating current step. It takes target function, current point and gradients in this point
        :param stop_criterion - determining if algorithm should stop. Takes function, previous and next point.
        """
        res = [np.copy(start)]

        prev = start
        while True:
            grad = f.grad(prev)
            step = step_f(f, prev, grad)
            nxt = prev - step * grad
            res.append(np.copy(nxt))

            if stop_criterion(f, prev, nxt):
                break

            prev = nxt

        return res


class ConstantStep:
    def __init__(self, step: float):
        self.step = step

    def __call__(self, f: FunctionWithGrad, point: np.ndarray, point_grad: np.ndarray):
        return self.step


class OneDimSearchStep:
    def __init__(self, name: str, lin_d=0.01, lin_k=2, lin_eps=1e-3, one_dim_eps=1e-5, **search_kwargs):
        if name == 'golden':
            func = GoldenSection(find_min=True)
        elif name == 'dichotomy':
            func = Dichotomy(find_min=True)
        elif name == 'fibonacci':
            func = Fibonacci(**search_kwargs)
        else:
            raise RuntimeError(f'Function {name} is not supported')

        self.ond_dim = func
        self.lin_d = lin_d
        self.lin_k = lin_k
        self.one_dim_eps = one_dim_eps
        self.lin_eps = lin_eps

    @staticmethod
    def _lin(f: OneDimFunction, start: float, d: float, eps: float, k: float) -> Range:
        assert d > 0
        if f(start) < f(start + d):
            d *= -1

        start_y = f(start)
        x = start + d
        step = d

        while f(x) <= start_y + eps:
            step *= k
            x += step

        if d > 0:
            return Range(start, x)
        else:
            return Range(x, start)

    def __call__(self, f: FunctionWithGrad, point: np.ndarray, point_grad: np.ndarray):
        ff = OneDimFunction(func=lambda step: f(point - step * point_grad))

        find_range = self._lin(ff, 0, d=self.lin_d, k=self.lin_k, eps=self.lin_eps)
        result_range = self.ond_dim.search(r=find_range, func=ff, eps=self.one_dim_eps)

        return result_range.final_range().center()
