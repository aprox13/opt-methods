from typing import Optional, Callable, Any, Tuple

from core.extended_functions import DelegateFunction
from one_dim_search import *
from utils import eq_tol


class StepStrategy:
    prev_step = None

    @property
    def name(self) -> str:
        raise NotImplementedError

    def __init__(self, f: ExtendedFunction, eps: float = None):
        self.f = f
        self.eps = eps or 1e-7

    def calculate_step(self, x: np.ndarray):
        raise NotImplementedError

    def __call__(self, x: np.ndarray):
        res = self.calculate_step(x)
        assert res >= 0, 'negative step'
        self.prev_step = res
        return res


class DividePrevStrategy(StepStrategy):
    @property
    def name(self) -> str:
        return "divide"

    def calculate_step(self, x: np.ndarray):
        assert self.prev_step is not None

        step = self.prev_step

        x_new = x - step * self.f.grad(x)
        fx = self.f(x)
        while self.f(x_new) >= fx and step > self.eps:
            step /= 2
            x_new = x - step * self.f.grad(x)

        return step


class OneDimOptStrategy(StepStrategy):
    @property
    def name(self) -> str:
        return f'onedim/{self._m_name}'

    _SUPPORTED = {'dichotomy', 'golden', 'fibonacci'}

    def __init__(self, f: ExtendedFunction, method: str, search_range: Range, eps: float = None):
        super().__init__(f, eps)

        assert method in self._SUPPORTED, f'Unsupported method "{method}", expected one of {",".join(self._SUPPORTED)}'
        self._m_name = method
        if method == 'dichotomy':
            self.method = Dichotomy()
        elif method == 'golden':
            self.method = GoldenSection()
        else:
            self.method = Fibonacci()

        self.rng = search_range.copy()

    def calculate_step(self, x: np.ndarray):
        def target_f(step):
            return self.f(x - step * self.f.grad(x))

        res = self.method.search(self.rng.copy(), DelegateFunction(func=target_f), self.eps)
        return res.x


class GradDescent:
    _STOP_CRITERION = {'arg_margin', 'func_margin'}

    def search_with_iterations(self, f: ExtendedFunction,
                               start: np.ndarray,
                               step_strategy: StepStrategy,
                               stop_criterion: Optional[str] = None,
                               max_iters=10000,
                               initial_step=1,
                               eps=1e-8,
                               before_iteration: Callable[[int, np.ndarray], Any] = None,
                               ) -> Tuple[np.ndarray, int]:
        assert stop_criterion is None or stop_criterion in self._STOP_CRITERION, f'Unknown stop criterion "{stop_criterion}" '

        def stop_criterion(_prev_x, _x):
            if stop_criterion == 'arg_margin':
                return np.linalg.norm(_x, _prev_x) < eps
            else:
                return eq_tol(f(_x), f(_prev_x), eps)
            # return False

        prev = start
        step_strategy.prev_step = initial_step
        for iteration in range(max_iters):
            if before_iteration is not None:
                before_iteration(iteration, np.copy(prev))

            step = step_strategy(prev)

            next_x = prev - step * f.grad(prev)
            if stop_criterion(prev, next_x):
                return prev, iteration

            prev = next_x

        return prev, max_iters

    def search(self, f: ExtendedFunction,
               start: np.ndarray,
               step_strategy: StepStrategy,
               stop_criterion: Optional[str] = None,
               max_iters=10000,
               initial_step=1,
               eps=1e-8,
               before_iteration: Callable[[int, np.ndarray], Any] = None,
               ) -> np.ndarray:
        prev, _ = self.search_with_iterations(f,
                                              start,
                                              step_strategy,
                                              stop_criterion,
                                              max_iters,
                                              initial_step,
                                              eps,
                                              before_iteration)
        return prev
