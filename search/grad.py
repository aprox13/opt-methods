from typing import Optional, Callable, Any, Tuple

from core.extended_functions import DelegateFunction
from core.function import Function
from search.one_dim_search import *
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


def fibonacci(f, a, b, eps=1e-5, verbose=False, n=60):
    if b < a:
        a, b = b, a
    start_len = b - a

    fib_numbers = [1, 1]
    if n is None:
        fib_condition = start_len / eps
        while fib_numbers[- 1] <= fib_condition:
            fib_numbers.append(fib_numbers[-1] + fib_numbers[-2])
        n = len(fib_numbers) - 3
    else:
        while len(fib_numbers) <= n + 2:
            fib_numbers.append(fib_numbers[-1] + fib_numbers[-2])
        n = 60

    x1 = a + start_len * (fib_numbers[n] / fib_numbers[n + 2])
    x2 = a + start_len * (fib_numbers[n + 1] / fib_numbers[n + 2])
    f_x1 = f(x1)
    f_x2 = f(x2)
    func_evals = 2

    step = 1
    if verbose:
        print('startA = %0.6f, startB = %0.6f, len = %0.6f' % (a, b, b - a))
    while step <= n:
        start_iter_len = b - a
        if f_x1 > f_x2:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = a + start_len * (fib_numbers[n - step + 1] / fib_numbers[n + 2])
            f_x2 = f(x2)
            func_evals += 1

        else:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + start_len * (fib_numbers[n - step] / fib_numbers[n + 2])
            f_x1 = f(x1)
            func_evals += 1

        end_iter_len = b - a
        if verbose:
            print('Iteration #%d, newA = %0.6f, newB = %0.6f, len = %0.6f, lenChangeCoef = %0.6f' % (
                step, a, b, b - a, start_iter_len / end_iter_len))

        step = step + 1

    return x1, step, func_evals


def search_range(f, x0, step=0.1, eps=1e-6, max_iter=1e4):
    if f(x0) < f(x0 + step):
        x0 += step
        step *= -1

    it = 0
    y0 = f(x0)
    x = x0 + step

    while f(x) <= y0 + eps and it < max_iter:
        step *= 2
        x += step
        it += 1

    if step > 0:
        return x0, x
    return x, x0


def step_func_one_dimensional_method(method):
    def result(f, f_grad_in_point, point):
        def optimization_func(alpha):
            return f(point - alpha * f_grad_in_point)

        left, right = search_range(optimization_func, 0)
        x, _, _ = method(optimization_func, left, right)
        return x

    return result


def grad_adapter(f: Function, w0):
    class F(ExtendedFunction):
        @property
        def name(self):
            return "F"

        def apply(self, x: np.ndarray) -> float:
            return f.f(x)

        def grad_apply(self, x: np.ndarray) -> np.ndarray:
            return f.grad(x)

    strategy = OneDimOptStrategy(F(), "fibonacci", search_range=Range(0, 1000), eps=1e-8)
    path = []
    w, iterations = GradDescent().search_with_iterations(F(),
                                                         w0,
                                                         step_strategy=strategy,
                                                         stop_criterion="func_margin",
                                                         eps=1e-8,
                                                         before_iteration=lambda iter_no, point: path.append(point)
                                                         )
    return w, iterations, path
