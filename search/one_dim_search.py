import math
from abc import ABC
from typing import List, Callable


class TargetFunction:
    calls = 0

    def apply(self, x: float) -> float:
        raise NotImplementedError("function not implemented")

    def __call__(self, x) -> float:
        self.calls += 1
        return self.apply(x)


class Range:
    def __init__(self, left: float, right: float):
        self.left = left
        self.right = right
        assert left <= right

    def tupled(self):
        return self.left, self.right

    def __repr__(self):
        return f"[{self.left}, {self.right}]"

    def length(self) -> float:
        return abs(self.right - self.left)

    def center(self):
        return (self.left + self.right) / 2

    def move_left(self, left: float):
        assert left <= self.right
        self.left = left

    def move_right(self, right: float):
        assert right >= self.left
        self.right = right

    def copy(self):
        return Range(self.left, self.right)


Ranges = List[Range]


class SearchResult:
    def __init__(self, x: float, result: float, ranges: Ranges, iterations: int, calls_count: int):
        self.result = result
        self.ranges = ranges
        self.iterations = iterations
        self.calls_count = calls_count
        self.x = x

    @staticmethod
    def of(ranges: Ranges, func: TargetFunction):
        calls = func.calls

        x = ranges[-1].center()
        result = func(x)
        return SearchResult(x, result, ranges, len(ranges), calls)


class CachedFunction(TargetFunction, ABC):
    _cache = {}

    def __call__(self, x: float) -> float:
        if x in self._cache:
            return self._cache[x]
        self._cache[x] = super().__call__(x)
        return self._cache[x]

    def has_cache(self):
        return len(self._cache) != 0


class OneDimFunction(CachedFunction):
    def __init__(self, func: Callable[[float], float]):
        self.func = func
        self._cache = {}

    def apply(self, x: float) -> float:
        return self.func(x)


class OneDimSearch:
    support_calls_count = 0

    def search(self, r: Range, func: TargetFunction, eps: float) -> SearchResult:
        raise NotImplementedError("search not impl")


class Dichotomy(OneDimSearch):
    def __init__(self, find_min: bool = True):
        self.support_calls_count = 2
        if find_min:
            self.C = 1
        else:
            self.C = -1

    def search(self, r: Range, func: TargetFunction, eps: float) -> SearchResult:
        lf = func(r.left)
        rf = func(r.right)

        func.calls = 0

        if lf * rf > 0:
            raise RuntimeError("Required distinct values at the ends of range")

        current_range = r.copy()
        result = [r.copy()]

        while current_range.length() > eps:
            x = current_range.center()
            f1, f2 = func(x - eps), func(x + eps)

            if self.C * f1 < self.C * f2:
                current_range.move_right(x)
            else:
                current_range.move_left(x)

            result.append(current_range.copy())

        return SearchResult.of(result, func)


# https://ru.wikipedia.org/wiki/Метод_золотого_сечения#Формализация
class GoldenSection(OneDimSearch):
    _PHI = (1 + math.sqrt(5)) / 2

    def __init__(self, find_min: bool = True):
        self.support_calls_count = 2
        if find_min:
            self.C = 1
        else:
            self.C = -1

    def search(self, r: Range, func: TargetFunction, eps: float) -> SearchResult:
        func.calls = 0

        def x1x2(rng: Range) -> (float, float):
            d = (rng.right - rng.left) / self._PHI
            return rng.right - d, rng.left + d

        current_range = r.copy()
        result = [r.copy()]

        while current_range.length() > eps:
            x1, x2 = x1x2(current_range)

            if self.C * func(x1) >= self.C * func(x2):
                current_range.move_left(x1)
            else:
                current_range.move_right(x2)
            result.append(current_range.copy())

        return SearchResult.of(result, func)


class Fib:

    def __init__(self):
        self._fib = {
            0: 1,
            1: 1
        }

    def __getitem__(self, item):
        if item in self._fib:
            return self._fib[item]

        self._fib[item] = self[item - 1] + self[item - 2]
        return self._fib[item]

class Fibonacci(OneDimSearch):

    def __init__(self, fib: Fib, L: float = None):
        self.fib = fib
        self.L = L

    def _lambda(self, r: Range, n, k):
        return r.left + (self.fib[n - k - 2] / self.fib[n - k]) * (r.right - r.left)

    def _mu(self, r: Range, n, k):
        return r.left + (self.fib[n - k - 1] / self.fib[n - k]) * (r.right - r.left)

    def search(self, r: Range, func: TargetFunction, eps: float) -> SearchResult:
        func.calls = 0
        n = 0

        L = self.L if self.L is not None else eps

        t = r.length() / L
        while self.fib[n] <= t:
            n += 1

        lmbda = self._lambda(r, n, k=0)
        mu = self._mu(r, n, k=0)
        k = 1

        result = [r.copy()]
        current_range = r.copy()

        while k != n - 2:
            # first
            if func(lmbda) > func(mu):
                # second
                current_range.move_left(lmbda)
                lmbda = mu
                mu = self._mu(current_range, n, k + 1)
            else:
                current_range.move_right(mu)
                mu = lmbda
                lmbda = self._lambda(current_range, n, k + 1)

            k += 1
            result.append(current_range.copy())

        mu = lmbda + eps
        if func(lmbda) == func(mu):
            current_range.move_left(lmbda)
        else:
            current_range.move_right(mu)

        result.append(current_range.copy())
        return SearchResult.of(result, func)







