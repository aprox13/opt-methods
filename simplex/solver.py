from typing import Tuple, Optional, Callable

import numpy as np
import scipy.optimize as opt

np.seterr(divide='ignore', invalid='ignore')


def first_true_or_none(cond):
    for i, is_true in enumerate(cond):
        if is_true:
            return i
    return None


def single_one_index(x: np.ndarray):
    res, *_ = x.nonzero()
    assert len(res) == 1
    return res[0]


def add_first_row(M: np.ndarray, row: np.ndarray):
    assert len(row.shape) == 1, f'{row.shape}'
    return np.vstack((row[np.newaxis], M))


def add_first_column(M: np.ndarray, row: np.ndarray):
    assert len(row.shape) == 1
    return np.hstack((row[np.newaxis].T, M))


def concat(x, y):
    return np.concatenate((x, y))


class SimplexCanonicalIssue:
    """
    F = <x, c>
    Ax = b

    ∀.i: b[i] >= 0
    """

    @staticmethod
    def from_lte(A: np.ndarray, b: np.ndarray, c: np.ndarray):
        additional = len(b)

        A = np.hstack((A, np.eye(additional)))
        c = np.hstack((c, np.zeros(additional)))

        return SimplexCanonicalIssue(A, b, c, additional_vars_count=additional)

    def original_Abc(self):
        if self.is_from_lte:
            firstly_vars = self.A.shape[1] - self.additional_vars_count
            A = self.A[:, 0:firstly_vars]
            c = self.c[0:firstly_vars]
        else:
            A = self.A
            c = self.c
        b = self.b

        return np.copy(A), np.copy(b), np.copy(c)

    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, additional_vars_count=0):
        self._A = np.array(np.copy(A), float)
        self._b = np.array(np.copy(b), float)
        self._c = np.array(np.copy(c), float)
        self.additional_vars_count = additional_vars_count

        assert len(b.shape) == 1, f"Required 1d vector for b, but got {b.shape}"
        assert len(c.shape) == 1, f"Required 1d vector for c, but got {c.shape}"
        assert A.shape == (len(b), len(c)), f"A required with shape {len(c)}x{len(b)}, but got {A.shape}"

        for i in range(len(b)):
            if b[i] < 0:
                self._b[i] *= -1
                self._A[i] *= -1

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def Abc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.A, self.b, self.c

    def with_negate_function(self):
        return SimplexCanonicalIssue(A=np.copy(self.A), b=np.copy(self.b), c=np.copy(self.c) * -1)

    @property
    def is_from_lte(self):
        return self.additional_vars_count != 0

    def __copy__(self):
        return SimplexCanonicalIssue(self.A.copy(), self.b.copy(), self.c.copy(), int(self.additional_vars_count))

    def __repr__(self):
        return f'Issue{{A={self.A.shape}, is_lte={self.is_from_lte}, additional={self.additional_vars_count}}}'


class SimplexResult:
    def __init__(self, x, solution: np.ndarray):
        self._x = x
        self._sol = np.copy(solution)

    @property
    def x(self):
        return self._x

    @property
    def solution(self) -> np.ndarray:
        return self._sol

    def negated(self):
        return SimplexResult(self._x * -1, self._sol)

    def __eq__(self, other):
        return self.x == other.x

    def __repr__(self):
        return f'Result(x={self.x}, point=({", ".join(map(str, self._sol))}))'


class SimplexSolver:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def resolve_min(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        raise NotImplementedError()

    def resolve_max(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        raise NotImplementedError()

    @staticmethod
    def negated_task(f: Callable[[SimplexCanonicalIssue], SimplexResult],
                     issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        neg = issue.with_negate_function()

        res = f(neg)
        if res is not None:
            f = res.solution @ issue.c if len(res.solution) != 0 else -res.x
            return SimplexResult(f, res.solution)

        return None


class ScipySimplexSolver(SimplexSolver):

    def resolve_min(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        A, b, c = issue.original_Abc()

        params = dict(method='simplex', options=dict(tol=self.eps))
        if issue.is_from_lte:

            params['A_ub'] = A
            params['b_ub'] = b
        else:
            params['A_eq'] = A
            params['b_eq'] = b

        params['c'] = c

        res = opt.linprog(**params)

        if not res.success:
            return None
        else:
            return SimplexResult(x=res.fun, solution=np.copy(res.x))

    def resolve_max(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        return self.negated_task(self.resolve_min, issue)


class SimplexV2(SimplexSolver):
    def resolve_min(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        return self.negated_task(self.resolve_max, issue)

    M = 1e12

    def resolve_max(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        A, b, c = issue.Abc

        n, m = A.shape
        eps = self.eps

        A = np.hstack((A, np.eye(n)))
        c = np.hstack((c, np.repeat(-self.M, n)))

        result = (b * self.M).sum()
        c = c + (A * self.M).sum(axis=0)

        # начальное решение X = 0, X_new (базис) = b
        basis = np.arange(n) + m

        while c[c.argmax()] > 0:
            max_elem_in_c_index = c.argmax()

            resolving_colimn = b / A[:, max_elem_in_c_index]
            minimal = np.inf

            resolver_index = -1
            for i in range(n):
                if A[i, max_elem_in_c_index] > eps and resolving_colimn[i] <= minimal:
                    resolver_index = i
                    minimal = resolving_colimn[i]
            if resolver_index == -1:
                return None

            basis[resolver_index] = max_elem_in_c_index

            mul = c[max_elem_in_c_index] / A[resolver_index, max_elem_in_c_index]
            c -= A[resolver_index] * mul
            result -= b[resolver_index] * mul

            for i in range(n):
                if i == resolver_index:
                    continue
                mul = A[i, max_elem_in_c_index] / A[resolver_index, max_elem_in_c_index]
                A[i] -= A[resolver_index] * mul
                b[i] -= b[resolver_index] * mul

        if abs(result) < self.M:
            x = np.zeros(m)

            for i in range(n):
                x[basis[i]] = b[i] / A[i, basis[i]]

            return SimplexResult(-result, x)
        return None


def make_issue(A, b, c):
    return SimplexCanonicalIssue(np.array(A), np.array(b), np.array(c))
