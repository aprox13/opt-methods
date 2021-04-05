import math
from typing import Tuple, Optional

import numpy as np
import scipy.optimize as opt


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
        self._A = np.copy(A)
        self._b = np.copy(b)
        self._c = np.copy(c)
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


class SimplexSolver:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def resolve_min(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        res = self.resolve_max(issue.with_negate_function())
        if res is not None:
            return res.negated()
        return None

    def resolve_max(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        raise NotImplementedError()


class SimplexSolverImpl(SimplexSolver):

    def _gauss(self, A: np.ndarray, r, s):
        n, m = A.shape

        A[r, :m] /= A[r, s]
        for i in range(n):
            if r == i:
                continue
            A[i, 0:m] -= A[r, 0:m] * A[i, s]

    def _has_solution(self, A: np.ndarray, B: np.ndarray) -> bool:
        n, m = A.shape
        while True:
            s = first_true_or_none(A[0, 1:m] < 0)

            if s is None:
                return True
            s += 1

            r, t = None, math.inf
            for i in range(1, n):
                if A[i, s] > 0 and A[i, 0] / A[i, s] < t:
                    r = i
                    t = A[i, 0] / A[i, s]

            if r is None:
                return False

            self._gauss(A, r, s)

            B[r - 1] = s

    def resolve_max(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        A, b, c = issue.Abc

        m, n = A.shape

        A = np.hstack((A, np.eye(m)))

        z = concat(np.zeros(1 + n), np.ones(m))
        A = add_first_column(A, b)
        A = add_first_row(A, z)

        Bz = np.arange(n + 1, n + m + 1)
        print('First A.shape', A.shape)

        # уберем единицы в 1 строке
        A[0] -= A[1: m + 1].sum(axis=0)
        if not self._has_solution(A, Bz):
            return None
        if A[0, 0] < -self.eps:
            return None
        # set invariant B_z
        for b_ind, s in enumerate(Bz):
            if n < s:
                r = single_one_index(A[1:m + 1, s]) + 1

                k = first_true_or_none(A[r, 1:n + 1] != 0)

                if k is None:
                    A = np.delete(A, r, axis=0)
                    Bz = np.delete(Bz, r - 1)
                else:
                    self._gauss(A, r, k + 1)
                    Bz[b_ind] = k + 1

        A = add_first_row(A[1:, 0:n + 1], concat([0], -c))

        for i, s in enumerate(Bz, start=1):
            if A[0, s] != 0:
                A[0] -= A[i] * A[0, s]

        # second phase
        if not self._has_solution(A, Bz):
            return None
        solution = np.zeros(n)
        for i, s in enumerate(Bz, start=1):
            solution[s - 1] = A[i, 0]

        return SimplexResult(A[0, 0], solution[:-issue.additional_vars_count])


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

        print('calling with params', params)
        res = opt.linprog(**params)

        if not res.success:
            return None
        else:
            return SimplexResult(x=res.fun, solution=np.copy(res.x))

    def resolve_max(self, issue: SimplexCanonicalIssue) -> Optional[SimplexResult]:
        res = self.resolve_min(issue.with_negate_function())

        if res is not None:
            return res.negated()
        return None
