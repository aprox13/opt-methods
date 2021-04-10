import pytest

from simplex.sample_registry import IssueRegistry
from simplex.solver import *
from utils import *

np.seterr(divide='ignore', invalid='ignore')

registry = IssueRegistry.load('test_data.json')

LAB_ISSUES = [registry[f'Lab example #{i}'] for i in range(1, 7 + 1)]

TEST_EPS = 1e-10
SCIPY_SOLVER = ScipySimplexSolver(TEST_EPS)


def compare_with_scipy_solver_max(issue: SimplexCanonicalIssue, target):
    eps = SCIPY_SOLVER.eps

    def target_fun(_solver: SimplexSolver):
        if target == 'min':
            return _solver.resolve_min(issue)
        return _solver.resolve_max(issue)

    scipy_solver_res = target_fun(SCIPY_SOLVER)
    solver_res = target_fun(SimplexV2(eps))

    assert not ((scipy_solver_res is None) ^ (solver_res is None))

    if scipy_solver_res is not None:
        assert eq_tol(solver_res.x, scipy_solver_res.x, tol=eps)


@pytest.mark.parametrize("issue,target", LAB_ISSUES)
def test_simplex_solver_impl_lab_examples(issue: SimplexCanonicalIssue, target):
    compare_with_scipy_solver_max(issue, target)
