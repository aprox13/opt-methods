from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from core.extended_functions import QuadraticFunction
from search.grad import GradDescent, DividePrevStrategy
from utils.matplotlib_utils import draw_grid


def get_point(matrix: np.ndarray, b: np.ndarray, c: float, x: np.ndarray) -> int:
    f = QuadraticFunction(matrix, b, c)

    _, iterations = GradDescent().search_with_iterations(f,
                                                         x,
                                                         step_strategy=DividePrevStrategy(f),
                                                         stop_criterion="func_margin",
                                                         eps=1e-3)
    return iterations


def gen_positive_matrix(n: int) -> np.ndarray:
    while True:
        matrix = np.random.random((n, n))
        if np.all(np.linalg.eigvals(matrix) > 0):
            return matrix


def draw_test_grad_util(n: int, points: List[Tuple[float, int]], ax):
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=10, c="black", edgecolors="black")

    ax.set_title(f'n={n}', fontsize=16)
    ax.set_xlabel("lambda", fontsize=12)
    ax.set_ylabel("iterations", fontsize=12)


def draw_one_test_grad(n: int, max_l=50, iterations=500):
    def _draw(ax):
        points = list()
        for i in range(1, iterations):
            matrix = gen_positive_matrix(n)
            b = np.random.random(n) - 0.5
            x = np.random.random(n)
            lambdas = np.abs(np.linalg.eigvals(matrix))
            l = max(lambdas) / min(lambdas)
            if l <= max_l:
                iters = get_point(matrix, b, 0.5, x)
                points.append((l, iters))
        points.sort(key=lambda pair: pair[0])
        draw_test_grad_util(n, points, ax)
    return _draw


def draw_test_grad():
    data = [draw_one_test_grad(n) for n in range(2, 6)]
    draw_grid(data, ncols=2)

draw_test_grad()
