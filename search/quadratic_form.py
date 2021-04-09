import matplotlib.pyplot as plt
import numpy as np

from core.extended_functions import QuadraticFunction
from search.grad import GradDescent, DividePrevStrategy


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


def draw_util(n, points):
    fig, ax = plt.subplots()
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=10, c="black", edgecolors="black")
    # ax.plot(xs, ys, c="black")

    ax.set_title(f'n={n}', fontsize=16)
    ax.set_xlabel("lambda", fontsize=12)
    ax.set_ylabel("iterations", fontsize=12)
    plt.show()


def draw(n: int, max_l=50, iterations=500):
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
            print(f'#{i} l - {l} iterations - {iters}')
    points.sort(key=lambda pair: pair[0])
    draw_util(n, points)


draw(4)
