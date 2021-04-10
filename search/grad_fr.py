import numpy as np

from core.function import Function
from search.grad import step_func_one_dimensional_method, fibonacci


def gradient_fr(f: Function, w0, step_fun=step_func_one_dimensional_method(fibonacci), eps=1e-8, max_iter=1e5, ):
    w = w0
    iteration = 0
    path = [w]
    p = -f.grad(w)

    while True:
        b = step_fun(f.f, -p, w)
        next_x = w + b * p
        path.append(next_x)
        p = -f.grad(next_x) + p * (np.linalg.norm(f.grad(next_x)) / np.linalg.norm(f.grad(w))) ** 2

        # if abs(f(next_x) - f(x)) < eps or np.linalg.norm(p) < eps or iteration >= max_iter:
        if abs(f.f(next_x) - f.f(w)) < eps or iteration >= max_iter:
            return w, iteration, path
        w = next_x
        iteration += 1
