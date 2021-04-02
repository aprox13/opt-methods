import numpy as np

from scipy.linalg import cho_factor, cho_solve

from core.function import Function


def stop_criterion_dx(w, w0, eps):
    norm = np.linalg.norm(w - w0)
    return norm < eps


def get_d(hessian, grad: np.ndarray):
    df2_i = cho_solve(cho_factor(hessian), np.eye(len(hessian)))
    d = np.matmul(grad, df2_i)
    return d


def newton(f: Function, w0, eps=1e-9, stop_criterion=stop_criterion_dx):
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    iterations = 0
    path = [w0]
    first = True
    while first or not stop_criterion(w, w0, eps):
        first = False
        delta_w = get_d(f.hessian(w), np.array(f.grad(w)))
        w0 = w.copy()
        w -= delta_w
        path.append(w.copy())
        iterations += 1
    return w, iterations, path
