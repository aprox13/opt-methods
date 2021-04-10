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


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def make_pos_def(x):
    l = np.linalg.eigvals(x)
    mu = (-l + 1) * np.eye(x.shape[0])
    return x + mu


def newton(f: Function, w0, eps=1e-8, stop_criterion=stop_criterion_dx):
    w0 = np.array(w0.copy(), np.float64)
    w = np.array(w0.copy(), np.float64)
    iterations = 0
    path = [w0]
    for _ in range(10000):
        grad = f.grad(w)
        hess = f.hessian(w)

        if not is_pos_def(hess):
            hess = make_pos_def(hess)  # метод Марквардта

        delta_w = get_d(hess, grad)
        new_w = w - delta_w
        if stop_criterion(new_w, w, eps):
            return w, iterations, path
        path.append(new_w.copy())
        w = new_w
        iterations += 1
    return w, iterations, path
