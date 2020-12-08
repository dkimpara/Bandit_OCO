import numpy as np
from numpy import linalg
import generator as gen


def gradient_descent_experiment(A, alpha, n, d, sigma, iters=100, projected=False):
    tracking_error = []

    if projected:  # pgd with sigma=1
        xstar_gen = gen.xstar2(d)
    else:  # gd with variable sigma
        xstar_gen = gen.xstar1(sigma, d)

    b_gen = gen.generate_bt(A, n, xstar_gen)
    x_t = np.ones(d)
    b_t, xstar_t = next(b_gen)

    tracking_error.append(linalg.norm(x_t - xstar_t))

    for i in range(iters):
        b_t, xstar_t = next(b_gen)
        x_t = x_t - alpha * gradient(A, x_t, b_t)
        if projected:  # do projection step in pgd
            x_t = project_unit_ball(x_t)
        tracking_error.append(linalg.norm(x_t - xstar_t))

    return tracking_error


def gradient(A, x_t, b_t):
    return A.T @ (A @ x_t - b_t)


def project_unit_ball(x):
    norm = linalg.norm(x)
    if norm > 1.0:
        return x / norm
    else:
        return x
