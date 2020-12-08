import numpy as np
from numpy import linalg
import generator as gen


#run experiment over sequence of loss function objects

def gradient_descent(loss_seq, projection, alpha):
    tracking_error = []
    regret = []

    loss = loss_seq[0]
    x_t = np.ones(loss.d)

    tracking_error.append(loss.tracking_error(x_t))
    regret.append(loss.regret(x_t))

    for loss in loss_seq[1:]:
        x_t = x_t - alpha * loss.gradient(x_t)
        x_t = projection(x_t)

        tracking_error.append(loss.tracking_error(x_t))
        regret.append(loss.regret(x_t))

    return tracking_error, regret


def bandit_descent(loss_seq, projection, alpha, delta, xi):
    tracking_error = []
    regret = []

    loss = loss_seq[0]
    x_t = np.ones(loss.d)

    tracking_error.append(loss.tracking_error(x_t))
    regret.append(loss.regret(x_t))

    for loss in loss_seq[1:]:
        x_t = x_t - alpha * d1_point_gradient(x_t, delta, loss)
        x_t = projection(x_t, xi)  # project onto set scaled by xi

        tracking_error.append(loss.tracking_error(x_t))
        regret.append(loss.regret(x_t))

    return tracking_error, regret


def d1_point_gradient(x_t, delta, loss):
    d = loss.d

    g = np.zeros(d)
    lxt = loss.evaluate(x_t)
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        g += (loss.evaluate(x_t + delta * e_i) - lxt) * e_i
    return 1 / delta * g


def k_point_gradient(x_t, delta, d, loss):
    pass


def project_ball(x, radius=1):
    norm = linalg.norm(x)
    if norm > radius:
        return x / norm * radius
    else:
        return x
