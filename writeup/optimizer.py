import numpy as np
from numpy import linalg
import generator as gen


#run experiment over sequence of loss function objects

def gradient_descent(loss_seq, projection, alpha):
    tracking_error = []
    regret = []

    x_t = np.ones(loss_seq[0].d)

    for loss in loss_seq:
        tracking_error.append(loss.tracking_error(x_t))
        regret.append(loss.regret(x_t))
        x_t = x_t - alpha * loss.gradient(x_t)
        x_t = projection(x_t)

    return tracking_error, regret


def bandit_descent(loss_seq, projection, alpha, delta, xi):
    tracking_error = []
    regret = []

    x_t = np.ones(loss_seq[0].d)  # init play

    for loss in loss_seq:
        tracking_error.append(loss.tracking_error(x_t))
        gradient, d1_loss = d1_point_gradient_loss(x_t, delta, loss)
        regret.append(d1_loss - loss.evaluate(loss.xstar))

        x_t = x_t - alpha * gradient
        x_t = projection(x_t, xi)  # project onto set scaled by xi

    return tracking_error, regret


def d1_point_gradient_loss(x_t, delta, loss):
    d = loss.d

    g = np.zeros(d)
    reg_query = 0

    lxt = loss.evaluate(x_t)
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        point_loss = loss.evaluate(x_t + delta * e_i)
        reg_query += point_loss
        g += (point_loss - lxt) * e_i
    return 1 / delta * g, 1 / (d+1) * (reg_query + lxt)


def k_point_gradient(x_t, delta, d, loss):
    pass


def project_ball(x, radius=1):
    norm = linalg.norm(x)
    if norm > radius:
        return x / norm * radius
    else:
        return x
