import numpy as np
from numpy import linalg

from scipy.stats import ortho_group
# Return a random orthogonal matrix, drawn from the O(N) Haar distribution (the only uniform distribution on O(N)).'''


class ls_loss:
    def __init__(self, A, b_t, xstar, d):
        self.A = A
        self.b = b_t
        self.xstar = xstar
        self.d = d

    def evaluate(self, x):
        A = self.A
        b_t = self.b
        return 0.5 * linalg.norm(A * x - b_t) ** 2

    def gradient(self, x):
        A = self.A
        b_t = self.b
        return A.T @ (A @ x - b_t)

    def tracking_error(self, x):
        return linalg.norm(x - self.xstar)

    def regret(self, x):
        return self.evaluate(x) - self.evaluate(self.xstar)


def generate_ls_seq(n, d, xstar_gen, iters = 500):
    #  generate list containing sequence of loss objects
    # ball xstar_gen = xstar2(d)

    losses = []
    A = generate_A(n, d)
    b_gen = generate_bt(A, n, xstar_gen)  #generator for b

    for i in range(iters):
        b_t, xstar_t = next(b_gen)
        losses.append(ls_loss(A, b_t, xstar_t, d))
    return losses


def generate_A(n, d):
    U = ortho_group.rvs(n)
    V = ortho_group.rvs(d)
    D = np.diagflat(np.flip(np.linspace(1 / np.sqrt(100), 1, min(n, d))))
    D = np.vstack((D, np.zeros([n - d, d])))
    return U @ D @ V


def generate_bt(A, n, x_gen):
    '''generator for b_t'''
    while True:
        xstar = next(x_gen)
        w = np.random.normal(0, 10 ** (-3), n)
        yield A @ xstar + w, xstar


def xstar1(sigma, d):
    '''generator for x_t^* for unconstrained'''
    x = np.zeros(d)
    while True:
        yield x
        x += sigma * sample_n_sphere_surface(d)


def xstar2(d):
    '''generator for x_t^* for unit ball'''
    x = np.zeros(d)
    while True:
        yield x
        x = xstar2_helper(x, d)


def xstar2_helper(x, d):
    step = sample_n_sphere_surface(d)  # step size 1
    while linalg.norm(x + step) >= 1.0:  # resample the step
        step = sample_n_sphere_surface(d)
    return x + step


def sample_n_sphere_surface(ndim, norm_p=2):
    """sample random vector from S^n-1 with norm_p"""

    vec = np.random.randn(ndim)
    vec = vec / linalg.norm(vec, norm_p)  # create random vector with norm 1
    return vec