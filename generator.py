import numpy as np
from numpy import linalg

from scipy.stats import ortho_group
# Return a random orthogonal matrix, drawn from the O(N) Haar distribution (the only uniform distribution on O(N)).'''


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
    '''generator for x_t^* for q1'''
    x = np.zeros(d)
    while True:
        yield x
        x += sigma * sample_n_sphere_surface(d)


def xstar2(d):
    '''generator for x_t^* for question 2'''
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