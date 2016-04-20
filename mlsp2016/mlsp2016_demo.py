from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy import newaxis as na
from transforms.taylor import Taylor1stOrder, TaylorGPQD
from transforms.quad import MonteCarlo


def sum_of_squares(x, pars, dx=False):
    """Sum of squares function.

    Parameters
    ----------
    x : numpy.ndarray 1D-array

    Returns
    -------

    """
    if not dx:
        return np.atleast_1d(x.T.dot(x))
    else:
        return np.atleast_1d(2 * x)


def toa(x, pars, dx=False):
    """Time of arrival.

    Parameters
    ----------
    x

    Returns
    -------

    """
    if not dx:
        return np.atleast_1d(x.T.dot(x) ** 0.5)
    else:
        return np.atleast_1d(x * x.T.dot(x) ** (-0.5))


def rss(x, pars, dx=False):
    """Received signal strength in 2D in dB scale.

    Parameters
    ----------
    x

    Returns
    -------

    """
    c = 10
    b = 2
    if not dx:
        return np.atleast_1d(c - b * 10 * np.log10(x.T.dot(x)))
    else:
        return np.atleast_1d(b * 10 * (2 / (x * np.log(10))))


def doa(x, pars, dx=False):
    """Direction of arrival.

    Parameters
    ----------
    x : real-valued scalar
    y : real-valued scalar

    Returns
    -------

    """
    if not dx:
        return np.atleast_1d(np.arctan2(x[1], x[0]))
    else:
        return np.array([-x[1], x[0]]) / (x[0] ** 2 + x[1] ** 2)


def radar(x, dx=False):
    if not dx:
        return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])
    else:
        return np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [np.sin(x[1]), x[0] * np.cos(x[1])]])


d = 2  # dimension
transforms = (
    Taylor1stOrder(d),
    TaylorGPQD(d, alpha=1.0, el=1.0),
    MonteCarlo(d, n=int(1e4))
)

f = toa  # sum_of_squares
mean = np.array([3, 0])
cov = np.array([[1, 0],
                [0, 10]])
for ti, t in enumerate(transforms):
    mean_f, cov_f, cc = t.apply(f, mean, cov, None)
    print "{}: mean: {}, cov: {}".format(t.__class__.__name__, mean_f, cov_f)
