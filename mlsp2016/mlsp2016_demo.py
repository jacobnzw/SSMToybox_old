from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy import newaxis as na
from transforms.taylor import Taylor1stOrder, TaylorGPQD
from transforms.quad import MonteCarlo, Unscented, GaussHermite
from transforms.bayesquad import GPQuad, GPQuadDerRBF
from models.ungm import UNGM
from scipy.stats import norm

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
ut_pts = Unscented.unit_sigma_points(d)
gh_pts = GaussHermite.unit_sigma_points(d, 5)
transforms = (
    Taylor1stOrder(d),
    TaylorGPQD(d, alpha=1.0, el=1.0),
    GPQuad(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8}),
    GPQuadDerRBF(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
                 which_der=np.arange(ut_pts.shape[1])),
    Unscented(d, kappa=0.0),
    # MonteCarlo(d, n=int(1e4)),
)

f = toa  # sum_of_squares
mean = np.array([3, 0])
cov = np.array([[1, 0],
                [0, 10]])
for ti, t in enumerate(transforms):
    mean_f, cov_f, cc = t.apply(f, mean, cov, None)
    print "{}: mean: {}, cov: {}".format(t.__class__.__name__, mean_f, cov_f)

# plot integral variance
d = 1
ut_pts = Unscented.unit_sigma_points(d)
ssm = UNGM()
f = ssm.dyn_eval
mean = np.zeros(d)
cov = np.eye(d)
gpq = GPQuad(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8})
gpqd = GPQuadDerRBF(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
                    which_der=np.arange(ut_pts.shape[1]))
mean_gpq, cov_gpq, cc_gpq = gpq.apply(f, mean, cov, np.atleast_1d(1.0))
mean_gpqd, cov_gpqd, cc_gpqd = gpqd.apply(f, mean, cov, np.atleast_1d(1.0))

plt.figure()
xmin_gpq = norm.ppf(0.0001, loc=mean_gpq, scale=gpq.integral_var)
xmax_gpq = norm.ppf(0.9999, loc=mean_gpq, scale=gpq.integral_var)
xmin_gpqd = norm.ppf(0.0001, loc=mean_gpqd, scale=gpqd.integral_var)
xmax_gpqd = norm.ppf(0.9999, loc=mean_gpqd, scale=gpqd.integral_var)
xgpq = np.linspace(xmin_gpq, xmax_gpq, 500)
ygpq = norm.pdf(xgpq, loc=mean_gpq, scale=gpq.integral_var)
xgpqd = np.linspace(xmin_gpqd, xmax_gpqd, 500)
ygpqd = norm.pdf(xgpqd, loc=mean_gpqd, scale=gpqd.integral_var)
plt.plot(xgpq, ygpq, lw=2)
plt.plot(xgpqd, ygpqd, lw=2)
plt.show()
