from __future__ import division
import numpy as np
import numpy.linalg as la
import pandas as pd
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
    """Direction of arrival in 2D.

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
    """Radar measurements."""
    if not dx:
        return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])
    else:
        return np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [np.sin(x[1]), x[0] * np.cos(x[1])]])


def kl_div(mu0, sig0, mu1, sig1):
    """KL divergence between two Gaussians. """
    k = 1 if np.isscalar(mu0) else mu0.shape[0]
    sig0, sig1 = np.atleast_2d(sig0, sig1)
    dmu = mu1 - mu0
    dmu = np.asarray(dmu)
    det_sig0 = np.linalg.det(sig0)
    det_sig1 = np.linalg.det(sig1)
    inv_sig1 = np.linalg.inv(sig1)
    kl = 0.5 * (np.trace(np.dot(inv_sig1, sig0)) + np.dot(dmu.T, inv_sig1).dot(dmu) + np.log(det_sig1 / det_sig0) - k)
    return np.asscalar(kl)


def kl_div_sym(mu0, sig0, mu1, sig1):
    """Symmetrized KL divergence."""
    return 0.5 * (kl_div(mu0, sig0, mu1, sig1) + kl_div(mu1, sig1, mu0, sig0))


def taylor_gpqd_demo(f):
    """Compares performance of GPQ+D-RBF transform w/ finite lengthscale and Linear transform."""
    d = 2  # dimension
    ut_pts = Unscented.unit_sigma_points(d)
    gh_pts = GaussHermite.unit_sigma_points(d, 5)
    # function to test on
    # f = toa  # sum_of_squares
    transforms = (
        Taylor1stOrder(d),
        TaylorGPQD(d, alpha=1.0, el=1.0),
        GPQuad(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8}),
        GPQuadDerRBF(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
                     which_der=np.arange(ut_pts.shape[1])),
        Unscented(d, kappa=0.0),
        # MonteCarlo(d, n=int(1e4)),
    )
    mean = np.array([3, 0])
    cov = np.array([[1, 0],
                    [0, 10]])
    for ti, t in enumerate(transforms):
        mean_f, cov_f, cc = t.apply(f, mean, cov, None)
        print "{}: mean: {}, cov: {}".format(t.__class__.__name__, mean_f, cov_f)


def gpq_int_var_demo():
    """Compares integral variances of GPQ and GPQ+D by plotting."""
    d = 1
    ut_pts = Unscented.unit_sigma_points(d)
    f = UNGM().dyn_eval
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


def gpq_kl_demo():
    """Compares moment transforms in terms of symmetrized KL divergence."""

    # input dimension
    d = 2
    # unit sigma-points
    pts = Unscented.unit_sigma_points(d)
    # derivative mask, which derivatives to use
    dmask = np.arange(pts.shape[1])
    # RBF kernel hyper-parameters
    hyp = {'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8}
    # tested moment trasforms
    transforms = (
        GPQuad(d, pts, hyp),
        GPQuadDerRBF(d, pts, hyp, dmask),
        Unscented(d),
    )
    # baseline: Monte Carlo transform w/ 10,000 samples
    mc_baseline = MonteCarlo(d, n=1e4)
    # tested functions
    test_functions = (sum_of_squares, toa, rss, radar)
    # moments of the input Gaussian density
    mean = np.zeros(d)
    cov_samples = 50
    # space allocation for KL divergence
    kl_data = np.zeros((len(transforms), len(test_functions), cov_samples))
    for idt, t in enumerate(transforms):
        for idf, f in enumerate(test_functions):
            for i in range(cov_samples):
                # random PD matrix
                a = np.random.randn(d, d)
                cov = a.dot(a.T)
                # baseline moments using Monte Carlo
                mean_mc, cov_mc, cc = mc_baseline.apply(f, mean, cov, None)
                # apply transform
                mean_t, cov_t, cc = t.apply(f, mean, cov, None)
                kl_data[idt, idf, i] = kl_div_sym(mean_mc, cov_mc, mean_t, cov_t)
    # average over MC samples
    kl_data = kl_data.mean(axis=2)
    # put into pandas dataframe for nice printing and latex output


gpq_kl_demo()
