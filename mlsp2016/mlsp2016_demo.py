from __future__ import division
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from numpy import newaxis as na
from transforms.taylor import Taylor1stOrder, TaylorGPQD
from transforms.quad import MonteCarlo, Unscented, GaussHermite, SphericalRadial
from transforms.bayesquad import GPQuad, GPQuadDerRBF
from models.ungm import UNGM
from scipy.stats import norm
from numba import jit


def sos(x, pars, dx=False):
    """Sum of squares function.

    Parameters
    ----------
    x : numpy.ndarray 1D-array

    Returns
    -------

    """
    x = np.atleast_1d(x)
    if not dx:
        return np.atleast_1d(np.sum(x ** 2, axis=0))
    else:
        return np.atleast_1d(2 * x).T.flatten()


def toa(x, pars, dx=False):
    """Time of arrival.

    Parameters
    ----------
    x

    Returns
    -------

    """
    x = np.atleast_1d(x)
    if not dx:
        return np.atleast_1d(np.sum(x ** 2, axis=0) ** 0.5)
    else:
        return np.atleast_1d(x * np.sum(x ** 2, axis=0) ** (-0.5)).T.flatten()


def rss(x, pars, dx=False):
    """Received signal strength in dB scale.

    Parameters
    ----------
    x : N-D ndarray

    Returns
    -------

    """
    c = 10
    b = 2
    x = np.atleast_1d(x)
    if not dx:
        return np.atleast_1d(c - b * 10 * np.log10(np.sum(x ** 2, axis=0)))
    else:
        return np.atleast_1d(-b * 20 / (x * np.log(10))).T.flatten()


def doa(x, pars, dx=False):
    """Direction of arrival in 2D.

    Parameters
    ----------
    x : 2-D ndarray

    Returns
    -------

    """
    if not dx:
        return np.atleast_1d(np.arctan2(x[1], x[0]))
    else:
        return np.array([-x[1], x[0]]) / (x[0] ** 2 + x[1] ** 2).T.flatten()


def rdr(x, pars, dx=False):
    """Radar measurements in 2D."""
    if not dx:
        return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])
    else:  # TODO: returned jacobian must be properly flattened, see dyn_eval in ssm
        return np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [np.sin(x[1]), x[0] * np.cos(x[1])]]).T.flatten()


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


def rel_error(mu_true, mu_approx):
    """Relative error."""
    assert mu_true.shape == mu_approx.shape
    return la.norm((mu_true - mu_approx) / mu_true)


def plot_func(f, d, n=100, xrng=(-3, 3)):
    xmin, xmax = xrng
    x = np.linspace(xmin, xmax, n)
    assert d <= 2, "Dimensions > 2 not supported. d={}".format(d)
    if d > 1:
        X, Y = np.meshgrid(x, x)
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Z[i, j] = f([X[i, j], Y[i, j]], None)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.5, linewidth=0.75)
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.viridis)
        ax.contour(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.viridis)
        ax.contour(X, Y, Z, zdir='y', offset=np.max(Y), cmap=cm.viridis)
        plt.show()
    else:
        y = np.zeros(n)
        for i in range(n):
            y[i] = f(x[i], None)
        fig = plt.plot(x, y)
        plt.show()
    return fig


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
    gpq = GPQuad(d, unit_sp=ut_pts, hypers={'sig_var': 10.0, 'lengthscale': 0.7 * np.ones(d), 'noise_var': 1e-8})
    gpqd = GPQuadDerRBF(d, unit_sp=ut_pts, hypers={'sig_var': 10.0, 'lengthscale': 0.7 * np.ones(d), 'noise_var': 1e-8},
                        which_der=np.arange(ut_pts.shape[1]))
    mct = MonteCarlo(d, n=1e4)
    mean_gpq, cov_gpq, cc_gpq = gpq.apply(f, mean, cov, np.atleast_1d(1.0))
    mean_gpqd, cov_gpqd, cc_gpqd = gpqd.apply(f, mean, cov, np.atleast_1d(1.0))
    mean_mc, cov_mc, cc_mc = mct.apply(f, mean, cov, np.atleast_1d(1.0))

    xmin_gpq = norm.ppf(0.0001, loc=mean_gpq, scale=gpq.integral_var)
    xmax_gpq = norm.ppf(0.9999, loc=mean_gpq, scale=gpq.integral_var)
    xmin_gpqd = norm.ppf(0.0001, loc=mean_gpqd, scale=gpqd.integral_var)
    xmax_gpqd = norm.ppf(0.9999, loc=mean_gpqd, scale=gpqd.integral_var)
    xgpq = np.linspace(xmin_gpq, xmax_gpq, 500)
    ygpq = norm.pdf(xgpq, loc=mean_gpq, scale=gpq.integral_var)
    xgpqd = np.linspace(xmin_gpqd, xmax_gpqd, 500)
    ygpqd = norm.pdf(xgpqd, loc=mean_gpqd, scale=gpqd.integral_var)
    plt.figure()
    plt.plot(xgpq, ygpq, lw=2)
    plt.plot(xgpqd, ygpqd, lw=2)
    plt.gca().add_line(Line2D([mean_mc, mean_mc], [0, 150], linewidth=2, color='k'))
    plt.show()


def gpq_kl_demo():
    """Compares moment transforms in terms of symmetrized KL divergence."""

    # input dimension
    d = 2
    # unit sigma-points
    pts = SphericalRadial.unit_sigma_points(d)
    # derivative mask, which derivatives to use
    dmask = np.arange(pts.shape[1])
    # RBF kernel hyper-parameters
    hyp = {
        'sos': {'sig_var': 10.0, 'lengthscale': 6.0 * np.ones(d), 'noise_var': 1e-8},
        'rss': {'sig_var': 10.0, 'lengthscale': 0.2 * np.ones(d), 'noise_var': 1e-8},  # el=0.2, d=2
        'toa': {'sig_var': 10.0, 'lengthscale': 3.0 * np.ones(d), 'noise_var': 1e-8},
        'doa': {'sig_var': 1.0, 'lengthscale': 2.0 * np.ones(d), 'noise_var': 1e-8},  # al=2, np.array([2, 2])
        'rdr': {'sig_var': 1.0, 'lengthscale': 5.0 * np.ones(d), 'noise_var': 1e-8},
    }
    # baseline: Monte Carlo transform w/ 10,000 samples
    mc_baseline = MonteCarlo(d, n=2e4)
    # tested functions
    # rss has singularity at 0, therefore no derivative at 0
    # toa does not have derivative at 0, for d = 1
    # rss, toa and sos can be tested for all d > 0; physically d=2,3 make sense
    # radar and doa only for d = 2
    test_functions = (
        sos,
        toa,
        rss,
        doa,
        rdr,
    )
    # moments of the input Gaussian density
    mean = np.zeros(d)
    cov_samples = 100
    # space allocation for KL divergence
    kl_data = np.zeros((3, len(test_functions), cov_samples))
    re_data_mean = np.zeros((3, len(test_functions), cov_samples))
    re_data_cov = np.zeros((3, len(test_functions), cov_samples))
    for i in range(cov_samples):
        # random PD matrix
        a = np.random.randn(d, d)
        cov = a.dot(a.T)
        a = np.diag(1.0 / np.sqrt(np.diag(cov)))  # 1 on diagonal
        cov = a.dot(cov).dot(a.T)
        for idf, f in enumerate(test_functions):
            # print "Testing {}".format(f.__name__)
            mean[:d - 1] = 0.2 if f.__name__ in 'rss' else mean[:d - 1]
            mean[:d - 1] = 3.0 if f.__name__ in 'doa' else mean[:d - 1]
            jitter = 1e-8 * np.eye(2) if f.__name__ == 'rdr' else 1e-8 * np.eye(1)
            # baseline moments using Monte Carlo
            mean_mc, cov_mc, cc = mc_baseline.apply(f, mean, cov, None)
            # tested moment trasforms
            transforms = (
                SphericalRadial(d),
                GPQuad(d, pts, hyp[f.__name__]),
                GPQuadDerRBF(d, pts, hyp[f.__name__], dmask),
            )
            for idt, t in enumerate(transforms):
                # apply transform
                mean_t, cov_t, cc = t.apply(f, mean, cov, None)
                # calculate KL distance to the baseline moments
                kl_data[idt, idf, i] = kl_div_sym(mean_mc, cov_mc + jitter, mean_t, cov_t + jitter)
                re_data_mean[idt, idf, i] = rel_error(mean_mc, mean_t)
                re_data_cov[idt, idf, i] = rel_error(cov_mc, cov_t)
    # average over MC samples
    kl_data = kl_data.mean(axis=2)
    re_data_mean = re_data_mean.mean(axis=2)
    re_data_cov = re_data_cov.mean(axis=2)
    # put into pandas dataframe for nice printing and latex output
    row_labels = [t.__class__.__name__ for t in transforms]
    col_labels = [f.__name__ for f in test_functions]
    kl_df = pd.DataFrame(kl_data, index=row_labels, columns=col_labels)
    re_mean_df = pd.DataFrame(re_data_mean, index=row_labels, columns=col_labels)
    re_cov_df = pd.DataFrame(re_data_mean, index=row_labels, columns=col_labels)
    return kl_df, re_mean_df, re_cov_df


def gpq_hypers_demo():
    # input dimension, we can only plot d = 1
    d = 1
    # unit sigma-points
    pts = SphericalRadial.unit_sigma_points(d)
    # pts = Unscented.unit_sigma_points(d)
    # pts = GaussHermite.unit_sigma_points(d, degree=5)
    # shift the points away from the singularity
    # pts += 3*np.ones(d)[:, na]
    # derivative mask, which derivatives to use
    dmask = np.arange(pts.shape[1])
    # functions to test
    test_functions = (sos, toa, rss,)
    # RBF kernel hyper-parameters
    hyp = {
        'sos': {'sig_var': 10.0, 'lengthscale': 6.0 * np.ones(d), 'noise_var': 1e-8},
        'rss': {'sig_var': 10.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
        'toa': {'sig_var': 10.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
    }
    hypd = {
        'sos': {'sig_var': 10.0, 'lengthscale': 6.0 * np.ones(d), 'noise_var': 1e-8},
        'rss': {'sig_var': 10.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
        'toa': {'sig_var': 10.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
    }
    # GP plots
    # for f in test_functions:
    #     # TODO: plot_gp_model can be made static
    #     GPQuad(d, pts, hyp[f.__name__]).plot_gp_model(f, pts, None)
    # GP plots with derivatives
    for f in test_functions:
        # TODO: plot_gp_model can be made static
        GPQuadDerRBF(d, pts, hypd[f.__name__], dmask).plot_gp_model(f, pts, None)


def gpq_sos_demo():
    """Sum of squares analytical moments compared with GPQ, GPQ+D and Spherical Radial transforms."""
    # input dimension
    d = 1
    # input mean and covariance
    mean_in, cov_in = np.zeros(d), np.eye(d)
    # unit sigma-points
    pts = SphericalRadial.unit_sigma_points(d)
    # derivative mask, which derivatives to use
    dmask = np.arange(pts.shape[1])
    # RBF kernel hyper-parameters
    hyp = {
        'gpq': {'sig_var': 10.0, 'lengthscale': 6.0 * np.ones(d), 'noise_var': 1e-8},
        'gpqd': {'sig_var': 10.0, 'lengthscale': 6.0 * np.ones(d), 'noise_var': 1e-8},
    }
    transforms = {
        SphericalRadial(d),
        GPQuad(d, pts, hyp['gpq']),
        GPQuadDerRBF(d, pts, hyp['gpqd'], dmask),
    }
    mean_true, cov_true = d, 2 * d
    print "{:<15}:\t {:.4f} \t{:.4f}".format("True moments", mean_true, cov_true)
    for t in transforms:
        m, c, cc = t.apply(sos, mean_in, cov_in, None)
        print "{:<15}:\t {:.4f} \t{:.4f}".format(t.__class__.__name__, np.asscalar(m), np.asscalar(c))


gpq_sos_demo()
# gpq_int_var_demo()

# fig = plot_func(rss, 2, n=100)

# kl_tab, re_mean_tab, re_cov_tab = gpq_kl_demo()
# pd.set_option('display.float_format', '{:.2e}'.format)
# print "\nSymmetrized KL-divergence"
# print kl_tab
# print "\nRelative error in the mean"
# print re_mean_tab
# print "\nRelative error in the covariance"
# print re_cov_tab
# fo = open('kl_div_table.tex', 'w')
# table.T.to_latex(fo)
# fo.close()
