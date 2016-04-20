from __future__ import division
import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.lines import Line2D

"""Demo showcasing the GP regression with derivative observations."""


def maha(x, y, V=None):
    """Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.

    :param x: (n, d) matrix of row vectors
    :param y: (n, d) matrix of row vectors
    :param V: weight matrix (d, d), if V=None, V=eye(d) is used
    :return:
    """
    if V is None:
        V = np.eye(x.shape[1])
    x2V = np.sum(x.dot(V) * x, 1)
    y2V = np.sum(y.dot(V) * y, 1)
    return (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)


def kern_rbf_der(xs, x, alpha=10.0, el=0.7, which_der=None):
    """RBF kernel w/ derivatives."""
    x, xs = np.atleast_2d(x), np.atleast_2d(xs)
    D, N = x.shape
    Ds, Ns = xs.shape
    assert Ds == D
    which_der = np.arange(N) if which_der is None else which_der
    Nd = len(which_der)  # points w/ derivative observations
    # extract hypers
    # alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
    iLam = np.diag(el ** -1 * np.ones(D))
    iiLam = np.diag(el ** -2 * np.ones(D))

    x = iLam.dot(x)  # sqrt(Lambda^-1) * X
    xs = iLam.dot(xs)
    Kff = np.exp(2 * np.log(alpha) - 0.5 * maha(xs.T, x.T))  # cov(f(xi), f(xj))
    x = iLam.dot(x)  # Lambda^-1 * X
    xs = iLam.dot(xs)
    XmX = xs[..., na] - x[:, na, :]  # pair-wise differences
    Kfd = np.zeros((Ns, D * Nd))  # cov(f(xi), df(xj))
    Kdd = np.zeros((D * Nd, D * Nd))  # cov(df(xi), df(xj))
    for i in range(Ns):
        for j in range(Nd):
            jstart, jend = j * D, j * D + D
            j_d = which_der[j]
            Kfd[i, jstart:jend] = Kff[i, j_d] * XmX[:, i, j_d]
    for i in range(Nd):
        for j in range(Nd):
            istart, iend = i * D, i * D + D
            jstart, jend = j * D, j * D + D
            i_d, j_d = which_der[i], which_der[j]  # indices of points with derivatives
            Kdd[istart:iend, jstart:jend] = Kff[i_d, j_d] * (iiLam - np.outer(XmX[:, i_d, j_d], XmX[:, i_d, j_d]))
    return Kff, Kfd, Kdd  # np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))


def f(x):
    # return np.sin(x)*np.cos(x)**2
    return np.asarray([0.5 * x + 25 * (x / (1 + x ** 2))]).squeeze()


def df(x):
    # return np.cos(x)**3 - 2*np.sin(x)**2*np.cos(x)
    return np.asarray([0.5 + 25 * (1 - x ** 2) / (1 + x ** 2) ** 2]).squeeze()


xs = np.linspace(-3, 3, 50)  # test set
fx = f(xs)
xtr = np.array([0, -1, 1], dtype=float)  # train set
ytr = f(xtr)  # function observations + np.random.randn(xtr.shape[0])
dtr = df(xtr)  # derivative observations
y = np.hstack((ytr, dtr))
m, n = len(xs), len(xtr)  # # train and test points
jitter = 1e-8

# evaluate kernel matrices
kss, kfd, kdd = kern_rbf_der(xs, xs)
kff, kfd, kdd = kern_rbf_der(xs, xtr)
kfy = np.hstack((kff, kfd))
Kff, Kfd, Kdd = kern_rbf_der(xtr, xtr)
K = np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))
# GP fit w/ function values only
kff_iK = cho_solve(cho_factor(Kff + jitter * np.eye(n)), kff.T).T
gp_mean = kff_iK.dot(ytr)
gp_var = np.diag(kss - kff_iK.dot(kff.T))
gp_std = np.sqrt(gp_var)
# GP fit w/ functionn values and derivatives
kfy_iK = cho_solve(cho_factor(K + jitter * np.eye(n + n * 1)), kfy.T).T  # kx.dot(inv(K))
gp_mean_d = kfy_iK.dot(y)
gp_var_d = np.diag(kss - kfy_iK.dot(kfy.T))
gp_std_d = np.sqrt(gp_var_d)

# setup plotting
fmin, fmax, fp2p = np.min(fx), np.max(fx), np.ptp(fx)
axis_limits = [-3, 3, fmin - 0.2 * fp2p, fmax + 0.2 * fp2p]
tick_settings = {'which': 'both', 'bottom': 'off', 'top': 'off', 'left': 'off', 'right': 'off', 'labelleft': 'off',
                 'labelbottom': 'off'}
# use tex to render text in the figure
mpl.rc('text', usetex=True)
# use lmodern font package which is also used in the paper
mpl.rc('text.latex', preamble=['\usepackage{lmodern}'])
# sans serif font for figure, size 10pt
mpl.rc('font', family='sans-serif', size=10)
plt.style.use('seaborn-paper')
# set figure width to fit the column width of the article
pti = 1.0 / 72.0
fig_width_pt = 244
golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
fig_w = fig_width_pt * pti * 1.0
fig_h = fig_w * golden_mean
plt.figure(figsize=(fig_w, fig_h))
# plot ordinary GP regression fit
plt.subplot(211)
plt.axis(axis_limits)
plt.tick_params(**tick_settings)
plt.title('GP regression')
plt.plot(xs, fx, 'r--', label='true')
plt.plot(xtr, ytr, 'ko', ms=10, label='observed fcn values')
plt.plot(xs, gp_mean, 'k-', lw=2, label='GP mean')
plt.fill_between(xs, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, color='k', alpha=0.15)
# plot GP regression fit w/ derivative observations
plt.subplot(212)
plt.axis(axis_limits)
plt.tick_params(**tick_settings)
plt.title('GP regression with gradient observations')
plt.plot(xs, fx, 'r--', label='true')
plt.plot(xtr, ytr, 'ko', ms=10, label='observed fcn values')
plt.plot(xs, gp_mean_d, 'k-', lw=2, label='GP mean')
plt.fill_between(xs, gp_mean_d - 2 * gp_std_d, gp_mean_d + 2 * gp_std_d, color='k', alpha=0.15)
# plot line segments to indicate derivative observations
h = 0.15
for i in range(len(dtr)):
    x0, x1 = xtr[i] - h, xtr[i] + h
    y0 = dtr[i] * (x0 - xtr[i]) + ytr[i]
    y1 = dtr[i] * (x1 - xtr[i]) + ytr[i]
    plt.gca().add_line(Line2D([x0, x1], [y0, y1], linewidth=6, color='k'))
plt.tight_layout()
plt.savefig('gpr_grad_compar.pdf', format='pdf')

# two figure version
scale = 0.5
fig_w = fig_width_pt * pti
fig_h = fig_w * golden_mean * scale
# plot ordinary GP regression fit
plt.figure(figsize=(fig_w, fig_h))
plt.axis(axis_limits)
plt.tick_params(**tick_settings)
plt.plot(xs, fx, 'r--', label='true')
plt.plot(xtr, ytr, 'ko', ms=10, label='observed fcn values')
plt.plot(xs, gp_mean, 'k-', lw=2, label='GP mean')
plt.fill_between(xs, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, color='k', alpha=0.15)
plt.tight_layout(pad=0.5)
plt.savefig('gpr_fcn_obs.pdf', format='pdf')
# plot GP regression fit w/ derivative observations
plt.figure(figsize=(fig_w, fig_h))
plt.axis(axis_limits)
plt.tick_params(**tick_settings)
plt.plot(xs, fx, 'r--', label='true')
plt.plot(xtr, ytr, 'ko', ms=10, label='observed fcn values')
plt.plot(xs, gp_mean_d, 'k-', lw=2, label='GP mean')
plt.fill_between(xs, gp_mean_d - 2 * gp_std_d, gp_mean_d + 2 * gp_std_d, color='k', alpha=0.15)
# plot line segments to indicate derivative observations
h = 0.15
for i in range(len(dtr)):
    x0, x1 = xtr[i] - h, xtr[i] + h
    y0 = dtr[i] * (x0 - xtr[i]) + ytr[i]
    y1 = dtr[i] * (x1 - xtr[i]) + ytr[i]
    plt.gca().add_line(Line2D([x0, x1], [y0, y1], linewidth=6, color='k'))
plt.tight_layout(pad=0.5)
plt.savefig('gpr_grad_obs.pdf', format='pdf')



# fig = plt.figure()
# ax = fig.add_axes()
# ax.add_subplot
