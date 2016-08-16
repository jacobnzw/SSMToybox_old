import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from inference.gpquad import GPQMKalman
from models.ungm import UNGM
from models.pendulum import Pendulum
from models.tracking import ReentryRadar


def param_contraction():
    """Plots SKL between successive parameter posterior approximations of GPQMKalman"""
    ssm = Pendulum()
    steps, mc = 200, 5
    # simulate system
    x, y = ssm.simulate(steps, mc)
    # set parameter prior moments
    # mean_0 = np.log([1, 10, 10, 10, 10, 10, 1, 15, 15, 1e5, 1e5, 1e5])
    # cov_0 = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1])
    alg = GPQMKalman(ssm, 'rbf-ard', 'sr')
    d, steps, mc = y.shape
    skl = np.zeros((steps, mc))
    cov_tr = skl.copy()

    for imc in range(mc):
        for i in range(steps):
            alg._time_update(i)
            mean_par_0, cov_par_0 = alg.param_mean, alg.param_cov
            alg._measurement_update(y[:, i, imc], i)
            skl[i, imc] = _skl(mean_par_0, cov_par_0, alg.param_mean, alg.param_cov)
            cov_tr[i, imc] = np.trace(alg.param_cov)
        alg.reset()

    plt.subplot(211)
    plt.title('Symmetrized KL-divergence')
    plt.plot(np.arange(steps), skl)
    plt.subplot(212)
    plt.title('Covariance trace')
    plt.plot(np.arange(steps), cov_tr)
    plt.show()


def param_optimization():
    """Given sigma-points, minimize distance between CQ and BQ weights."""
    from transforms.quad import SphericalRadial
    from transforms.bqkernel import RBFARD
    d = 2
    points_cq = SphericalRadial.unit_sigma_points(d)
    weights_cq = SphericalRadial.weights(d)
    kernel = RBFARD(d)

    # choose random starting parameters
    log_par0 = np.random.randn(d+1)

    # minimize distance between BQ and CQ weights over parameters
    opt_res = minimize(weight_norm_reg, log_par0, (kernel, points_cq, weights_cq, 2), method='BFGS')
    # convert parameters from log-space
    par_opt = np.exp(opt_res.x)

    # calculate BQ weights given optimized parameters
    q = kernel.exp_x_kx(points_cq, par_opt)
    iK = kernel.eval(points_cq, hyp=par_opt)
    Q = kernel.exp_x_kxkx(points_cq, par_opt)
    bq_weights = kernel.eval_inv_dot(points_cq, par_opt, q)

    # print results
    print("Optimized BQ params: {}".format(par_opt))
    print("Optimized BQ weights: {}".format(bq_weights))
    print("Eigvals of BQ cov weights: {}".format(la.eigvals(iK.dot(Q).dot(iK.T))))


def weight_norm(log_par, kern, points, cq_weights):
    """
    Distance between BQ and CQ weights as a function of BQ parameters.

    Parameters
    ----------
    log_par: numpy.ndarray
        BQ log-parameters
    kern: bqkernel.Kernel
        Kernel object
    points: numpy.ndarray
        Point set
    cq_weights: numpy.ndarray
        CQ weights
    Returns
    -------

    """
    par = np.exp(log_par)
    q = kern.exp_x_kx(points, par)
    return la.norm(kern.eval_inv_dot(points, par, q) - cq_weights)


def weight_norm_reg(log_par, kern, points, cq_weights, p=2):
    """
    Distance between BQ and CQ weights as a function of BQ parameters.

    Parameters
    ----------
    log_par: numpy.ndarray
        BQ log-parameters
    kern: bqkernel.Kernel
        Kernel object
    points: numpy.ndarray
        Point set
    cq_weights: numpy.ndarray
        CQ weights
    p:
        Order of the norm as in numpy.linalg.norm
    Returns
    -------

    """
    par = np.exp(log_par)
    q = kern.exp_x_kx(points, par)
    return la.norm(kern.eval_inv_dot(points, par, q) - cq_weights) + 0.05*la.norm(log_par, p)


def weight_norm_ivar(log_par, kern, points, cq_weights):
    """
    Distance between BQ and CQ weights as a function of BQ parameters.

    Parameters
    ----------
    log_par: numpy.ndarray
        BQ log-parameters
    kern: bqkernel.Kernel
        Kernel object
    points: numpy.ndarray
        Point set
    cq_weights: numpy.ndarray
        CQ weights
    Returns
    -------

    """
    par = np.exp(log_par)
    q = kern.exp_x_kx(points, par)
    iK = la.inv(kern.eval(points, hyp=par))
    bq_weights = iK.dot(q)

    k_bar = kern.exp_xy_kxy(par)
    integral_variance = k_bar - q.dot(bq_weights)

    Q = kern.exp_x_kxkx(points, par)
    q_bar = kern.exp_x_kxx(par)
    exp_model_variance = q_bar - np.trace(iK.dot(Q))
    return la.norm(bq_weights - cq_weights) + integral_variance  # + 0.05*la.norm(par, 2)


def _kl(mean_0, cov_0, mean_1, cov_1):
    """
    KL-divergence

    Parameters
    ----------
    mean_0
    cov_0
    mean_1
    cov_1

    Returns
    -------

    """
    k = 1 if np.isscalar(mean_0) else mean_0.shape[0]
    cov_0, cov_1 = np.atleast_2d(cov_0, cov_1)
    dmu = mean_0 - mean_1
    dmu = np.asarray(dmu)
    det_0 = np.linalg.det(cov_0)
    det_1 = np.linalg.det(cov_1)
    inv_1 = np.linalg.inv(cov_1)
    kl = 0.5 * (np.trace(np.dot(inv_1, cov_0)) + np.dot(dmu.T, inv_1).dot(dmu) + np.log(det_0 / det_1) - k)
    return np.asscalar(kl)


def _skl(mean_0, cov_0, mean_1, cov_1):
    """
    Symmetrized KL-divergence.

    Parameters
    ----------
    mean_0
    cov_0
    mean_1
    cov_1

    Returns
    -------
    """
    return 0.5 * (_kl(mean_0, cov_0, mean_1, cov_1) + _kl(mean_1, cov_1, mean_0, cov_0))


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    param_contraction()
    # param_optimization()
