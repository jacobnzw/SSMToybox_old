from transforms.bayesquad import GPQ
from transforms.quad import MonteCarlo, SphericalRadial
from utils import *
import numpy.linalg as la
import matplotlib.pyplot as plt


"""
Gaussian Process Quadrature moment transformation tested on a mapping from polar to cartesian coordinates.
"""


def polar2cartesian(x, pars, dx=False):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def ellipse_points(x0, P):
    # x0 center, P SPD matrix
    w, v = la.eig(P)
    theta = np.linspace(0, 2 * np.pi)
    t = np.asarray((np.cos(theta), np.sin(theta)))
    return x0[:, na] + np.dot(v, np.sqrt(w[:, na]) * t)


def gpq_polar2cartesian_demo():
    dim = 2

    # Initialize transforms
    # high el[0], because the function is linear given x[1]
    tf_gpq = GPQ(dim, 'rbf', 'sr', {'alpha': 1.0, 'el': [600, 6]})
    tf_sr = SphericalRadial(dim)
    tf_mc = MonteCarlo(dim, n=1e4)  # 10k samples

    # Input mean and covariance
    mean_in = np.array([1, np.pi / 2])
    cov_in = np.diag([0.05 ** 2, (np.pi / 10) ** 2])
    # mean_in = np.array([10, 0])
    # cov_in = np.diag([0.5**2, (5*np.pi/180)**2])

    # Mapped samples
    x = np.random.multivariate_normal(mean_in, cov_in, size=int(1e3)).T
    fx = np.apply_along_axis(polar2cartesian, 0, x, None)

    # MC transformed moments
    mean_mc, cov_mc, cc_mc = tf_mc.apply(polar2cartesian, mean_in, cov_in, None)
    ellipse_mc = ellipse_points(mean_mc, cov_mc)

    # GPQ transformed moments with ellipse points
    mean_gpq, cov_gpq, cc = tf_gpq.apply(polar2cartesian, mean_in, cov_in, None)
    ellipse_gpq = ellipse_points(mean_gpq, cov_gpq)

    # SR transformed moments with ellipse points
    mean_sr, cov_sr, cc = tf_sr.apply(polar2cartesian, mean_in, cov_in, None)
    ellipse_sr = ellipse_points(mean_sr, cov_sr)

    # Plots
    plt.figure()

    # MC ground truth mean w/ covariance ellipse
    plt.plot(mean_mc[0], mean_mc[1], 'ro', markersize=6, lw=2)
    plt.plot(ellipse_mc[0, :], ellipse_mc[1, :], 'r--', lw=2, label='MC')

    # GPQ transformed mean w/ covariance ellipse
    plt.plot(mean_gpq[0], mean_gpq[1], 'go', markersize=6)
    plt.plot(ellipse_gpq[0, :], ellipse_gpq[1, :], color='g', label='GPQ')

    # SR transformed mean w/ covariance ellipse
    plt.plot(mean_sr[0], mean_sr[1], 'bo', markersize=6)
    plt.plot(ellipse_sr[0, :], ellipse_sr[1, :], color='b', label='SR')

    # Transformed samples of the input random variable
    plt.plot(fx[0, :], fx[1, :], 'k.', alpha=0.15)
    plt.axes().set_aspect('equal')
    plt.legend()
    plt.show()

    np.set_printoptions(precision=2)
    print("GPQ")
    print("Mean weights: {}".format(tf_gpq.wm))
    print("Cov weight matrix eigvals: {}".format(la.eigvals(tf_gpq.Wc)))
    print("Integral variance: {:.2e}".format(tf_gpq.model.integral_variance(None)))
    print("Expected model variance: {:.2e}".format(tf_gpq.model.exp_model_variance(None)))
    print("SKL Score:")
    print("SR: {:.2e}".format(skl(mean_mc, cov_mc, mean_sr, cov_sr)))
    print("GPQ: {:.2e}".format(skl(mean_mc, cov_mc, mean_gpq, cov_gpq)))


def polar2cartesian_sandblom_demo():
    # TODO: try replicating the experimental validation from Marginalized Transform paper
    # test for several different input positions and noise levels
    # average SKL score results for each configuration
    # show comparisons with SR
    pass


if __name__ == '__main__':
    gpq_polar2cartesian_demo()
