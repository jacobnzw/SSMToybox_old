from transforms.bayesquad import GPQ
from transforms.quad import MonteCarlo, SphericalRadial
from utils import *
import numpy.linalg as la
import matplotlib.pyplot as plt


"""
Gaussian Process Quadrature moment transformation tested on a mapping from polar to cartesian coordinates.
"""

# TODO: create decorator with 3 arguments to make functions compatible w GPQ transform


def polar2cartesian(x):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def cartesian2polar(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    theta = np.arctan2(x[1], x[0])
    return np.array([r, theta])


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
    fx = np.apply_along_axis(polar2cartesian, 0, x)

    # MC transformed moments
    mean_mc, cov_mc, cc_mc = tf_mc.apply(polar2cartesian, mean_in, cov_in)
    ellipse_mc = ellipse_points(mean_mc, cov_mc)

    # GPQ transformed moments with ellipse points
    mean_gpq, cov_gpq, cc = tf_gpq.apply(polar2cartesian, mean_in, cov_in)
    ellipse_gpq = ellipse_points(mean_gpq, cov_gpq)

    # SR transformed moments with ellipse points
    mean_sr, cov_sr, cc = tf_sr.apply(polar2cartesian, mean_in, cov_in)
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


def polar2cartesian_spiral_demo():
    # test for several different input positions and noise levels
    # average SKL score results for each configuration
    # show comparisons with SR

    # Archimedean spiral polar form
    r_spiral = lambda x: 2*x

    theta_min, theta_max = 0.25*np.pi, 2.25*np.pi
    num_locs = 10
    # spiral
    theta = np.linspace(theta_min, theta_max, 100)
    r = r_spiral(theta)

    # equidistant points on a spiral
    theta_pt = np.linspace(theta_min, theta_max, 10)
    r_pt = r_spiral(theta_pt)

    # samples from 12 normal RVs centered on the points of the spiral
    mean = np.array([r_pt, theta_pt])
    cov = np.diag([0.2 ** 2, (np.pi / 10) ** 2])

    num_dim, num_samples = 2, 50
    x12 = np.zeros((num_dim, num_samples, num_locs))
    for loc in range(num_locs):
        x12[..., loc] = np.random.multivariate_normal(mean[..., loc], cov, size=num_samples).T

    # PLOTS: Polar coordinates
    fig = plt.figure()

    ax = fig.add_subplot(121, projection='polar')
    ax.plot(0, 0, 'r+', ms=12)
    ax.plot(theta, r)
    ax.plot(theta_pt, r_pt, 'o')
    for loc in range(num_locs):
        # ax.plot(x12[0, :, loc], x12[1, :, loc], '.')
        pol_ellipse = ellipse_points(mean[..., loc], cov)
        # pol_ellipse = np.apply_along_axis(cartesian2polar, 0, ellipse_points(mean[..., loc], cov))
        ax.plot(pol_ellipse[1, :], pol_ellipse[0, :])

    # PLOTS: Cartesian coordinates
    pol_spiral = np.array([r, theta])
    pol12_spiral = np.array([r_pt, theta_pt])
    car_spiral = np.apply_along_axis(polar2cartesian, 0, pol_spiral)
    car12_spiral = np.apply_along_axis(polar2cartesian, 0, pol12_spiral)
    car_x12 = np.apply_along_axis(polar2cartesian, 0, x12)

    ax = fig.add_subplot(122)
    ax.plot(0, 0, 'r+', ms=12)
    ax.plot(car_spiral[0, :], car_spiral[1, :])
    ax.plot(car12_spiral[0, :], car12_spiral[1, :], 'o')
    for loc in range(num_locs):
        # ax.plot(car_x12[0, :, loc], car_x12[1, :, loc], '.')
        car_ellipse = np.apply_along_axis(polar2cartesian, 0, ellipse_points(mean[..., loc], cov))
        ax.plot(car_ellipse[0, :], car_ellipse[1, :])

    plt.show()


if __name__ == '__main__':
    polar2cartesian_spiral_demo()
