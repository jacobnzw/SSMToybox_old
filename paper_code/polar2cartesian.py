from transforms.bayesquad import GPQ
from transforms.quad import MonteCarlo, SphericalRadial, GaussHermite
from utils import *
import numpy.linalg as la
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from paper_code.journal_figure import *
from collections import OrderedDict


"""
Gaussian Process Quadrature moment transformation tested on a mapping from polar to cartesian coordinates.
"""


def no_par(f):
    def wrapper(x):
        return f(x, None)
    return wrapper


def polar2cartesian(x, pars):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def cartesian2polar(x, pars):
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


def polar2cartesian_skl_demo():
    num_dim = 2

    # create spiral in polar domain
    r_spiral = lambda x: 10 * x
    theta_min, theta_max = 0.25 * np.pi, 2.25 * np.pi

    # equidistant points on a spiral
    num_mean = 10
    theta_pt = np.linspace(theta_min, theta_max, num_mean)
    r_pt = r_spiral(theta_pt)

    # samples from normal RVs centered on the points of the spiral
    mean = np.array([r_pt, theta_pt])
    r_std = 0.5

    # multiple azimuth covariances in increasing order
    num_cov = 10
    theta_std = np.deg2rad(np.linspace(6, 36, num_cov))
    cov = np.zeros((num_dim, num_dim, num_cov))
    for i in range(num_cov):
        cov[..., i] = np.diag([r_std**2, theta_std[i]**2])

    # COMPARE moment transforms
    moment_tforms = OrderedDict([
        ('gpq-sr', GPQ(num_dim, 'rbf', 'sr', {'alpha': 1.0, 'el': [60, 6]})),
        ('sr', SphericalRadial(num_dim)),
    ])
    baseline_mtf = MonteCarlo(num_dim, n=10000)
    num_tforms = len(moment_tforms)

    # initialize storage of SKL scores
    skl_dict = dict([(mt_str, np.zeros((num_mean, num_cov))) for mt_str in moment_tforms.keys()])

    # for each mean
    for i in range(num_mean):

        # for each covariance
        for j in range(num_cov):
            mean_in, cov_in = mean[..., i], cov[..., j]

            # calculate baseline using Monte Carlo
            mean_out_mc, cov_out_mc, cc = baseline_mtf.apply(polar2cartesian, mean_in, cov_in, None)

            # for each MT
            for mt_str in moment_tforms.keys():

                # calculate the transformed moments
                mean_out, cov_out, cc = moment_tforms[mt_str].apply(polar2cartesian, mean_in, cov_in, None)

                # compute SKL
                skl_dict[mt_str][i, j] = skl(mean_out_mc, cov_out_mc, mean_out, cov_out)

    # PLOT the SKL score for each MT and position on the spiral
    plt.style.use('seaborn-deep')
    fig = plt.figure(figsize=figsize(1.0))

    # Average over mean indexes
    ax1 = fig.add_subplot(121)
    index = np.arange(num_mean)+1
    for mt_str in moment_tforms.keys():
        ax1.plot(index, skl_dict[mt_str].mean(axis=1), marker='o', label=mt_str.upper())
    ax1.set_xlabel('Position index')
    ax1.set_ylabel('SKL')

    # Average over azimuth variances
    ax2 = fig.add_subplot(122, sharey=ax1)
    for mt_str in moment_tforms.keys():
        ax2.plot(np.rad2deg(theta_std), skl_dict[mt_str].mean(axis=0), marker='o', label=mt_str.upper())
    ax2.set_xlabel('Azimuth STD [$ \circ $]')
    ax2.legend()
    fig.tight_layout(pad=0.5)

    # save figure
    savefig('polar2cartesian_skl')


def polar2cartesian_spiral_demo():
    num_dim = 2

    # create spiral in polar domain
    r_spiral = lambda x: 10 * x
    theta_min, theta_max = 0.25 * np.pi, 2.25 * np.pi
    theta = np.linspace(theta_min, theta_max, 100)
    r = r_spiral(theta)

    # equidistant points on a spiral
    num_mean = 10
    theta_pt = np.linspace(theta_min, theta_max, num_mean)
    r_pt = r_spiral(theta_pt)

    # samples from normal RVs centered on the points of the spiral
    mean = np.array([r_pt, theta_pt])
    r_std = 0.5

    # multiple azimuth covariances in increasing order
    num_cov = 10
    theta_std = np.deg2rad(np.linspace(6, 36, num_cov))
    cov = np.zeros((num_dim, num_dim, num_cov))
    for i in range(num_cov):
        cov[..., i] = np.diag([r_std ** 2, theta_std[i] ** 2])

    pol_spiral = np.array([r, theta])
    pol_spiral_pt = np.array([r_pt, theta_pt])
    car_spiral = np.apply_along_axis(polar2cartesian, 0, pol_spiral, None)
    car_spiral_pt = np.apply_along_axis(polar2cartesian, 0, pol_spiral_pt, None)

    # PLOTS: Input moments in Cartesian coordinates
    fig = plt.figure(figsize=figsize(1.0))
    ax = fig.add_subplot(111)

    # origin
    ax.plot(0, 0, 'r+', ms=12)

    # spiral
    ax.plot(car_spiral[0, :], car_spiral[1, :])

    # points on a spiral, i.e. input means
    ax.plot(car_spiral_pt[0, :], car_spiral_pt[1, :], 'o')

    # for every input mean and covariance
    for i in range(num_mean):
        for j in range(num_cov):

            # plot covariance ellipse
            car_ellipse = np.apply_along_axis(polar2cartesian, 0, ellipse_points(mean[..., i], cov[..., j]), None)
            ax.plot(car_ellipse[0, :], car_ellipse[1, :])

    fig.tight_layout(pad=0.5)

    savefig('polar2cartesian_spiral')


if __name__ == '__main__':
    # polar2cartesian_skl_demo()
    polar2cartesian_spiral_demo()
