import numpy as np
from utils import *
from inference.gpquad import GPQKalman
from inference.unscented import UnscentedKalman
from inference.cubature import CubatureKalman

import matplotlib as mpl
mpl.use('pgf')


def figsize(scale=1.0):
    """
    Calculates figure width and height given the scale.

    Parameters
    ----------
    scale: float
        Figure scale.

    Returns
    -------

    """

    FIG_WIDTH_PT = 347.12354  # Get this from LaTeX using \the\columnwidth
    INCH_PER_PT = 1.0 / 72.27  # Convert pt to inch
    PHI = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)

    fig_width = FIG_WIDTH_PT * INCH_PER_PT * scale    # width in inches
    fig_height = fig_width * PHI * 0.85         # height in inches
    return [fig_width, fig_height]

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "font.size": 12,
    "axes.labelsize": 12,               # LaTeX default is 10pt font.
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "axes.prop_cycle": ['#5DA5DA', '#FAA43A', '#60BD68',
    #                      '#F17CB0', '#B2912F', '#B276B2',
    #                      '#DECF3F', '#F15854', '#4D4D4D'],
    "figure.figsize": figsize(1.0),      # default fig size of 0.9 textwidth
    "pgf.preamble": [                    # plots will be generated using this preamble
        r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",
        ]
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt


def newfig(width):
    # for making new figure
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename):
    """
    Save figure to PGF. PDF copy created for viewing convenience.

    Parameters
    ----------
    filename

    Returns
    -------

    """
    plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))


def reentry_simple_data(dur=30, tau=0.1, mc=100):
    from system.datagen import ReentryRadar, ReentryRadarSimple
    from models.tracking import ReentryRadarSimple as ReentryRadarSimpleModel

    # Generate reference trajectory by ODE integration
    sys = ReentryRadarSimple()
    x = sys.simulate_trajectory(method='rk4', dt=tau, duration=dur, mc_sims=mc)

    # pick only non-divergent trajectories
    x = x[..., np.all(x >= 0, axis=(0, 1))]
    mc = x.shape[2]

    y = np.zeros((sys.zD,) + x.shape[1:])
    for i in range(mc):
        y[..., i] = sys.simulate_measurements(x[..., i], mc_per_step=1).squeeze()

    # GPQKF kernel parameters
    # hdyn = {'alpha': 1.0, 'el': 3 * [20]}
    # hobs = {'alpha': 1.0, 'el': [20, 1e2, 1e2]}
    hdyn = {'alpha': 1.0, 'el': [7, 7, 7]}
    hobs = {'alpha': 1.0, 'el': [7, 20, 20]}

    # Initialize model
    ssm = ReentryRadarSimpleModel(dt=tau)

    # Initialize filters
    alg = (
        GPQKalman(ssm, 'rbf', 'ut', hdyn, hobs),
        # CubatureKalman(ssm),
        UnscentedKalman(ssm),
    )

    num_alg = len(alg)
    d, steps, mc = x.shape
    mean, cov = np.zeros((d, steps, mc, num_alg)), np.zeros((d, d, steps, mc, num_alg))
    for imc in range(mc):
        for ia, a in enumerate(alg):
            # Do filtering and reset the filters for each new track
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
            a.reset()

    # time index for plotting
    time_ind = np.linspace(1, dur, x.shape[1])

    return time_ind, x, mean, cov


def reentry_simple_plots(time, x, mean, cov):
    d, steps, mc, num_alg = mean.shape
    error2 = mean.copy()
    pos_lcr = np.zeros((steps, mc, num_alg))
    vel_lcr = pos_lcr.copy()

    print("Calculating scores ...")
    for a in range(num_alg):
        for k in range(steps):
            pos_mse = mse_matrix(x[:1, k, :], mean[:1, k, :, a])
            vel_mse = mse_matrix(x[1:2, k, :], mean[1:2, k, :, a])
            for imc in range(mc):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                pos_lcr[k, imc, a] = log_cred_ratio(x[:1, k, imc], mean[:1, k, imc, a],
                                                    cov[:1, :1, k, imc, a], pos_mse)
                vel_lcr[k, imc, a] = log_cred_ratio(x[1:2, k, imc], mean[1:2, k, imc, a],
                                                    cov[1:2, 1:2, k, imc, a], vel_mse)

    # Averaged position/velocity RMSE and inclination in time
    pos_rmse = np.sqrt(error2[:1, ...].sum(axis=0))
    pos_rmse_vs_time = pos_rmse.mean(axis=1)
    pos_inc_vs_time = pos_lcr.mean(axis=1)

    fig = plt.figure(figsize=figsize())
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='RMSE')
    ax.plot(time, pos_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    ax.plot(time, pos_rmse_vs_time[:, 1], lw=2, label='UKF')
    ax.legend()
    fig.tight_layout(pad=0.5)

    print("Saving figure ...")
    savefig("reentry_rmse")

if __name__ == '__main__':
    time, x, mean, cov = reentry_simple_data(mc=100)
    reentry_simple_plots(time, x, mean, cov)
