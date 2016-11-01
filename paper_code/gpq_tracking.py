from utils import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from inference.gpquad import GPQKalman
from inference.unscented import UnscentedKalman
from system.datagen import ReentryRadar, ReentryRadarSimple
from models.tracking import ReentryRadar as ReentryRadarModel
from models.tracking import ReentryRadarSimple as ReentryRadarSimpleModel


def reentry_gpq_demo():
    mc_sims = 20
    disc_tau = 0.5  # discretization period

    # Generate reference trajectory by ODE integration
    sys = ReentryRadar()
    x = sys.simulate_trajectory(method='rk4', dt=disc_tau, duration=200, mc_sims=mc_sims)
    x_ref = x.mean(axis=2)
    y = np.zeros((sys.zD,) + x.shape[1:])
    for i in range(mc_sims):
        y[..., i] = sys.simulate_measurements(x[..., i], mc_per_step=1).squeeze()

    # Initialize model
    ssm = ReentryRadarModel(dt=disc_tau)
    # x, y = ssm.simulate(steps=750, mc_sims=10)
    # x_ref = x.mean(axis=2)

    # Initialize filters
    hdyn = {'alpha': 1.0, 'el': 5 * [25.0]}
    hobs = {'alpha': 1.0, 'el': [25.0, 25.0, 1e4, 1e4, 1e4]}
    alg = (
        GPQKalman(ssm, 'rbf', 'ut', hdyn, hobs),
        UnscentedKalman(ssm),
    )

    # Are both filters using the same sigma-points?
    # assert np.array_equal(alg[0].tf_dyn.model.points, alg[1].tf_dyn.unit_sp)

    num_alg = len(alg)
    d, steps, mc_sims = x.shape
    mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
    for imc in range(mc_sims):
        for ia, a in enumerate(alg):
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
            a.reset()

    # Plots
    plt.figure()
    g = GridSpec(2, 4)
    plt.subplot(g[:, :2])

    # Earth surface w/ radar position
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t), color='darkblue', lw=2)
    plt.plot(sys.sx, sys.sy, 'ko')

    plt.plot(x_ref[0, :], x_ref[1, :], color='r', ls='--')
    # Convert from polar to cartesian
    meas = np.stack((sys.sx + y[0, ...] * np.cos(y[1, ...]), sys.sy + y[0, ...] * np.sin(y[1, ...])), axis=0)
    for i in range(mc_sims):
        # Vehicle trajectory
        # plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')

        # Plot measurements
        plt.plot(meas[0, :, i], meas[1, :, i], 'k.', alpha=0.3)

        # Filtered position estimate
        plt.plot(mean[0, 1:, i, 0], mean[1, 1:, i, 0], color='g', alpha=0.3)
        plt.plot(mean[0, 1:, i, 1], mean[1, 1:, i, 1], color='orange', alpha=0.3)

    # Performance score plots
    error2 = mean.copy()
    lcr = np.zeros((steps, mc_sims, num_alg))
    for a in range(num_alg):
        for k in range(steps):
            mse = mse_matrix(x[:4, k, :], mean[:4, k, :, a])
            for imc in range(mc_sims):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                lcr[k, imc, a] = log_cred_ratio(x[:4, k, imc], mean[:4, k, imc, a], cov[:4, :4, k, imc, a], mse)

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1)
    inc_ind_vs_time = lcr.mean(axis=1)

    # Plots
    plt.subplot(g[0, 2:])
    plt.title('RMSE')
    plt.plot(pos_rmse_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(pos_rmse_vs_time[:, 1], label='UKF', color='r')
    plt.legend()
    plt.subplot(g[1, 2:])
    plt.title('Inclination Indicator $I^2$')
    plt.plot(inc_ind_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(inc_ind_vs_time[:, 1], label='UKF', color='r')
    plt.legend()
    plt.show()

    print('Average RMSE: {}'.format(pos_rmse_vs_time.mean(axis=0)))
    print('Average I2: {}'.format(inc_ind_vs_time.mean(axis=0)))


def reentry_simple_gpq_demo():
    mc_sims = 100
    disc_tau = 0.05  # discretization period
    dur = 30  # duration

    # Generate reference trajectory by ODE integration
    sys = ReentryRadarSimple()
    x = sys.simulate_trajectory(method='rk4', dt=disc_tau, duration=dur, mc_sims=mc_sims)

    # pick only non-divergent trajectories
    x = x[..., np.all(np.abs(x) != np.inf, axis=(0, 1))]
    mc_sims = x.shape[2]

    x_ref = x.mean(axis=2)
    y = np.zeros((sys.zD,) + x.shape[1:])
    for i in range(mc_sims):
        y[..., i] = sys.simulate_measurements(x[..., i], mc_per_step=1).squeeze()

    # Initialize model
    ssm = ReentryRadarSimpleModel(dt=disc_tau)

    # Initialize filters
    hdyn = {'alpha': 1.0, 'el': 3 * [10.0]}
    hobs = {'alpha': 1.0, 'el': 3 * [10.0]}
    alg = (
        # GPQKalman(ssm, 'rbf', 'ut', hdyn, hobs),
        UnscentedKalman(ssm),
    )

    # Are both filters using the same sigma-points?
    # assert np.array_equal(alg[0].tf_dyn.model.points, alg[1].tf_dyn.unit_sp)

    num_alg = len(alg)
    d, steps, mc_sims = x.shape
    mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
    for imc in range(mc_sims):
        for ia, a in enumerate(alg):
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
            a.reset()

    # Plots
    plt.figure()
    g = GridSpec(2, 4)
    plt.subplot(g[:, :2])

    # Earth surface w/ radar position
    t = np.arange(0.48 * np.pi, 0.52 * np.pi, 0.01)
    plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t) - sys.R0, 'darkblue', lw=2)
    plt.plot(sys.sx, sys.sy, 'ko')

    plt.plot(x_ref[0, :], x_ref[1, :], color='r', ls='--')
    # Convert from polar to cartesian
    # meas = np.stack((sys.sx + y[0, ...] * np.cos(y[1, ...]), sys.sy + y[0, ...] * np.sin(y[1, ...])), axis=0)
    xzer = np.zeros(x.shape[1])
    for i in range(mc_sims):
        # Vehicle trajectory
        plt.plot(xzer, x[0, :, i], alpha=0.35, color='r', ls='--', lw=2)

        # Plot measurements
        # plt.plot(meas[0, :, i], meas[1, :, i], 'k.', alpha=0.3)

        # Filtered position estimate
        plt.plot(xzer, mean[0, :, i, 0], color='g', alpha=0.3)
        # plt.plot(xzer, mean[0, :, i, 1], color='orange', alpha=0.3)

    # Performance score plots
    error2 = mean.copy()
    lcr = np.zeros((steps, mc_sims, num_alg))
    for a in range(num_alg):
        for k in range(steps):
            mse = mse_matrix(x[:1, k, :], mean[:1, k, :, a])
            for imc in range(mc_sims):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                lcr[k, imc, a] = log_cred_ratio(x[:1, k, imc], mean[:1, k, imc, a], cov[:1, :1, k, imc, a], mse)

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[:1, ...]).sum(axis=0)).mean(axis=1)
    inc_ind_vs_time = lcr.mean(axis=1)

    # Plots
    plt.subplot(g[0, 2:])
    plt.title('RMSE')
    plt.plot(pos_rmse_vs_time[:, 0], label='GPQKF', color='g')
    # plt.plot(pos_rmse_vs_time[:, 1], label='UKF', color='r')
    plt.legend()
    plt.subplot(g[1, 2:])
    plt.title('Inclination Indicator $I^2$')
    plt.plot(inc_ind_vs_time[:, 0], label='GPQKF', color='g')
    # plt.plot(inc_ind_vs_time[:, 1], label='UKF', color='r')
    plt.legend()
    plt.show()

    print('Average RMSE: {}'.format(pos_rmse_vs_time.mean(axis=0)))
    print('Average I2: {}'.format(inc_ind_vs_time.mean(axis=0)))

if __name__ == '__main__':
    # reentry_gpq_demo()
    reentry_simple_gpq_demo()
