from utils import *
import numpy.linalg as la
from paper_code.journal_figure import *
from matplotlib.gridspec import GridSpec
from inference.gpquad import GPQKalman
from inference.unscented import UnscentedKalman
from inference.cubature import CubatureKalman
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


def reentry_simple_gpq_demo(dur=30, tau=0.1, mc=100):
    """

    Parameters
    ----------
    tau: float
        discretization period for the dynamics ODE integration method
    dur: int
        Duration of the dynamics simulation
    mc: int
        Number of Monte Carlo simulations.

    Notes
    -----
    The parameter mc determines the number of trajectories simulated.

    Returns
    -------

    """

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

    # PLOTS: Trajectories
    # plt.figure()
    # g = GridSpec(4, 2)
    # plt.subplot(g[:2, :])
    #
    # # Earth surface w/ radar position
    # t = np.arange(0.48 * np.pi, 0.52 * np.pi, 0.01)
    # plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t) - sys.R0, 'darkblue', lw=2)
    # plt.plot(sys.sx, sys.sy, 'ko')
    #
    # xzer = np.zeros(x.shape[1])
    # for i in range(mc):
    #     # Vehicle trajectory
    #     plt.plot(xzer, x[0, :, i], alpha=0.35, color='r', ls='--', lw=2)
    #
    #     # Filtered position estimate
    #     plt.plot(xzer, mean[0, :, i, 0], color='g', alpha=0.3)
    #     plt.plot(xzer, mean[0, :, i, 1], color='orange', alpha=0.3)

    # Altitude
    # x0 = sys.pars['x0_mean']
    # plt.subplot(g[2, :])
    # plt.ylim([0, x0[0]])
    # for i in range(mc):
    #     plt.plot(time_ind, x[0, :, i], alpha=0.35, color='b')
    # plt.ylabel('altitude [ft]')
    # plt.xlabel('time [s]')
    #
    # # Velocity
    # plt.subplot(g[3, :])
    # plt.ylim([0, x0[1]])
    # for i in range(mc):
    #     plt.plot(time_ind, x[1, :, i], alpha=0.35, color='b')
    # plt.ylabel('velocity [ft/s]')
    # plt.xlabel('time [s]')

    # Compute Performance Scores
    error2 = mean.copy()
    pos_lcr = np.zeros((steps, mc, num_alg))
    vel_lcr = pos_lcr.copy()
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
    vel_rmse = np.sqrt(error2[1:2, ...].sum(axis=0))
    vel_rmse_vs_time = vel_rmse.mean(axis=1)
    vel_inc_vs_time = vel_lcr.mean(axis=1)

    # PLOTS: Performance Scores
    plt.figure()
    g = GridSpec(4, 3)

    plt.subplot(g[0, :2])
    plt.ylabel('RMSE')
    plt.plot(time_ind, pos_rmse_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(time_ind, pos_rmse_vs_time[:, 1], label='UKF', color='r')
    plt.legend()

    plt.subplot(g[1, :2])
    plt.ylabel('Inclination')
    plt.plot(time_ind, pos_inc_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(time_ind, pos_inc_vs_time[:, 1], label='UKF', color='r')
    plt.legend()

    plt.subplot(g[2, :2])
    plt.ylabel('RMSE')
    plt.plot(time_ind, vel_rmse_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(time_ind, vel_rmse_vs_time[:, 1], label='UKF', color='r')
    plt.legend()

    plt.subplot(g[3, :2])
    plt.ylabel('Inclination')
    plt.plot(time_ind, vel_inc_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(time_ind, vel_inc_vs_time[:, 1], label='UKF', color='r')
    plt.legend()

    # Box plots of time-averaged scores
    plt.subplot(g[0, 2:])
    plt.boxplot(pos_rmse.mean(axis=0), labels=['GPQKF', 'UKF'])

    plt.subplot(g[1, 2:])
    plt.boxplot(pos_lcr.mean(axis=0), labels=['GPQKF', 'UKF'])

    plt.subplot(g[2, 2:])
    plt.boxplot(vel_rmse.mean(axis=0), labels=['GPQKF', 'UKF'])

    plt.subplot(g[3, 2:])
    plt.boxplot(vel_lcr.mean(axis=0), labels=['GPQKF', 'UKF'])

    plt.show()

    # TODO: pandas tables for printing into latex
    print('{:=^30}'.format(' Position '))
    print('Average RMSE: {}'.format(np.sqrt(error2.sum(axis=0)).mean(axis=(0, 1))))
    print('Average I2: {}'.format(pos_inc_vs_time.mean(axis=0)))


def reentry_simple_data(dur=30, tau=0.1, mc=100):
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

    # RMSE
    ax1 = fig.add_subplot(211, ylabel='RMSE')
    ax1.plot(time, pos_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    ax1.plot(time, pos_rmse_vs_time[:, 1], lw=2, label='UKF')
    ax1.legend()
    ax1.tick_params(axis='both', which='both', top='off', right='off', labelright='off', labelbottom='off')

    # inclination indicator
    ax2 = fig.add_subplot(212, xlabel='time [s]', ylabel=r'$ \nu $', sharex=ax1)
    ax2.plot(time, pos_inc_vs_time[:, 0], lw=2, label='GPQKF')
    ax2.plot(time, pos_inc_vs_time[:, 1], lw=2, label='UKF')
    ax2.tick_params(axis='both', which='both', top='off', right='off', labelright='off')
    fig.tight_layout(pad=0.5)

    print("Saving figure ...")
    savefig("reentry_position_rmse_inc")

if __name__ == '__main__':
    import pickle
    # get simulation results
    # time, x, mean, cov = reentry_simple_data(mc=100)
    #
    # # dump simulated data for fast re-plotting
    # print('Pickling data ...')
    # with open('reentry_data_mc100_tau0.1.dat', 'wb') as f:
    #     pickle.dump((time, x, mean, cov), f)
    #     f.close()

    # load pickled data
    print('Unpickling data ...')
    with open('reentry_data_mc100_tau0.1.dat', 'rb') as f:
        time, x, mean, cov = pickle.load(f)
        f.close()

    # calculate scores and generate publication ready figures
    reentry_simple_plots(time, x, mean, cov)
