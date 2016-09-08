from utils import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from inference.gpquad import GPQKalman
from inference.unscented import UnscentedKalman
from system.datagen import ReentryRadar
from models.tracking import ReentryRadar as ReentryRadarModel


def reentry_gpq_demo():
    # Generate reference trajectory by ODE integration
    sys = ReentryRadar()
    mc_sims = 25
    # x = sys.simulate_trajectory(method='rk4', dt=0.5, duration=200, mc_sims=mc_sims)
    # x_ref = x.mean(axis=2)
    # y = sys.simulate_measurements(x_ref, mc_per_step=mc_sims)

    # Initialize filters
    ssm = ReentryRadarModel()
    x, y = ssm.simulate(steps=750, mc_sims=10)
    x_ref = x.mean(axis=2)
    hdyn = {'alpha': 1.0, 'el': 25*np.ones(5,)}
    hobs = {'alpha': 1.0, 'el': [25.0, 25.0, 1e4, 1e4, 1e4]}
    alg = (GPQKalman(ssm, 'rbf', 'ut', hdyn, hobs),
           UnscentedKalman(ssm),)
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

    for i in range(mc_sims):
        # Vehicle trajectory
        plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')

        # Filtered position estimate
        plt.plot(mean[0, 1:, i, 0], mean[1, 1:, i, 0], color='g', alpha=0.3)
        plt.plot(mean[0, 1:, i, 1], mean[1, 1:, i, 1], color='orange', alpha=0.3)

    # Performance score plots
    error2 = mean.copy()
    for k in range(steps):
        for imc in range(mc_sims):
            for a in range(num_alg):
                error2[:, k, imc, a] = squared_error(x_ref[:, k], mean[:, k, imc, a])
    pos_rmse_vs_time = np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1)
    plt.subplot(g[:, 2:])
    plt.plot(pos_rmse_vs_time[:, 0], label='gpqkf', color='g')
    plt.plot(pos_rmse_vs_time[:, 1], label='ukf', color='orange')
    plt.legend()
    plt.show()

    print(pos_rmse_vs_time.mean(axis=0))

if __name__ == '__main__':
    reentry_gpq_demo()
