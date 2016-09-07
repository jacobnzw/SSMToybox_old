from utils import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from inference.gpquad import GPQKalman
from inference.cubature import CubatureKalman
from system.datagen import ReentryRadar
from models.tracking import ReentryRadar as ReentryRadarModel


def reentry_gpq_demo():
    # Generate reference trajectory by ODE integration
    sys = ReentryRadar()
    mc_sims = 10
    x = sys.simulate_trajectory(method='rk4', dt=0.05, duration=200, mc_sims=mc_sims)
    x_ref = x.mean(axis=2)
    y = sys.simulate_measurements(x_ref, mc_per_step=mc_sims)

    # Plots
    plt.figure()
    g = GridSpec(2, 4)
    plt.subplot(g[:, :2])

    # Earth surface w/ radar position
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t), 'darkblue', lw=2)
    plt.plot(sys.sx, sys.sy, 'ko')

    # vehicle trajectory
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')
    plt.subplot(g[:, 2:], polar=True)
    plt.show()

    # Initialize filters
    ssm = ReentryRadarModel()
    hdyn = {'alpha': 1.0, 'el': [15.0, 15.0, 15.0, 15.0, 15.0]}
    hobs = {'alpha': 1.0, 'el': [15.0, 15.0, 1e4, 1e4, 1e4]}
    alg = (GPQKalman(ssm, 'rbf', 'sr', hdyn, hobs),
           CubatureKalman(ssm),)

    d, steps = x.shape
    mean, cov = np.zeros((d, steps, mc_sims)), np.zeros((d, d, steps, mc_sims))
    for imc in range(mc_sims):
        for ia, a in enumerate(alg):
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])

if __name__ == '__main__':
    reentry_gpq_demo()
