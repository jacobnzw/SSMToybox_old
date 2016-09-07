from utils import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from inference.gpquad import GPQKalman
from inference.cubature import CubatureKalman
from system.datagen import ReentryRadar
from models.tracking import ReentryRadar as ReentryRadarModel

# Generate reference trajectory by ODE integration
sys = ReentryRadar()
mc = 10
x = sys.simulate_trajectory(method='rk4', dt=0.05, duration=200, mc_sims=mc)
x = x.mean(axis=2)
# TODO: code simulation of measurements

# Initialize filters
ssm = ReentryRadarModel()
hdyn = {'alpha': 1.0, 'el': [15.0, 15.0, 15.0, 15.0, 15.0]}
hobs = {'alpha': 1.0, 'el': [15.0, 15.0, 1e4, 1e4, 1e4]}
gpqkf = GPQKalman(ssm, 'rbf', 'sr', hdyn, hobs)
ckf = CubatureKalman(ssm)




# Plots
plt.figure()
g = GridSpec(2, 4)
plt.subplot(g[:, :2])

# Earth surface w/ radar position
t = 0.02 * np.arange(-1, 4, 0.1)
plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t), 'darkblue', lw=2)
plt.plot(sys.sx, sys.sy, 'ko')

# vehicle trajectory
for i in range(mc):
    plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')
plt.subplot(g[:, 2:], polar=True)
plt.show()
