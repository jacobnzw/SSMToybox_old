import numpy as np
from numpy import newaxis as na
from transforms.bqmodel import GaussianProcess
import scipy as sp


# true function
# f = lambda x: x**2
f = lambda x: 2*x + 0.5*x**2 + np.sin(x**2) - np.cos(2*x**2)
f = lambda x: 25*x / (1 + x**2)

model = GaussianProcess(1, np.array([[1.0, 1.0]]))

# sigma-point template
x = model.points
y = np.apply_along_axis(f, 1, x)
num_dim, num_pts = x.shape

# samples used for non-parametric kernel matrix estimation
num_samples = 100
sample_std = 1

# plot model
xtest = np.linspace(-3, 3, 100)[na, :]
ytest = np.apply_along_axis(f, 1, xtest)
model.plot_model(xtest, y, np.array([[1.0, 0.1]]), ytest)


# generate pertubed templates
def perturb_template(x, samples=10, std=1.0):
    num_dim, num_pts = x.shape
    delta = np.random.multivariate_normal(np.zeros((num_dim, )), std*np.eye(num_dim), size=samples).T
    return x[..., na] + delta[:, na, :]


# non-parametric kernel matrix estimate
def estimate_kernel_mat(y):
    # y (num_pts, num_samples)
    num_pts, num_samples = y.shape
    K = np.zeros((num_pts, num_pts))
    for i in range(num_samples):
        K += np.outer(y[:, i], y[:, i])
    return num_samples ** -1 * K


# kernel fitting objective
def kernel_obj(log_theta, x, K_true):
    K_theta = model.kernel.eval(np.exp(log_theta), x)
    return np.linalg.norm(K_true - K_theta)


X = perturb_template(x, samples=num_samples, std=sample_std)
Y = np.zeros((num_pts, num_samples))
for i in range(num_samples):
    Y[:, i] = np.apply_along_axis(f, 1, X[..., i])

K_true = estimate_kernel_mat(Y)

# minimize matrix norm to find optimal kernel parameters
from scipy.optimize import minimize
log_theta_0 = np.log(np.array([[1, 1]], dtype=float))
opt = minimize(kernel_obj, log_theta_0, args=(x, K_true), jac=False, method='BFGS')

print('Optimal kernel parameters (samples={0:d}): {1:}'.format(num_samples, np.exp(opt.x)))

# plot the GP fit
model.plot_model(xtest, y, np.exp(opt.x), ytest)