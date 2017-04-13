import numpy as np
import scipy as sp
from numpy import newaxis as na
from transforms.bqmodel import GaussianProcess
from scipy.optimize import minimize

# true function
# f = lambda x: x**2
# f = lambda x: 2*x + 0.5*x**2 + np.sin(x**2) - np.cos(2*x**2)
# f = lambda x: 25*x / (1 + x**2)
f = lambda x: sp.special.expit(np.sin((x+4)**2)+np.cos(x+4))
f = lambda x: np.cos(x)+np.sin(0.5*x)

model = GaussianProcess(1, np.array([[1.0, 1.0]]), points='gh', point_par={'degree': 3})

# sigma-point template
x = model.points
y = np.apply_along_axis(f, 1, x)
num_dim, num_pts = x.shape

# samples used for non-parametric kernel matrix estimation
num_samples = 100
sample_std = 0.0001

# plot model
xtest = np.linspace(-5, 5, 100)[na, :]
ytest = np.apply_along_axis(f, 1, xtest)
theta_0 = np.array([[1.0, 0.1]])
model.plot_model(xtest, y, theta_0, ytest)


# generate pertubed templates
def perturb_template(x, samples=10, std=1.0):
    num_dim, num_pts = x.shape

    # random translation
    delta = np.random.multivariate_normal(np.zeros((num_dim, )), std*np.eye(num_dim), size=samples).T
    res = x[..., na] + delta[:, na, :]

    if num_dim > 1:
        # random rotation in n-D
        A = np.random.rand(num_dim, num_dim, num_samples)
        for s in range(num_samples):
            Q, R = np.linalg.qr(A[..., s])
            res[..., s] = Q.dot(res[..., s])

    return res


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
    return np.linalg.norm(K_true - K_theta, ord=np.inf)
    # return np.linalg.norm(K_true - K_theta, ord='fro')


X = perturb_template(x, samples=num_samples, std=sample_std)
Y = np.zeros((num_pts, num_samples))
for i in range(num_samples):
    Y[:, i] = np.apply_along_axis(f, 1, X[..., i])

K_true = estimate_kernel_mat(Y)

# minimize matrix norm to find optimal kernel parameters
log_theta_0 = np.log(np.array([[1.0, 1.0]]))
opt = minimize(kernel_obj, log_theta_0, args=(x, K_true), jac=False, method='BFGS')
theta_star = np.exp(opt.x)

# optimization with multiple restarts
# opt_list = []
# for i in range(10):
#     log_theta_0 = np.random.multivariate_normal(np.zeros((2,)), 0.1*np.eye(2)).T
#     opt = minimize(kernel_obj, log_theta_0, args=(x, K_true), jac=False, method='BFGS')
#     opt_list.append(opt)
# theta_star = np.exp(opt_list[np.argmin([ol.fun for ol in opt_list])].x)

print('Optimal kernel parameters (samples={0:d}): {1:}'.format(num_samples, theta_star))

# plot the GP fit
model.plot_model(xtest, y, theta_star, ytest)