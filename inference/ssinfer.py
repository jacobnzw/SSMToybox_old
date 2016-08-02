import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.stats import multivariate_normal
from numpy import newaxis as na
from models.ssmodel import StateSpaceModel
from transforms.mtform import MomentTransform


class StateSpaceInference(object):
    def __init__(self, ssm, transf_dyn, transf_meas):
        # separate moment transforms for system dynamics and measurement model
        assert isinstance(transf_dyn, MomentTransform) and isinstance(transf_meas, MomentTransform)
        self.transf_dyn = transf_dyn
        self.transf_meas = transf_meas
        # dynamical system whose state is to be estimated
        assert isinstance(ssm, StateSpaceModel)
        self.ssm = ssm
        # set initial condition mean and covariance, and noise covariances
        self.x_mean_fi, self.x_cov_fi, self.q_mean, self.q_cov, self.r_mean, self.r_cov, self.G = ssm.get_pars(
            'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov', 'q_factor'
        )
        self.flags = {'filtered': False, 'smoothed': False}
        self.x_mean_pr, self.x_cov_pr, = None, None
        self.x_mean_sm, self.x_cov_sm = None, None
        self.xx_cov, self.xz_cov = None, None
        self.pr_mean, self.pr_cov, self.pr_xx_cov = None, None, None
        self.fi_mean, self.fi_cov = None, None
        self.sm_mean, self.sm_cov = None, None
        self.D, self.N = None, None

    def get_flag(self, key):
        return self.flags[key]

    def set_flag(self, key, value):
        self.flags[key] = value

    def forward_pass(self, data):
        self.D, self.N = data.shape
        self.fi_mean = np.zeros((self.ssm.xD, self.N))
        self.fi_cov = np.zeros((self.ssm.xD, self.ssm.xD, self.N))
        self.fi_mean[:, 0], self.fi_cov[..., 0] = self.x_mean_fi, self.x_cov_fi
        self.pr_mean = self.fi_mean.copy()
        self.pr_cov = self.fi_cov.copy()
        self.pr_xx_cov = self.fi_cov.copy()
        for k in range(1, self.N):  # iterate over columns of data
            self._time_update(k - 1)
            self.pr_mean[..., k] = self.x_mean_pr
            self.pr_cov[..., k] = self.x_cov_pr
            self.pr_xx_cov[..., k] = self.xx_cov
            self._measurement_update(data[:, k])
            self.fi_mean[..., k], self.fi_cov[..., k] = self.x_mean_fi, self.x_cov_fi
        # set flag that filtered state sequence is available
        self.set_flag('filtered', True)
        # smoothing estimate at the last time step == the filtering estimate at the last time step
        self.x_mean_sm, self.x_cov_sm = self.x_mean_fi, self.x_cov_fi
        return self.fi_mean, self.fi_cov

    def backward_pass(self):
        assert self.get_flag('filtered')  # require filtered state
        self.sm_mean = self.fi_mean.copy()
        self.sm_cov = self.fi_cov.copy()
        for k in range(self.N-2, 0, -1):
            self.x_mean_pr = self.pr_mean[..., k + 1]
            self.x_cov_pr = self.pr_cov[..., k + 1]
            self.xx_cov = self.pr_xx_cov[..., k+1]
            self.x_mean_fi = self.fi_mean[..., k]
            self.x_cov_fi = self.fi_cov[..., k]
            self._smoothing_update()
            self.sm_mean[..., k] = self.x_mean_sm
            self.sm_cov[..., k] = self.x_cov_sm
        self.set_flag('smoothed', True)
        return self.sm_mean, self.sm_cov

    def reset(self):
        self.x_mean_fi, self.x_cov_fi = self.ssm.get_pars('x0_mean', 'x0_cov')
        self.flags = {'filtered': False, 'smoothed': False}
        self.x_mean_pr, self.x_cov_pr, = None, None
        self.x_mean_sm, self.x_cov_sm = None, None
        self.xx_cov, self.xz_cov = None, None
        self.pr_mean, self.pr_cov, self.pr_xx_cov = None, None, None
        self.fi_mean, self.fi_cov = None, None
        self.sm_mean, self.sm_cov = None, None
        self.D, self.N = None, None

    def _time_update(self, time, *args):
        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.ssm.q_additive else np.hstack((self.x_mean_fi, self.q_mean))
        cov = self.x_cov_fi if self.ssm.q_additive else block_diag(self.x_cov_fi, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        self.x_mean_pr, self.x_cov_pr, self.xx_cov = self.transf_dyn.apply(self.ssm.dyn_eval, mean, cov,
                                                                           self.ssm.par_fcn(time), *args)
        if self.ssm.q_additive:
            self.x_cov_pr += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pr if self.ssm.r_additive else np.hstack((self.x_mean_pr, self.r_mean))
        cov = self.x_cov_pr if self.ssm.r_additive else block_diag(self.x_cov_pr, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        self.z_mean_pred, self.z_cov_pred, self.xz_cov = self.transf_meas.apply(self.ssm.meas_eval, mean, cov,
                                                                                self.ssm.par_fcn(time), *args)
        # in additive case, noise covariances need to be added
        if self.ssm.r_additive:
            self.z_cov_pred += self.r_cov

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xz_cov = self.xz_cov[:, :self.ssm.xD]
        self.xx_cov = self.xx_cov[:, :self.ssm.xD]

    def _measurement_update(self, y, *args):
        gain = cho_solve(cho_factor(self.z_cov_pred), self.xz_cov).T
        self.x_mean_fi = self.x_mean_pr + gain.dot(y - self.z_mean_pred)
        self.x_cov_fi = self.x_cov_pr - gain.dot(self.z_cov_pred).dot(gain.T)

    def _smoothing_update(self):
        gain = cho_solve(cho_factor(self.x_cov_pr), self.xx_cov).T
        self.x_mean_sm = self.x_mean_fi + gain.dot(self.x_mean_sm - self.x_mean_pr)
        self.x_cov_sm = self.x_cov_fi + gain.dot(self.x_cov_sm - self.x_cov_pr).dot(gain.T)


class MarginalInference(StateSpaceInference):

    def __init__(self, ssm, transf_dyn, transf_meas):
        super(self, MarginalInference).__init__(ssm, transf_dyn, transf_meas)
        # prior parameter mean and covariance
        self.param_dim = 2 * self.ssm.xD
        self.param_mean = np.zeros(self.param_dim, )  # FIXME: not general, assumes 2*xD parameters
        self.param_cov = np.eye(self.param_dim)
        from transforms.quad import SphericalRadial
        self.param_upts = SphericalRadial.unit_sigma_points(self.param_dim)
        self.param_wts = SphericalRadial.weights(self.param_dim)
        self.param_pts_num = self.param_upts.shape[1]

    def _measurement_update(self, y, *args):
        """
        Computes the posterior state mean and covariance by marginalizing out the moment transform parameters.

        Procedure has two steps:
          1. Compute Laplace approximation of the GPQ parameter posterior
          2. Use fully-symmetric quadrature rule to compute posterior state mean and covariance by marginalizing
             out the GPQ-parameters over the approximated posterior.

        Parameters
        ----------
        y: ndarray
          Measurement at a given time step

        Returns
        -------

        """

        k = args[0]  # time index
        # Mean and covariance of the parameter posterior by Laplace approximation
        self._param_posterior_moments(y, k)

        # Marginalization of moment transform parameters
        param_cov_chol = la.cholesky(self.param_cov)
        param_pts = self.param_mean[:, na] + param_cov_chol.dot(self.param_upts)
        mean = np.zeros(self.param_dim, self.param_pts_num)
        cov = np.zeros(self.param_dim, self.param_dim, self.param_pts_num)

        # Evaluate state posterior with different values of transform parameters
        for i in range(self.param_upts.shape[1]):
            mean[:, i], cov[:, :, i] = self._state_posterior_moments(param_pts[:, i], y, k)

        # Weighted sum of means and covariances approximates Gaussian mixture state posterior
        self.x_mean_fi = np.einsum('ij, j -> i', mean, self.param_wts)
        self.x_cov_fi = np.einsum('ijk, k -> ij', cov, self.param_wts)

    def _smoothing_update(self):
        gain = cho_solve(cho_factor(self.x_cov_pr), self.xx_cov).T
        self.x_mean_sm = self.x_mean_fi + gain.dot(self.x_mean_sm - self.x_mean_pr)
        self.x_cov_sm = self.x_cov_fi + gain.dot(self.x_cov_sm - self.x_cov_pr).dot(gain.T)

    def _state_posterior_moments(self, theta, y, k):
        self._time_update(k, theta)
        gain = cho_solve(cho_factor(self.z_cov_pred), self.xz_cov).T
        mean = self.x_mean_pr + gain.dot(y - self.z_mean_pred)
        cov = self.x_cov_pr - gain.dot(self.z_cov_pred).dot(gain.T)
        return mean, cov

    def _param_log_likelihood(self, theta, y, k):
        """
        l(theta) = p(y_k | theta) = N(y_k | m_k^y(theta), P_k^y(theta))

        Parameters
        ----------
        theta: ndarray
            Vector of transform parameters.
        y: ndarray
            Observation
        k: int
            Time (for time varying dynamics)

        Returns
        -------
            Value of likelihood for given vector of parameters and observation.
        """

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.ssm.q_additive else np.hstack((self.x_mean_fi, self.q_mean))
        cov = self.x_cov_fi if self.ssm.q_additive else block_diag(self.x_cov_fi, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        mean, cov, ccov = self.transf_dyn.apply(self.ssm.dyn_eval, mean, cov, self.ssm.par_fcn(k), theta)
        if self.ssm.q_additive:
            cov += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = mean if self.ssm.r_additive else np.hstack((mean, self.r_mean))
        cov = cov if self.ssm.r_additive else block_diag(cov, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        mean, cov, ccov = self.transf_meas.apply(self.ssm.meas_eval, mean, cov, self.ssm.par_fcn(k), theta)
        if self.ssm.r_additive:
            cov += self.r_cov

        return multivariate_normal.logpdf(y, mean, cov)

    def _param_log_prior(self, theta):
        """
        Prior on transform parameters.

        p(theta) = N(theta | m^theta_k-1, P^theta_k-1)

        Parameters
        ----------
        theta: ndarray
            Vector of transform parameters.

        Notes
        -----
        At the moment, only Gaussian prior is supported. Student-t prior might be implemented in the future.

        Returns
        -------
        p(theta): return type of scipy.stats.multivariate_normal.pdf
            Value of a Gaussian prior PDF.

        """
        return multivariate_normal.logpdf(theta, self.param_mean, self.param_cov)

    def _param_neg_log_posterior(self, theta, y, k):
        """
        Un-normalized negative log-posterior over transform parameters.

        Parameters
        ----------
        theta: ndarray
            Transform parameters
        y: ndarray
            Observation
        k: int
            Time

        Returns
        -------
        x: float
            Evaluation of un-normalized negative logarithm of posterior over transform parameters.
        """
        return -self._param_log_likelihood(theta, y, k) - self._param_log_prior(theta)

    def _param_posterior_moments(self, y, k):
        """
        Laplace approximation of the intractable transform parameter posterior.

        Parameters
        ----------
        y: ndarray
            Observation
        k: int
            Time

        Returns
        -------
        (mean, cov): tuple
            Mean and covariance of the intractable parameter posterior.
        """

        from scipy.optimize import minimize
        opt_res = minimize(self._param_neg_log_posterior, self.param_mean, (y, k), method='BFGS')
        self.param_mean, self.param_cov = opt_res.x, opt_res.hess_inv
