from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib import pyplot as plt, gridspec, gridspec, gridspec
from numpy import newaxis as na

from ssinf import StateSpaceInference
from utils import multivariate_t

# NOTE : The class should recognize input dimensions of dynamics and observation models separately
# This is because observation models do not always use all the state dimensions, e.g. radar only uses position
# to produce range and bearing measurements, the remaining states (velocity, ...) remain unused. Therefore the moment
# transform for the observation model should have different dimension. I think this approach should be followed by
# all the sigma-point bq. The question is how to compute I/O covariance? Perhaps using the lower-dimensional
# rule with sigma-points extended with zeros to match the state dimension.


class StateSpaceModel(metaclass=ABCMeta):

    xD = None  # state dimension
    zD = None  # measurement dimension
    qD = None  # state noise dimension
    rD = None  # measurement noise dimension

    q_additive = None  # True = state noise is additive, False = non-additive
    r_additive = None

    def __init__(self, **kwargs):
        self.pars = kwargs
        self.zero_q = np.zeros(self.qD)
        self.zero_r = np.zeros(self.rD)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        pass

    @abstractmethod
    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        pass

    @abstractmethod
    def state_noise_sample(self, size=None):
        """
        Sample from a state noise distribution.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    @abstractmethod
    def measurement_noise_sample(self, size=None):
        """
        Sample from a measurement noise distribution.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    @abstractmethod
    def initial_condition_sample(self, size=None):
        """
        Sample from a distribution over the system initial conditions.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    def dyn_eval(self, xq, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xq : 1-D array_like
            Augmented system state
        pars : 1-D array_like
            System dynamics parameters
        dx : bool
            * ``True``: Evaluates derivatives (Jacobian) of the system dynamics
            * ``False``: Evaluates system dynamics
        Returns
        -------
            Evaluated system dynamics or evaluated Jacobian of the system dynamics.
        """

        if self.q_additive:
            assert len(xq) == self.xD
            if dx:
                out = (self.dyn_fcn_dx(xq, self.zero_q, pars).T.flatten())
            else:
                out = self.dyn_fcn(xq, self.zero_q, pars)
        else:
            assert len(xq) == self.xD + self.qD
            x, q = xq[:self.xD], xq[-self.qD:]
            if dx:
                out = (self.dyn_fcn_dx(x, q, pars).T.flatten())
            else:
                out = self.dyn_fcn(x, q, pars)
        return out

    def meas_eval(self, xr, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xr : 1-D array_like
            Augmented system state
        pars : 1-D array_like
            Measurement model parameters
        dx : bool
            * ``True``: Evaluates derivatives (Jacobian) of the measurement model
            * ``False``: Evaluates measurement model
        Returns
        -------
            Evaluated measurement model or evaluated Jacobian of the measurement model.
        """

        if self.r_additive:
            # assert len(xr) == self.xD
            if dx:
                out = (self.meas_fcn_dx(xr, self.zero_r, pars).T.flatten())
            else:
                out = self.meas_fcn(xr, self.zero_r, pars)
        else:
            assert len(xr) == self.xD + self.rD
            x, r = xr[:self.xD], xr[-self.rD:]
            if dx:
                out = (self.meas_fcn_dx(x, r, pars).T.flatten())
            else:
                out = self.meas_fcn(x, r, pars)
        return out

    def check_jacobians(self, h=1e-8):
        """Checks implemented Jacobians.

        Checks that both Jacobians are correctly implemented using numerical approximations.

        Parameters
        ----------
        h : float
            step size in derivative approximations

        Returns
        -------
            Prints the errors and user decides whether they're acceptable.
        """

        nq = self.xD if self.q_additive else self.xD + self.qD
        nr = self.xD if self.r_additive else self.xD + self.rD
        xq, xr = np.random.rand(nq), np.random.rand(nr)
        hq_diag, hr_diag = np.diag(h * np.ones(nq)), np.diag(h * np.ones(nr))
        assert hq_diag.shape == (nq, nq) and hr_diag.shape == (nr, nr)
        xqph, xqmh = xq[:, na] + hq_diag, xq[:, na] - hq_diag
        xrph, xrmh = xr[:, na] + hr_diag, xr[:, na] - hr_diag

        # allocate space for Jacobians
        fph = np.zeros((self.xD, nq))
        hph = np.zeros((self.zD, nr))
        fmh, hmh = fph.copy(), hph.copy()

        # approximate Jacobians by central differences
        par = self.par_fcn(1.0)
        for i in range(nq):
            fph[:, i] = self.dyn_eval(xqph[:, i], par)
            fmh[:, i] = self.dyn_eval(xqmh[:, i], par)
        for i in range(nr):
            hph[:, i] = self.meas_eval(xrph[:, i], par)
            hmh[:, i] = self.meas_eval(xrmh[:, i], par)
        jac_fx = (2 * h) ** -1 * (fph - fmh)
        jac_hx = (2 * h) ** -1 * (hph - hmh)

        # report approximation error
        print("Errors in Jacobians\n{}\n{}".format(np.abs(jac_fx - self.dyn_eval(xq, par, dx=True)),
                                                   np.abs(jac_hx - self.meas_eval(xr, par, dx=True))))

    def simulate(self, steps, mc_sims=1):
        """State-space model simulation.

        SSM simulation starting from initial conditions for a given number of time steps

        Parameters
        ----------
        steps : int
            Number of time steps in state trajectory
        mc_sims : int
            Number of trajectories to simulate (the initial state is drawn randomly)

        Returns
        -------
        tuple
            Tuple (x, z) where both element are of type numpy.ndarray and where:

                * x : 3-D array of shape (self.xD, steps, mc_sims) containing the true system state trajectory
                * z : 3-D array of shape (self.zD, steps, mc_sims) containing simulated measurements of the system state
        """

        # allocate space for state and measurement sequences
        x = np.zeros((self.xD, steps, mc_sims))
        z = np.zeros((self.zD, steps, mc_sims))

        # generate state and measurement noise
        q = self.state_noise_sample((mc_sims, steps))
        r = self.measurement_noise_sample((mc_sims, steps))

        # generate initial conditions, store initial states at k=0
        x0 = self.initial_condition_sample(mc_sims)  # (D, mc_sims)
        x[:, 0, :] = x0

        # simulate SSM `mc_sims` times for `steps` time steps
        for imc in range(mc_sims):
            for k in range(1, steps):
                theta = self.par_fcn(k - 1)
                x[:, k, imc] = self.dyn_fcn(x[:, k-1, imc], q[:, k-1, imc], theta)
                z[:, k, imc] = self.meas_fcn(x[:, k, imc], r[:, k, imc], theta)
        return x, z

    def set_pars(self, key, value):
        self.pars[key] = value

    def get_pars(self, *keys):
        values = []
        for k in keys:
            values.append(self.pars.get(k))
        return values


class GaussianStateSpaceModel(StateSpaceModel):
    """
    State-space model with Gaussian noise and initial conditions.
    """

    def __init__(self, x0_mean=None, x0_cov=None, q_mean=None, q_cov=None, r_mean=None, r_cov=None, q_gain=None):

        # use default value of statistics for Gaussian SSM if None provided
        kwargs = {
            'x0_mean': x0_mean if x0_mean is not None else np.zeros(self.xD),
            'x0_cov': x0_cov if x0_cov is not None else np.eye(self.xD),
            'q_mean': q_mean if q_mean is not None else np.zeros(self.qD),
            'q_cov': q_cov if q_cov is not None else np.eye(self.qD),
            'r_mean': r_mean if r_mean is not None else np.zeros(self.rD),
            'r_cov': r_cov if r_cov is not None else np.eye(self.rD),
            'q_gain': q_gain if q_gain is not None else np.eye(self.qD)
        }
        super(GaussianStateSpaceModel, self).__init__(**kwargs)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        pass

    @abstractmethod
    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        pass

    def state_noise_sample(self, size=None):
        q_mean, q_cov = self.get_pars('q_mean', 'q_cov')
        return np.random.multivariate_normal(q_mean, q_cov, size).T

    def measurement_noise_sample(self, size=None):
        r_mean, r_cov = self.get_pars('r_mean', 'r_cov')
        return np.random.multivariate_normal(r_mean, r_cov, size).T

    def initial_condition_sample(self, size=None):
        x0_mean, x0_cov = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(x0_mean, x0_cov, size).T


class UNGM(GaussianStateSpaceModel):
    """
    Univariate Non-linear Growth Model frequently used as a benchmark.
    """

    # CLASS VARIABLES are shared by all instances of this class
    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1
    q_additive = True
    r_additive = True

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGM, self).__init__(**kwargs)
        self.set_pars('x0_mean', np.atleast_1d(x0_mean))
        self.set_pars('x0_cov', np.atleast_2d(x0_cov))
        self.set_pars('q_mean', np.atleast_1d(q_mean))
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_mean', np.atleast_1d(r_mean))
        self.set_pars('r_cov', np.atleast_2d(r_cov))
        self.set_pars('q_gain', np.eye(1))

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * x[0]])


class UNGMnonadd(GaussianStateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    # CLASS VARIABLES are shared by all instances of this class
    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = False
    r_additive = False

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGMnonadd, self).__init__(**kwargs)
        self.set_pars('x0_mean', np.atleast_1d(x0_mean))
        self.set_pars('x0_cov', np.atleast_2d(x0_cov))
        self.set_pars('q_mean', np.atleast_1d(q_mean))
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_mean', np.atleast_1d(r_mean))
        self.set_pars('r_cov', np.atleast_2d(r_cov))

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * q[0] * np.cos(1.2 * pars[0])])

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * r[0] * x[0] ** 2])

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])


class Pendulum(GaussianStateSpaceModel):

    xD = 2  # state dimension
    zD = 1  # measurement dimension
    qD = 2
    rD = 1

    q_additive = True
    r_additive = True

    g = 9.81  # gravitation constant

    def __init__(self, x0_mean=np.array([1.5, 0]), x0_cov=0.01 * np.eye(2), r_cov=np.array([[0.1]]), dt=0.01):
        self.dt = dt
        assert x0_mean.shape == (self.xD,) and x0_cov.shape == (self.xD, self.xD)
        assert r_cov.shape == (self.zD, self.zD)
        req_kwargs = {
            'x0_mean': np.atleast_1d(x0_mean),
            'x0_cov': np.atleast_2d(x0_cov),
            'q_mean': np.zeros(self.qD),
            'q_cov': 0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.atleast_2d(r_cov),
            'q_gain': np.eye(2)
        }
        super(Pendulum, self).__init__(**req_kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] + x[1] * self.dt, x[1] - self.g * self.dt * np.sin(x[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.array([np.sin(x[0])]) + r

    def par_fcn(self, time):
        pass  # pendulum model does not have time-varying parameters

    def dyn_fcn_dx(self, x, q, pars):
        return np.array([[1.0, self.dt],
                         [-self.g * self.dt * np.cos(x[0]), 1.0]])

    def meas_fcn_dx(self, x, r, pars):
        return np.array([np.cos(x[0]), 0.0])


class VanDerPol(GaussianStateSpaceModel):

    xD = 2  # state dimension
    zD = 1  # measurement dimension
    qD = 2  # state noise dimension
    rD = 1  # measurement noise dimension

    q_additive = True  # True = state noise is additive, False = non-additive
    r_additive = True

    def __init__(self, dt=0.1, alpha=1):
        self.dt = dt
        self.alpha = alpha
        req_kwargs = {
            'x0_mean': np.zeros(self.xD),
            'x0_cov': np.diag([6.3e-4, 2.2e-4]),
            'q_mean': np.zeros(self.qD),
            'q_cov': np.diag([2.62e-2, 8e-3]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[3e-3]]),
            'q_gain': np.eye(2)
        }
        super(VanDerPol, self).__init__(**req_kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] + self.dt*x[1], x[1] + self.dt*(self.alpha*x[1]*(1 - x[0]**2) - x[0])])

    def meas_fcn(self, x, r, pars):
        return np.array([x[1]]) + r

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass


class FrequencyDemodulation(GaussianStateSpaceModel):
    """
    Frequence demodulation experiment from [1]_

    The objective is to estimate the frequency message $$ x_1 = \omega $$ from noisy in-phase and quadrature
    observations.


    References
    ==========
    .. [1] Pakki, K., et al., (2011) Cubature Information Filter and its Applications, Proceedings of the ACC 2011
    """

    xD = 2
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = True

    # system (process) parameters
    mu = 0.9
    lam = 0.99

    def __init__(self):
        kwargs = {
            'x0_mean': np.zeros(self.xD),  # strange to have zero initial frequency
            'x0_cov': np.eye(self.xD),
            'q_mean': np.zeros(self.qD),
            'q_cov': np.eye(self.qD),
            'r_mean': np.zeros(self.rD),
            'r_cov': 2e-3 * np.eye(self.rD),
            'q_gain': np.eye(self.qD),
        }
        super(FrequencyDemodulation, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([self.mu * x[0], np.arctan(self.lam * x[1] + x[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.array([np.cos(x[1]), np.sin(x[1])]) + r

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass


class CoordinatedTurnBOT(GaussianStateSpaceModel):
    """
    Bearings only target tracking in 2D using multiple sensors as in [3]_.

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 4  # measurement dimension == # sensors
    qD = 5
    rD = 4  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=0.1, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        sensor_pos :
            sensor [x, y] positions in rows
        """
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        kwargs = {
            'x0_mean': np.array([1000, 300, 1000, 0, -3.0 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([100, 10, 100, 10, 0.1]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array(
                [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                 [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                 [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                 [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                 [0, 0, 0, 0, self.rho_2 * self.dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': 10e-3 * np.eye(self.rD)  # 10e-3 rad == 10 mrad
        }
        super(CoordinatedTurnBOT, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om
        d = (1 - np.cos(om * self.dt)) / om
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        return np.arctan(a / b) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class CoordinatedTurnRadar(GaussianStateSpaceModel):
    """
    Maneuvering target tracking using radar measurements .

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 4  # measurement dimension == # sensors
    qD = 5
    rD = 4  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.5, 1e-6  # noise intensities

    def __init__(self, dt=0.2, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        sensor_pos :
            sensor [x, y] positions in rows
        """
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        kwargs = {
            'x0_mean': np.array([130, 25, -20, 1, -4 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([5, 5, 2e4, 10, 1e-7]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array(
                [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                 [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                 [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                 [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                 [0, 0, 0, 0, self.rho_2 * self.dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': 1e-2 * np.eye(self.rD)  # 1e-2 rad == 10 mrad
        }
        super(CoordinatedTurnRadar, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om
        d = (1 - np.cos(om * self.dt)) / om
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        return np.arctan(a / b) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class ReentryRadar(GaussianStateSpaceModel):
    """
    Radar tracking of the reentry vehicle as described in [1]_.
    Vehicle is entering Earth's atmosphere at high altitude and with great speed, ground radar is tracking it.

    State
    -----
    [px, py, vx, vy, x5]
    (px, py) - position,
    (vx, vy) - velocity,
    x5 - aerodynamic parameter

    Measurements
    ------------
    range and bearing


    References
    ----------
    .. [1] Julier, S. J., & Uhlmann, J. K. (2004). Unscented Filtering and Nonlinear Estimation.
           Proceedings of the IEEE, 92(3), 401-422

    """

    xD = 5
    zD = 2  # measurement dimension
    qD = 3
    rD = 2  # measurement noise dimension

    q_additive = True
    r_additive = True

    R0 = 6374  # Earth's radius
    H0 = 13.406
    Gm0 = 3.9860e5
    b0 = -0.59783  # ballistic coefficient of a typical vehicle
    sx, sy = R0, 0  # radar location

    def __init__(self, dt=0.1):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        self.dt = dt
        kwargs = {
            'x0_mean': np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': self.dt ** -1 * np.array(
                [[2.4064e-5, 0, 0],
                 [0, 2.4064e-5, 0],
                 [0, 0, 1e-6]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[1e-6, 0],
                               [0, 0.17e-3 ** 2]]),
            'q_gain': np.vstack((np.zeros((2, 3)), np.eye(3)))
        }
        super(ReentryRadar, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        # scaled ballistic coefficient
        b = self.b0 * np.exp(x[4])
        # distance from center of the Earth
        R = np.sqrt(x[0] ** 2 + x[1] ** 2)
        # speed
        V = np.sqrt(x[2] ** 2 + x[3] ** 2)
        # drag force
        D = b * np.exp((self.R0 - R) / self.H0) * V
        # gravity force
        G = -self.Gm0 / R ** 3
        return np.array([x[0] + self.dt * x[2],
                         x[1] + self.dt * x[3],
                         x[2] + self.dt * (D * x[2] + G * x[0]) + q[0],
                         x[3] + self.dt * (D * x[3] + G * x[1]) + q[1],
                         x[4] + q[2]])

    def meas_fcn(self, x, r, pars):
        # range
        rng = np.sqrt((x[0] - self.sx) ** 2 + (x[1] - self.sy) ** 2)
        # bearing
        theta = np.arctan2((x[1] - self.sy), (x[0] - self.sx))
        return np.array([rng, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class ReentryRadarSimple(GaussianStateSpaceModel):
    """
    Radar tracking of the reentry vehicle as described in [1]_.
    Vehicle is entering Earth's atmosphere at high altitude and with great speed, ground radar is tracking it.

    State
    -----
    [px, py, vx, vy, x5]
    (px, py) - position,
    (vx, vy) - velocity,
    x5 - aerodynamic parameter

    Measurements
    ------------
    range and bearing


    References
    ----------
    .. [1] Julier, S. J., & Uhlmann, J. K. (2004). Unscented Filtering and Nonlinear Estimation.
           Proceedings of the IEEE, 92(3), 401-422

    """

    xD = 3
    zD = 1  # measurement dimension
    qD = 3
    rD = 1  # measurement noise dimension
    q_additive = True
    r_additive = True

    R0 = 6371  # Earth's radius [km]  #2.0925e7  # Earth's radius [ft]
    # radar location: 30km (~100k ft) above the surface, radar-to-body horizontal range
    sx, sy = 30, 30
    Gamma = 1/6.096

    def __init__(self, dt=0.1):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        self.dt = dt
        kwargs = {
            'x0_mean': np.array([90, 6, 1.7]),  # km, km/s
            'x0_cov': np.diag([0.3048**2, 1.2192**2, 10]),  # km^2, km^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[0.03048**2]]),
            'q_factor': np.vstack(np.eye(3))
        }
        super(ReentryRadarSimple, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] - self.dt * x[1] + q[0],
                         x[1] - self.dt * np.exp(-self.Gamma*x[0])*x[1]**2*x[2] + q[1],
                         x[2] + q[2]])

    def meas_fcn(self, x, r, pars):
        # range
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class StudentStateSpaceModel(StateSpaceModel):

    def __init__(self, x0_mean=None, x0_cov=None, x0_dof=None, q_mean=None, q_cov=None, q_dof=None, q_gain=None,
                 r_mean=None, r_cov=None, r_dof=None):
        """
        State-space model where the noises are Student distributed.
        Takes covariances instead of scale matrices.

        Parameters
        ----------
        x0_mean
        x0_cov
        x0_dof
        q_mean
        q_cov
        q_dof
        q_gain
        r_mean
        r_cov
        r_dof
        """
        kwargs = {
            'x0_mean': x0_mean if x0_mean is not None else np.zeros(self.xD),
            'x0_cov': x0_cov if x0_cov is not None else np.eye(self.xD),
            'x0_dof': x0_dof if x0_dof is not None else 4.0,  # desired DOF
            'q_mean': q_mean if q_mean is not None else np.zeros(self.qD),
            'q_cov': q_cov if q_cov is not None else np.eye(self.qD),
            'q_gain': q_gain if q_gain is not None else np.eye(self.qD),
            'q_dof': q_dof if q_dof is not None else 4.0,
            'r_mean': r_mean if r_mean is not None else np.zeros(self.rD),
            'r_cov': r_cov if r_cov is not None else np.eye(self.rD),
            'r_dof': r_dof if r_dof is not None else 4.0,
        }
        super(StudentStateSpaceModel, self).__init__(**kwargs)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        pass

    @abstractmethod
    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        pass

    def state_noise_sample(self, size=None):
        q_mean, q_cov, q_dof = self.get_pars('q_mean', 'q_cov', 'q_dof')
        return multivariate_t(q_mean, q_cov, q_dof, size).T

    def measurement_noise_sample(self, size=None):
        r_mean, r_cov, r_dof = self.get_pars('r_mean', 'r_cov', 'r_dof')
        return multivariate_t(r_mean, r_cov, r_dof, size).T

    def initial_condition_sample(self, size=None):
        x0_mean, x0_cov, x0_dof = self.get_pars('x0_mean', 'x0_cov', 'x0_dof')
        return multivariate_t(x0_mean, x0_cov, x0_dof, size).T


def ungm_demo():
    steps = 100
    mc_simulations = 50
    ssm = UNGM(q_cov=10, r_cov=.1)
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.show()


def ungm_nonadd_demo():
    steps = 100
    mc_simulations = 50
    ssm = UNGMnonadd(q_cov=10, r_cov=.1)
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.show()


def pendulum_demo():
    steps = 500
    mc_simulations = 100
    ssm = Pendulum()
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.plot(x[0, ...], color='b', lw=2, alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')

    # plt.figure()
    # plt.plot(x[0, :, 0], x[1, :, 0])
    plt.show()


def vanderpol_demo():
    steps = 250
    mc_simulations = 10
    ssm = VanDerPol()
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.subplot(211)
    plt.plot(x[0, ...], color='b', lw=2, alpha=0.25, label='x_1[k]')
    plt.subplot(212)
    plt.plot(x[1, ...], color='b', lw=2, alpha=0.25, label='x_1[k]')

    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')

    # plt.figure()
    # plt.plot(x[0, :, 0], x[1, :, 0])
    plt.show()


def frequency_demodulation_demo():
    steps = 100
    mc_simulations = 50
    ssm = FrequencyDemodulation()
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.plot(x[0, ...], color='b', alpha=0.15, label='state')
    plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.show()


def bot_demo(steps=100, mc_sims=1):
    ssm = CoordinatedTurnBOT()
    x, z = ssm.simulate(steps, mc_sims=mc_sims)
    # plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.figure()
    g = gridspec.GridSpec(4, 1)
    plt.subplot(g[:2, 0])
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[2, :, i], alpha=0.35, color='b')
    plt.subplot(g[2, 0])
    plt.plot(x[0, :, :], 'b', alpha=0.25)
    plt.subplot(g[3, 0])
    plt.plot(x[2, :, :], 'b', alpha=0.25)
    plt.show()


def reentry_demo(steps=100, mc_sims=1):
    ssm = ReentryRadar()
    x, z = ssm.simulate(steps, mc_sims=mc_sims)
    # plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.figure()
    g = gridspec.GridSpec(2, 4)
    plt.subplot(g[:, :2])
    # Earth surface w/ radar position
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(ssm.R0 * np.cos(t), ssm.R0 * np.sin(t), 'r', ssm.sx, ssm.sy, 'ko')
    # vehicle trajectory
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='b')
    plt.subplot(g[:, 2:], polar=True)
    plt.plot((z[1, :, 0]), z[0, :, 0], 'ko')
    plt.show()


# TODO: make the code independent of the model, code suitable would be useful in tests
def ungm_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = UNGM(x0_mean=0.1, x0_cov=5.0)
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 50, 5
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    rmse_filter = np.zeros(mc)
    rmse_smoother = np.zeros(mc)
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[imc] = np.sqrt(np.mean((x[..., imc] - mean_f) ** 2, axis=1))
        rmse_smoother[imc] = np.sqrt(np.mean((x[..., imc] - mean_s) ** 2, axis=1))
        filt.reset()
    # print average filter/smoother RMSE
    print("Filter RMSE: {:.4f}".format(np.asscalar(rmse_filter.mean())))
    print("Smoother RMSE: {:.4f}".format(np.asscalar(rmse_smoother.mean())))
    # plot one realization of the system trajectory, measurements and filtered/smoothed state estimate
    plt.figure()
    time = list(range(1, time_steps))
    plt.plot(x[0, :, 0], color='r', ls='--', label='true state')
    plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], color='b', label='filtered estimate')
    plt.fill_between(time,
                     mean_f[0, 1:] - 2 * np.sqrt(cov_f[0, 0, 1:]),
                     mean_f[0, 1:] + 2 * np.sqrt(cov_f[0, 0, 1:]),
                     color='b', alpha=0.15)
    plt.plot(mean_s[0, ...], color='g', label='smoothed estimate')
    plt.fill_between(time,
                     mean_s[0, 1:] - 2 * np.sqrt(cov_s[0, 0, 1:]),
                     mean_s[0, 1:] + 2 * np.sqrt(cov_s[0, 0, 1:]),
                     color='g', alpha=0.25)
    plt.legend()
    plt.show()


def frequency_demodulation_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = FrequencyDemodulation()
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 500, 100
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    mse_filter = np.zeros((system.xD, time_steps, mc))
    mse_smoother = np.zeros((system.xD, time_steps, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        mse_filter[..., imc] = (x[..., imc] - mean_f) ** 2
        mse_smoother[..., imc] = (x[..., imc] - mean_s) ** 2
        filt.reset()
    # position and velocity RMSE (time_steps, MC)
    rmse_filter = np.sqrt(mse_filter)
    rmse_smoother = np.sqrt(mse_smoother)
    # print average filter/smoother RMSE
    print("Filter stats:\n=============")
    print("Time-averaged RMSE: {}".format((rmse_filter.mean(axis=(1, 2)))))
    print("Smoother stats:\n===============")
    print("Time-averaged RMSE: {}".format((rmse_smoother.mean(axis=(1, 2)))))
    # plot one realization of the system trajectory, measurements and filtered/smoothed state estimate
    time = list(range(0, time_steps))
    rmse_time_f = rmse_filter.mean(axis=2)
    plt.figure()
    plt.plot(time, rmse_time_f[0, :], time, rmse_time_f[1, :])
    plt.show()
    # time = range(1, time_steps)
    # plt.plot(x[0, :, 0], color='r', ls='--', label='true state')
    # plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    # plt.plot(mean_f[0, ...], color='b', label='filtered estimate')
    # plt.fill_between(time,
    #                  mean_f[0, 1:] - 2 * np.sqrt(cov_f[0, 0, 1:]),
    #                  mean_f[0, 1:] + 2 * np.sqrt(cov_f[0, 0, 1:]),
    #                  color='b', alpha=0.15)
    # plt.plot(mean_s[0, ...], color='g', label='smoothed estimate')
    # plt.fill_between(time,
    #                  mean_s[0, 1:] - 2 * np.sqrt(cov_s[0, 0, 1:]),
    #                  mean_s[0, 1:] + 2 * np.sqrt(cov_s[0, 0, 1:]),
    #                  color='g', alpha=0.25)
    # plt.legend()
    # plt.show()


def pendulum_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = Pendulum()
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 500, 100
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    rmse_filter = np.zeros((system.xD, mc))
    rmse_smoother = np.zeros((system.xD, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_f) ** 2, axis=1))
        rmse_smoother[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_s) ** 2, axis=1))
        filt.reset()
    # print average filter/smoother RMSE
    print("Filter RMSE: {}".format((rmse_filter.mean(axis=1))))
    print("Smoother RMSE: {}".format((rmse_smoother.mean(axis=1))))
    # plot one realization of the system trajectory, measurements and filtered/smoothed state estimate
    plt.figure()
    time = list(range(1, time_steps))
    plt.plot(x[0, :, 0], color='r', ls='--', label='true state')
    plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], color='b', label='filtered estimate')
    plt.fill_between(time,
                     mean_f[0, 1:] - 2 * np.sqrt(cov_f[0, 0, 1:]),
                     mean_f[0, 1:] + 2 * np.sqrt(cov_f[0, 0, 1:]),
                     color='b', alpha=0.15)
    plt.plot(mean_s[0, ...], color='g', label='smoothed estimate')
    plt.fill_between(time,
                     mean_s[0, 1:] - 2 * np.sqrt(cov_s[0, 0, 1:]),
                     mean_s[0, 1:] + 2 * np.sqrt(cov_s[0, 0, 1:]),
                     color='g', alpha=0.25)
    plt.legend()
    plt.show()


def bot_filter_demo(filt_class, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    # sensor positions
    sen_pos = np.array([[1000, 900],
                        [600, 1000],
                        [1200, 800],
                        [1200, 1000]])
    # sen_pos = np.vstack((np.eye(2), -np.eye(2)))
    system = CoordinatedTurnBOT(sensor_pos=sen_pos)
    # create filter object, pass in additional kwargs
    filt = filt_class(system, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 130, 100
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    rmse_filter = np.zeros((system.xD, mc))
    rmse_smoother = np.zeros((system.xD, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_f) ** 2, axis=1))
        rmse_smoother[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_s) ** 2, axis=1))
        filt.reset()
    # print average filter/smoother RMSE
    print("Filter RMSE: {}".format((rmse_filter.mean(axis=1))))
    print("Smoother RMSE: {}".format((rmse_smoother.mean(axis=1))))
    # plot one realization of the system trajectory, measurements and filtered/smoothed state estimate
    plt.figure()
    time = list(range(1, time_steps))
    plt.plot(sen_pos[:, 0], sen_pos[:, 1], 'ko')
    plt.plot(x[0, :, imc], x[2, :, imc], color='r', ls='--', label='true state')
    # plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], mean_f[2, ...], color='b', label='filtered estimate')
    plt.plot(mean_s[0, ...], mean_s[2, ...], color='g', label='smoothed estimate')
    plt.legend()
    plt.show()


def reentry_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = ReentryRadar()
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 750, 50
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    mse_filter = np.zeros((system.xD, time_steps, mc))
    mse_smoother = np.zeros((system.xD, time_steps, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        mse_filter[..., imc] = (x[..., imc] - mean_f) ** 2
        mse_smoother[..., imc] = (x[..., imc] - mean_s) ** 2
        filt.reset()
    # position and velocity RMSE (time_steps, MC)
    rmse_filter_pos = np.sqrt(mse_filter[:2, ...].sum(axis=0))
    rmse_filter_vel = np.sqrt(mse_filter[2:4, ...].sum(axis=0))
    rmse_smoother_pos = np.sqrt(mse_smoother[:2, ...].sum(axis=0))
    rmse_smoother_vel = np.sqrt(mse_smoother[2:4, ...].sum(axis=0))
    # print average filter/smoother RMSE
    print("Filter stats:\n=============")
    print("Time-averaged RMSE (position): {}".format((rmse_filter_pos.mean())))
    print("Time-averaged RMSE (velocity): {}".format((rmse_filter_vel.mean())))
    print("Smoother stats:\n===============")
    print("Time-averaged RMSE (position): {}".format((rmse_smoother_pos.mean())))
    print("Time-averaged RMSE (velocity): {}".format((rmse_smoother_vel.mean())))
    # plot filter and smoother RMSE of position/velocity in time (MC averages)
    time = np.linspace(0, time_steps * system.dt, time_steps)
    g = gridspec.GridSpec(4, 2)
    plt.figure('Position and velocity RMSE (averaged over {} MC simulations)'.format(mc))
    ax = plt.subplot(g[1, 0])
    ax.set_title('Position: x[k]')
    ax.plot(time, mse_filter[0, ...].mean(axis=1))
    ax.plot(time, mse_smoother[0, ...].mean(axis=1))
    ax = plt.subplot(g[1, 1])
    ax.set_title('Position: y[k]')
    ax.plot(time, mse_filter[1, ...].mean(axis=1))
    ax.plot(time, mse_smoother[1, ...].mean(axis=1))
    ax = plt.subplot(g[0, :])
    ax.set_title('Position: \sqrt{x[k]^2 + y[k]^2}')
    ax.plot(time, rmse_filter_pos.mean(axis=1))
    ax.plot(time, rmse_smoother_pos.mean(axis=1))
    ax = plt.subplot(g[3, 0])
    ax.set_title('Velocity: \dot{x}[k]')
    ax.plot(time, mse_filter[2, ...].mean(axis=1))
    ax.plot(time, mse_smoother[2, ...].mean(axis=1))
    ax = plt.subplot(g[3, 1])
    ax.set_title('Velocity: \dot{y}[k]')
    ax.plot(time, mse_filter[3, ...].mean(axis=1))
    ax.plot(time, mse_smoother[3, ...].mean(axis=1))
    ax = plt.subplot(g[2, :])
    ax.set_title('Speed: \sqrt{\dot{x}[k]^2 + \dot{y}[k]^2}')
    ax.plot(time, rmse_filter_vel.mean(axis=1))
    ax.plot(time, rmse_smoother_vel.mean(axis=1))
    # plt.legend()
    plt.show()

