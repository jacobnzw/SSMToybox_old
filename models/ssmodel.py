import numpy as np
# TODO: Abstract Base Classes to enforce the requirements of the base class on the derived classes.


class StateSpaceModel(object):

    xD = None  # state dimension
    zD = None  # measurement dimension
    qD = None  # state noise dimension
    rD = None  # measurement noise dimension
    q_additive = None  # True = state noise is additive, False = non-additive
    r_additive = None
    # lists the keyword arguments currently required by the StateSpaceModel class
    _required_kwargs_ = 'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov'

    def __init__(self, **kwargs):
        self.pars = kwargs

    def dyn_fcn(self, x, q, *args):
        # system dynamics
        raise NotImplementedError

    def meas_fcn(self, x, r, *args):
        # state measurement model
        raise NotImplementedError

    def par_fcn(self, time):
        # describes how parameter value depends on time (for time varying systems)
        raise NotImplementedError

    # TODO: implment Jacobians for ExtendedKalman,
    # TODO: could approximate Jacobians with differences
    def dyn_eval(self, xq, *args):
        if self.q_additive:
            assert len(xq) == self.xD
            return self.dyn_fcn(xq, 0, *args)
        else:
            x, q = xq[:self.xD], xq[-self.qD:]
            return self.dyn_fcn(x, q, *args)

    def meas_eval(self, xr, *args):
        if self.r_additive:
            assert len(xr) == self.xD
            return self.meas_fcn(xr, 0, *args)
        else:
            x, r = xr[:self.xD], xr[-self.rD:]
            return self.meas_fcn(x, r, *args)

    def simulate(self, steps, mc_sims=1):
        """
        General implementation of the SSM simulation starting from initial conditions for a given number of time steps
        :param steps: number of time steps in state trajectory
        :param mc_sims: number of trajectories to simulate (the initial state is drawn randomly)
        :return: arrays with simulated state trajectories and measurements
        """
        x0_mean, x0_cov, q_mean, q_cov, r_mean, r_cov = self.get_pars(
                'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov'
        )
        x = np.empty((self.xD, steps, mc_sims))
        z = np.empty((self.zD, steps, mc_sims))
        q = np.random.multivariate_normal(q_mean, q_cov, size=(mc_sims, steps)).T
        r = np.random.multivariate_normal(r_mean, r_cov, size=(mc_sims, steps)).T
        x0 = np.random.multivariate_normal(x0_mean, x0_cov, size=mc_sims).T  # (D, mc_sims)
        x[:, 0, :] = x0  # store initial states at k=0
        for imc in xrange(mc_sims):
            for k in xrange(1, steps):
                theta = self.par_fcn(k)
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
