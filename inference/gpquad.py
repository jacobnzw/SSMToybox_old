import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQuad
from transforms.quad import Unscented


class GPQuadKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        unit_sp_f = Unscented.unit_sigma_points(nq, np.sqrt(nq + 1))
        unit_sp_h = Unscented.unit_sigma_points(nr, np.sqrt(nr + 1))
        hypers_f = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nq, 1)), 'noise_var': 1e-8}
        hypers_h = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nr, 1)), 'noise_var': 1e-8}
        tf = GPQuad(unit_sp_f, hypers_f)
        th = GPQuad(unit_sp_h, hypers_h)
        super(GPQuadKalman, self).__init__(tf, th, sys)


def main():
    from models.ungm import UNGM
    system = UNGM(q_cov=10, r_cov=1)
    print "q_additive: {}, r_additive: {}".format(system.q_additive, system.r_additive)

    time_steps = 100
    x, z = system.simulate(time_steps, 1)  # get some data from the system

    filt = GPQuadKalman(system)
    mean_f, cov_f = filt.forward_pass(z[..., 0])
    mean_s, cov_s = filt.backward_pass()

    rmse_filter = np.sqrt(((x[..., 0] - mean_f) ** 2).mean(axis=1))
    rmse_smoother = np.sqrt(((x[..., 0] - mean_s) ** 2).mean(axis=1))
    print "Filter RMSE: {:.4f}".format(np.asscalar(rmse_filter))
    print "Smoother RMSE: {:.4f}".format(np.asscalar(rmse_smoother))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x[0, :, 0], color='r', ls='--', label='true state')
    plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], color='b', label='filtered estimate')
    plt.fill_between(range(0, time_steps),
                     mean_f[0, ...] - 2 * np.sqrt(cov_f[0, 0, :]),
                     mean_f[0, ...] + 2 * np.sqrt(cov_f[0, 0, :]),
                     color='b', alpha=0.15)
    plt.plot(mean_s[0, ...], color='g', label='smoothed estimate')
    plt.fill_between(range(0, time_steps),
                     mean_s[0, ...] - 2 * np.sqrt(cov_s[0, 0, :]),
                     mean_s[0, ...] + 2 * np.sqrt(cov_s[0, 0, :]),
                     color='g', alpha=0.25)
    plt.legend()


if __name__ == '__main__':
    main()
