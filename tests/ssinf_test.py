from unittest import TestCase

import numpy as np
import numpy.linalg as la
from numpy import linalg as la

from bqmtran import GPQ, GPQMO
from mtran import SphericalRadialTrunc, MonteCarlo, FullySymmetricStudent
from ssinf import GPQMKalman
from ssmod import Pendulum, UNGM, ReentryRadar, CoordinatedTurnBOT


def sum_of_squares(x, pars, dx=False):
    """Sum of squares test function.

    If x is Gaussian random variable than x.T.dot(x) is chi-squared distributed with mean d and variance 2d,
    where d is the dimension of x.
    """
    if not dx:
        return np.atleast_1d(x.T.dot(x))
    else:
        return np.atleast_1d(2 * x)


def cartesian2polar(x, pars, dx=False):
    return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0])])


class SigmaPointTruncTest(TestCase):
    def test_apply(self):
        d, d_eff = 5, 2
        t = SphericalRadialTrunc(d, d_eff)
        f = cartesian2polar
        mean, cov = np.zeros(d), np.eye(d)
        t.apply(f, mean, cov, None)


class MonteCarloTest(TestCase):
    def test_crash(self):
        d = 1
        tmc = MonteCarlo(d, n=1e4)
        f = UNGM().dyn_eval
        mean = np.zeros(d)
        cov = np.eye(d)
        # does it crash ?
        tmc.apply(f, mean, cov, np.atleast_1d(1.0))

    def test_increasing_samples(self):
        d = 1
        tmc = (
            MonteCarlo(d, n=1e1),
            MonteCarlo(d, n=1e2),
            MonteCarlo(d, n=1e3),
            MonteCarlo(d, n=1e4),
            MonteCarlo(d, n=1e5),
        )
        f = sum_of_squares  # UNGM().dyn_eval
        mean = np.zeros(d)
        cov = np.eye(d)
        # does it crash ?
        for t in tmc:
            print(t.apply(f, mean, cov, np.atleast_1d(1.0)))


class FullySymmetricStudentTest(TestCase):

    def test_symmetric_set(self):

        # 1D points
        dim = 1
        sp = FullySymmetricStudent.symmetric_set(dim, [])
        self.assertEqual(sp.ndim, 2)
        self.assertEqual(sp.shape, (dim, 1))
        sp = FullySymmetricStudent.symmetric_set(dim, [1])
        self.assertEqual(sp.shape, (dim, 2*dim))
        sp = FullySymmetricStudent.symmetric_set(dim, [1, 1])
        self.assertEqual(sp.shape, (dim, 2*dim*(dim-1)))

        # 2D points
        dim = 2
        sp = FullySymmetricStudent.symmetric_set(dim, [])
        self.assertEqual(sp.shape, (dim, 1))
        sp = FullySymmetricStudent.symmetric_set(dim, [1])
        self.assertEqual(sp.shape, (dim, 2*dim))
        sp = FullySymmetricStudent.symmetric_set(dim, [1, 1])
        self.assertEqual(sp.shape, (dim, 2 * dim * (dim - 1)))

        # 3D points
        dim = 3
        sp = FullySymmetricStudent.symmetric_set(dim, [1, 1])
        self.assertEqual(sp.shape, (dim, 2 * dim * (dim - 1)))

    def test_crash(self):
        dim = 1
        mt = FullySymmetricStudent(dim, degree=3)
        f = UNGM().dyn_eval
        mean = np.zeros(dim)
        cov = np.eye(dim)
        # does it crash ?
        mt.apply(f, mean, cov, np.atleast_1d(1.0))

        dim = 2
        mt = FullySymmetricStudent(dim, degree=5)
        f = sum_of_squares
        mean = np.zeros(dim)
        cov = np.eye(dim)
        # does it crash ?
        mt.apply(f, mean, cov, np.atleast_1d(1.0))


class GPQuadTest(TestCase):
    models = [UNGM, Pendulum]

    def test_weights_rbf(self):
        dim = 1
        khyp = np.array([[1, 3]])
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GPQ(dim, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print('wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc))
        self.assertTrue(np.allclose(wc, wc.T), "Covariance weight matrix not symmetric.")
        # print 'GP model variance: {}'.format(tf.model.exp_model_variance())

        dim = 2
        khyp = np.array([[1, 3, 3]])
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GPQ(dim, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print('wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc))
        self.assertTrue(np.allclose(wc, wc.T), "Covariance weight matrix not symmetric.")

    def test_rbf_scaling_invariance(self):
        dim = 5
        ker_par = np.array([[1, 3, 3, 3, 3, 3]])
        tf = GPQ(dim, ker_par)
        w0 = tf._weights([1] + dim * [1000])
        w1 = tf._weights([358.0] + dim * [1000.0])
        self.assertTrue(np.alltrue([np.array_equal(a, b) for a, b in zip(w0, w1)]))

    def test_expected_model_variance(self):
        dim = 2
        ker_par = np.array([[1, 3, 3]])
        tf = GPQ(dim, ker_par, points='sr')
        emv0 = tf.model.exp_model_variance(tf.model.points)
        emv1 = tf.model.exp_model_variance(tf.model.points)
        # expected model variance must be positive even for numerically unpleasant settings
        self.assertTrue(np.alltrue(np.array([emv0, emv1]) >= 0))

    def test_integral_variance(self):
        dim = 2
        ker_par = np.array([[1, 3, 3]])
        tf = GPQ(dim, ker_par, points='sr')
        ivar0 = tf.model.integral_variance(tf.model.points, par=[1, 600, 6])
        ivar1 = tf.model.integral_variance(tf.model.points, par=[1.1, 600, 6])
        # expected model variance must be positive even for numerically unpleasant settings
        self.assertTrue(np.alltrue(np.array([ivar0, ivar1]) >= 0))

    def test_apply(self):
        for ssm in self.models:
            f = ssm().dyn_eval
            dim = ssm.xD
            ker_par = np.hstack((np.ones((1, 1)), 3*np.ones((1, dim))))
            tf = GPQ(dim, ker_par)
            mean, cov = np.zeros(dim, ), np.eye(dim)
            tmean, tcov, tccov = tf.apply(f, mean, cov, np.atleast_1d(1.0))
            print("Transformed moments\nmean: {}\ncov: {}\nccov: {}".format(tmean, tcov, tccov))

            # test positive definiteness
            try:
                la.cholesky(tcov)
            except la.LinAlgError:
                self.fail("Output covariance not positive definite.")

            # test symmetry
            self.assertTrue(np.allclose(tcov, tcov.T), "Output covariance not closely symmetric.")
            # self.assertTrue(np.array_equal(tcov, tcov.T), "Output covariance not exactly symmetric.")


class GPQMOTest(TestCase):
    models = [UNGM, Pendulum]

    def test_weights_rbf(self):
        dim_in, dim_out = 1, 1
        khyp = np.array([[1, 3]])
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GPQMO(dim_in, dim_out, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        self.assertTrue(np.allclose(wc, wc.swapaxes(0, 1).swapaxes(2, 3)), "Covariance weight matrix not symmetric.")

        dim_in, dim_out = 4, 4
        khyp = np.array([[1, 3, 3, 3, 3],
                         [1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 2],
                         [1, 3, 3, 3, 3]])
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GPQMO(dim_in, dim_out, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        self.assertTrue(np.allclose(wc, wc.swapaxes(0, 1).swapaxes(2, 3)), "Covariance weight matrix not symmetric.")

    def test_apply(self):
        ssm = Pendulum()
        f = ssm.dyn_eval
        dim_in, dim_out = ssm.xD, ssm.xD
        ker_par = np.hstack((np.ones((dim_out, 1)), 3*np.ones((dim_out, dim_in))))
        tf = GPQMO(dim_in, dim_out, ker_par)
        mean, cov = np.zeros(dim_in, ), np.eye(dim_in)
        tmean, tcov, tccov = tf.apply(f, mean, cov, np.atleast_1d(1.0))
        print("Transformed moments\nmean: {}\ncov: {}\nccov: {}".format(tmean, tcov, tccov))

        # test positive definiteness
        try:
            la.cholesky(tcov)
        except la.LinAlgError:
            self.fail("Output covariance not positive definite.")

        # test symmetry
        self.assertTrue(np.allclose(tcov, tcov.T), "Output covariance not closely symmetric.")
        # self.assertTrue(np.array_equal(tcov, tcov.T), "Output covariance not exactly symmetric.")

    def test_single_vs_multi_output(self):
        # results of the GPQ and GPQMO should be same if parameters properly chosen, GPQ is a special case of GPQMO
        ssm = ReentryRadar()
        f = ssm.dyn_eval
        dim_in, dim_out = ssm.xD, ssm.xD

        # input mean and covariance
        mean_in, cov_in = ssm.pars['x0_mean'], ssm.pars['x0_cov']

        # single-output GPQ
        ker_par_so = np.hstack((np.ones((1, 1)), 25 * np.ones((1, dim_in))))
        tf_so = GPQ(dim_in, ker_par_so)

        # multi-output GPQ
        ker_par_mo = np.hstack((np.ones((dim_out, 1)), 25 * np.ones((dim_out, dim_in))))
        tf_mo = GPQMO(dim_in, dim_out, ker_par_mo)

        # transformed moments
        # FIXME: transformed covariances different
        mean_so, cov_so, ccov_so = tf_so.apply(f, mean_in, cov_in, ssm.par_fcn(0))
        mean_mo, cov_mo, ccov_mo = tf_mo.apply(f, mean_in, cov_in, ssm.par_fcn(0))

        print('mean delta: {}'.format(np.abs(mean_so - mean_mo).max()))
        print('cov delta: {}'.format(np.abs(cov_so - cov_mo).max()))
        print('ccov delta: {}'.format(np.abs(ccov_so - ccov_mo).max()))

        # results of GPQ and GPQMO should be the same
        self.assertTrue(np.array_equal(mean_so, mean_mo))
        self.assertTrue(np.array_equal(cov_so, cov_mo))
        self.assertTrue(np.array_equal(ccov_so, ccov_mo))


class GPQMarginalizedTest(TestCase):
    def test_init(self):
        ssm = UNGM()
        alg = GPQMKalman(ssm, 'rbf', 'sr')

    def test_time_update(self):
        ssm = UNGM()
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg._time_update(1)
        par_dyn, par_obs = np.array([1, 1]), np.array([1, 1])
        alg._time_update(1, par_dyn, par_obs)

    def test_laplace_approx(self):
        ssm = UNGM()
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        # Random measurement
        y = np.sqrt(10)*np.random.randn(1)
        alg._param_posterior_moments(y, 10)
        # test positive definiteness
        try:
            la.cholesky(alg.param_cov)
        except la.LinAlgError:
            self.fail("Output covariance not positive definite.")

    def test_measurement_update(self):
        ssm = UNGM()
        ssm_state, ssm_observations = ssm.simulate(5)
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg._measurement_update(ssm_observations[:, 0, 0], 1)

    def test_filtering_ungm(self):
        ssm = UNGM()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])

    def test_filtering_pendulum(self):
        ssm = Pendulum()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])
