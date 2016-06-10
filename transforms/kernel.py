from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve


# TODO: documentation


class Kernel(object):
    __metaclass__ = ABCMeta
    # list of strings of supported hyperparameters
    _hyperparameters_ = None

    def __init__(self, dim, hypers, jitter):
        self.dim = dim
        self.jitter = jitter
        # check if supplied hypers are all supported by the kernel
        if hypers is not None:
            if not np.alltrue([hyp in hypers.keys() for hyp in self._hyperparameters_]):
                hypers = None
        # use default hypers if unspecified or if given some unsupported hyperparameter
        self.hypers = self._get_default_hyperparameters(dim) if hypers is None else hypers

    @staticmethod
    def _cho_inv(A, b=None):
        # inversion of PD matrix A using Cholesky decomposition
        if b is None:
            b = np.eye(A.shape[0])
        return cho_solve(cho_factor(A), b)

    # evaluation
    @abstractmethod
    def eval(self, x1, x2=None, hyp=None, diag=False):
        raise NotImplementedError

    # TODO: methods could return their byproducts, such as kernel matrix
    # TODO: other methods could optionally accept them to save computation
    def eval_inv_dot(self, x, hyp=None, b=None):
        # if b=None returns inverse of K
        return Kernel._cho_inv(self.eval(x, hyp=hyp) + self.jitter * np.eye(x.shape[1]), b)

    def eval_chol(self, x, hyp=None):
        return la.cholesky(self.eval(x, hyp=hyp) + self.jitter * np.eye(x.shape[1]))

    # expectations
    @abstractmethod
    def exp_x_kx(self, x):
        raise NotImplementedError

    @abstractmethod
    def exp_x_xkx(self, x):
        raise NotImplementedError

    @abstractmethod
    def exp_x_kxx(self):
        raise NotImplementedError

    @abstractmethod
    def exp_xy_kxy(self):
        raise NotImplementedError

    @abstractmethod
    def exp_x_kxkx(self, x):
        raise NotImplementedError

    # derivatives
    @abstractmethod
    def der_hyp(self, x, hyp0):
        # evaluates derivative of the kernel matrix at hyp0; x is data, now acting as parameter
        raise NotImplementedError

    @abstractmethod
    def _get_default_hyperparameters(self, dim):
        raise NotImplementedError


class RBF(Kernel):
    _hyperparameters_ = ['alpha', 'el']

    def __init__(self, dim, hypers=None, jitter=1e-8):
        super(RBF, self).__init__(dim, hypers, jitter)
        self.alpha = self.hypers['alpha']
        el = self.hypers['el']
        if not np.isscalar(el):
            if len(el) == 1 and dim > 1:
                # if el is a list/tuple/array w/ 1 element and dim > 1
                el = el[0] * np.ones(dim, )
        else:
            # turn scalar el into vector
            el = el * np.ones(dim, )
        self.el = el
        # pre-computation for convenience
        self.lam = np.diag(self.el ** 2)
        self.inv_lam = np.diag(self.el ** -2)
        self.sqrt_inv_lam = np.diag(self.el ** -1)
        self.eye_d = np.eye(dim)

    def __str__(self):  # TODO: improve string representation
        return '{} {}'.format(self.__class__.__name__, self.hypers.update({'jitter': self.jitter}))

    def eval(self, x1, x2=None, hyp=None, diag=False):
        # x1.shape = (D, N), x2.shape = (D, M)
        if x2 is None:
            x2 = x1
        # hyp (D+1,) array_like
        if hyp is not None:
            # use provided hypers
            alpha, sqrt_inv_lam = hyp[0], np.diag(hyp[1:] ** -1)
        else:
            # use hypers provided during init
            alpha, sqrt_inv_lam = self.alpha, self.sqrt_inv_lam

        x1 = sqrt_inv_lam.dot(x1)
        x2 = sqrt_inv_lam.dot(x2)
        if diag:  # only diagonal of kernel matrix
            assert x1.shape == x2.shape
            dx = x1 - x2
            return np.exp(2 * np.log(alpha) - 0.5 * np.sum(dx * dx, axis=0))
        else:
            return np.exp(2 * np.log(alpha) - 0.5 * self._maha(x1.T, x2.T))

    def exp_x_kx(self, x):
        # a.k.a. kernel mean map w.r.t. standard Gaussian PDF
        c = self.alpha ** 2 * (la.det(self.inv_lam + self.eye_d)) ** -0.5
        xl = la.inv(self.lam + self.eye_d).dot(x)
        return c * np.exp(-0.5 * np.sum(x * xl, axis=0))

    def exp_x_xkx(self, x):
        mu_q = la.inv(self.lam + self.eye_d).dot(x)
        q = self.exp_x_kx(x)
        return q[na, :] * mu_q

    def exp_x_kxx(self):
        return self.alpha ** 2

    def exp_xy_kxy(self):
        return self.alpha ** 2 * la.det(2 * self.inv_lam + self.eye_d) ** -0.5

    def exp_x_kxkx(self, x):
        inv_r = la.inv(2 * self.inv_lam + self.eye_d)
        xi = self.sqrt_inv_lam.dot(x)
        xi = 2 * np.log(self.alpha) - 0.5 * np.sum(xi * xi, axis=0)
        x = self.inv_lam.dot(x)
        n = (xi[:, na] + xi[na, :]) + 0.5 * self._maha(x.T, -x.T, V=inv_r)
        return la.det(inv_r) ** 0.5 * np.exp(n)

    def der_hyp(self, x, hyp0):  # K as kwarg would save computation (would have to be evaluated w/ hyp0)
        # hyp0: array_like [alpha, el_1, ..., el_D]
        # x: (D, N)
        alpha, el = hyp0[0], hyp0[1:]
        K = self.eval(x, hyp=hyp0)
        # derivative w.r.t. alpha (N,N)
        d_alpha = 2 * alpha ** -1 * K
        # derivatives w.r.t. el_1, ..., el_D (N,N,D)
        d_el = (x[:, na, :] - x[:, :, na]) ** 2 * (el ** -3)[:, na, na] * K[na, :, :]
        return np.concatenate((d_alpha[..., na], d_el.T), axis=2)

    def _get_default_hyperparameters(self, dim):
        return {'alpha': 1.0, 'el': 1.0 * np.ones(dim, )}

    def _maha(self, x, y, V=None):
        """
        Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.
        :param x: (n, d) matrix of row vectors
        :param y: (n, d) matrix of row vectors
        :param V: weight matrix (d, d), if V=None, V=eye(d) is used
        :return:
        """
        if V is None:
            V = np.eye(x.shape[1])
        x2V = np.sum(x.dot(V) * x, 1)
        y2V = np.sum(y.dot(V) * y, 1)
        return (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)


class Affine(Kernel):
    def __init__(self, dim, hypers):
        super(Affine, self).__init__(dim, hypers)

    def eval(self, x1, x2=None, hyp=None, diag=False):
        pass

    def exp_x_kx(self, x):
        pass

    def exp_x_xkx(self, x):
        pass

    def exp_x_kxx(self):
        pass

    def exp_xy_kxy(self):
        pass

    def exp_x_kxkx(self, x):
        pass

    def _get_default_hyperparameters(self, dim):
        pass
