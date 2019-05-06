from Graph import Potential
import numpy as np
from math import pow, pi, e, sqrt, exp


class TablePotential(Potential):
    def __init__(self, table, symmetric=False):
        Potential.__init__(self, symmetric=symmetric)
        self.table = table

    def get(self, parameters):
        return self.table[parameters]


class GaussianPotential(Potential):
    def __init__(self, mu, sig, w=1):
        Potential.__init__(self, symmetric=False)
        self.mu = np.array(mu)
        self.sig = np.matrix(sig)
        self.sig_inv = self.sig.I
        det = np.linalg.det(self.sig)
        p = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coefficient = w / (pow(2 * pi, p * 0.5) * pow(det, 0.5))

    def get(self, parameters, use_coef=False):
        x_mu = np.matrix(np.array(parameters) - self.mu)
        coef = self.coefficient if use_coef else 1.
        return coef * pow(e, -0.5 * (x_mu * self.sig_inv * x_mu.T))

    def to_log_potential(self):
        return GaussianLogPotential(self.mu, self.sig_inv)

        # def __eq__(self, other):
        #     return np.all(self.mu == other.mu) and np.all(self.sig == other.sig)  # self.w shouldn't make a difference
        #
        # def __hash__(self):
        #     return hash((self.mu, self.sig))


class GaussianLogPotential:
    def __init__(self, mu, sig_inv):
        self.mu = np.array(mu)
        self.sig_inv = np.array(sig_inv)  # must be ndarray

    def __call__(self, args):
        """

        :param args: list of n tensors or numpy arrays; must all have the same shape, or must be broadcastable to the
        largest common shape (e.g., if args have shapes [1, 3, 1] and [2, 1, 3], they'll first be broadcasted to having
        shape [2, 3, 3], then the result will be computed element-wise and will also have shape [2, 3, 3]
        :return:
        """
        n = len(args)
        mu = self.mu
        sig_inv = self.sig_inv
        import tensorflow as tf
        if n == 1:
            res = -0.5 * sig_inv[0, 0] * (args[0] - mu[0]) ** 2
        else:
            import utils
            if any(isinstance(a, (tf.Variable, tf.Tensor)) for a in args):
                backend = tf
            else:
                backend = np
            args = utils.broadcast_arrs_to_common_shape(args, backend=backend)
            v = backend.stack(args)  # n x ...
            args_ndim = len(v.shape) - 1
            mu = backend.reshape(mu, [n] + [1] * args_ndim)
            sig_inv = backend.reshape(sig_inv, [n, n] + [1] * args_ndim)
            diff = v - mu
            outer_prods = diff[None, ...] * diff[:, None, ...]  # n x n x ...
            if backend is tf:
                quad_form = tf.reduce_sum(outer_prods * sig_inv, axis=[0, 1])
            else:
                quad_form = np.sum(outer_prods * sig_inv, axis=(0, 1))
            res = -.5 * quad_form

        return res

        # def __eq__(self, other):
        #     return np.all(self.mu == other.mu) and np.all(self.sig_inv == other.sig_inv)
        #
        # def __hash__(self):
        #     return hash((self.mu, self.sig_inv))


class LinearGaussianPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-(parameters[1] - self.coeff * parameters[0]) ** 2 * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )

    def to_log_potential(self):
        # convert to equivalent Gaussian
        mu = np.zeros(2)
        a = self.coeff
        sig_inv = np.array([[a ** 2, -a], [-a, 1.]]) / self.sig
        return GaussianLogPotential(mu, sig_inv)


class X2Potential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-self.coeff * parameters[0] ** 2 * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )

    def to_log_potential(self):
        # convert to equivalent Gaussian
        mu = np.zeros(1)
        sig_inv = np.zeros([1, 1])
        sig_inv[0, 0] = self.coeff / self.sig
        return GaussianLogPotential(mu, sig_inv)


class XYPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=True)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-self.coeff * parameters[0] * parameters[1] * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )

    def to_log_potential(self):
        # convert to equivalent Gaussian
        mu = np.zeros(2)
        sig_inv = np.array([[0., 0.5], [0.5, 0.]]) * self.coeff / self.sig
        return GaussianLogPotential(mu, sig_inv)


class ImageNodePotential(Potential):
    def __init__(self, mu, sig):
        Potential.__init__(self, symmetric=True)
        self.mu = mu
        self.sig = sig

    def get(self, parameters):
        u = (parameters[0] - parameters[1] - self.mu) / self.sig
        return exp(-u * u * 0.5) / (2.506628274631 * self.sig)


class ImageEdgePotential(Potential):
    def __init__(self, distant_cof, scaling_cof, max_threshold):
        Potential.__init__(self, symmetric=True)
        self.distant_cof = distant_cof
        self.scaling_cof = scaling_cof
        self.max_threshold = max_threshold
        self.v = pow(e, -self.max_threshold / self.scaling_cof)

    def get(self, parameters):
        d = abs(parameters[0] - parameters[1])
        if d > self.max_threshold:
            return d * self.distant_cof + self.v
        else:
            return d * self.distant_cof + pow(e, -d / self.scaling_cof)
