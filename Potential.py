from Graph import Potential
import numpy as np
from math import pow, pi, e, sqrt, exp


def mu_prec_to_quad_params(mu, prec):
    mu, prec = np.asarray(mu), np.asarray(prec)
    A = -0.5 * prec
    b = prec @ mu
    c = -0.5 * np.dot(mu, b)
    return A, b, c


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
        self.prec = self.sig.I
        det = np.linalg.det(self.sig)
        p = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coefficient = w / (pow(2 * pi, p * 0.5) * pow(det, 0.5))

    def get(self, parameters, use_coef=False):
        x_mu = np.matrix(np.array(parameters) - self.mu)
        coef = self.coefficient if use_coef else 1.
        return coef * pow(e, -0.5 * (x_mu * self.prec * x_mu.T))

    def get_quadratic_params(self):
        return mu_prec_to_quad_params(self.mu, self.prec)

    def to_log_potential(self):
        return QuadraticLogPotential(*self.get_quadratic_params())

        # def __eq__(self, other):
        #     return np.all(self.mu == other.mu) and np.all(self.sig == other.sig)  # self.w shouldn't make a difference
        #
        # def __hash__(self):
        #     return hash((self.mu, self.sig))


class QuadraticPotential(Potential):
    """
    exp(x^T A x + b^T x + c)
    """

    def __init__(self, A, b, c):
        Potential.__init__(self, symmetric=False)
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = c
        self.log_potential = QuadraticLogPotential(A, b, c)

    def to_log_potential(self):
        return self.log_potential

    def get(self, args, ignore_const=True):
        return e ** self.log_potential(args, ignore_const)

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


class QuadraticLogPotential:
    def __init__(self, A, b, c=0):
        """
        Implement the function x^T A x + b^T x + c, over n variables.
        :param A: n x n arr
        :param b: n arr
        :param c: scalar
        """
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, args, ignore_const=True):
        """

        :param args: list of n tensors or numpy arrays; must all have the same shape, or must be broadcastable to the
        largest common shape (e.g., if args have shapes [1, 3, 1] and [2, 1, 3], they'll first be broadcasted to having
        shape [2, 3, 3], then the result will be computed element-wise and will also have shape [2, 3, 3]
        :return:
        """
        n = len(args)
        A = self.A
        b = self.b
        c = self.c
        if ignore_const:
            c = 0
        import tensorflow as tf
        if n == 1:
            res = A[0, 0] * args[0] ** 2 + b[0] * args[0]
        else:
            import utils
            if any(isinstance(a, (tf.Variable, tf.Tensor)) for a in args):
                backend = tf
            else:
                backend = np
            args = utils.broadcast_arrs_to_common_shape(args, backend=backend)
            v = backend.stack(args)  # n x ...
            args_ndim = len(v.shape) - 1
            b = backend.reshape(b, [n] + [1] * args_ndim)
            A = backend.reshape(A, [n, n] + [1] * args_ndim)
            outer_prods = v[None, ...] * v[:, None, ...]  # n x n x ...
            if backend is tf:
                res = tf.reduce_sum(outer_prods * A, axis=[0, 1]) + tf.reduce_sum(b * v, axis=0)
            else:
                res = np.sum(outer_prods * A, axis=(0, 1)) + np.sum(b * v, axis=0)
        if c != 0:
            res += c
        return res


class GaussianLogPotential:
    def __init__(self, mu, prec):
        self.mu = np.array(mu)
        self.prec = np.array(prec)  # must be ndarray

    def __call__(self, args):
        """

        :param args: list of n tensors or numpy arrays; must all have the same shape, or must be broadcastable to the
        largest common shape (e.g., if args have shapes [1, 3, 1] and [2, 1, 3], they'll first be broadcasted to having
        shape [2, 3, 3], then the result will be computed element-wise and will also have shape [2, 3, 3]
        :return:
        """
        n = len(args)
        mu = self.mu
        prec = self.prec
        import tensorflow as tf
        if n == 1:
            res = -0.5 * prec[0, 0] * (args[0] - mu[0]) ** 2
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
            prec = backend.reshape(prec, [n, n] + [1] * args_ndim)
            diff = v - mu
            outer_prods = diff[None, ...] * diff[:, None, ...]  # n x n x ...
            if backend is tf:
                quad_form = tf.reduce_sum(outer_prods * prec, axis=[0, 1])
            else:
                quad_form = np.sum(outer_prods * prec, axis=(0, 1))
            res = -.5 * quad_form

        return res

        # def __eq__(self, other):
        #     return np.all(self.mu == other.mu) and np.all(self.prec == other.prec)
        #
        # def __hash__(self):
        #     return hash((self.mu, self.prec))


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

    def get_quadratic_params(self):
        # get params of an equivalent quadratic log potential
        mu = np.zeros(2)
        a = self.coeff
        prec = np.array([[a ** 2, -a], [-a, 1.]]) / self.sig
        return mu_prec_to_quad_params(mu, prec)

    def to_log_potential(self):
        return QuadraticLogPotential(*self.get_quadratic_params())


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

    def get_quadratic_params(self):
        # get params of an equivalent quadratic log potential
        mu = np.zeros(1)
        prec = np.zeros([1, 1])
        prec[0, 0] = self.coeff / self.sig
        return mu_prec_to_quad_params(mu, prec)

    def to_log_potential(self):
        return QuadraticLogPotential(*self.get_quadratic_params())


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

    def get_quadratic_params(self):
        # get params of an equivalent quadratic log potential
        mu = np.zeros(2)
        prec = np.array([[0., 0.5], [0.5, 0.]]) * self.coeff / self.sig
        return mu_prec_to_quad_params(mu, prec)

    def to_log_potential(self):
        return QuadraticLogPotential(*self.get_quadratic_params())


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
