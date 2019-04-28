import tensorflow as tf
import numpy as np


def set_path():  # to facilitate importing from pardir
    import sys
    sys.path.append('..')


def set_seed(seed=0):
    tf.set_random_seed(seed)
    np.random.seed(seed)


try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


def softmax(a, axis=None):
    """
    Compute exp(a)/sumexp(a); relying on scipy logsumexp implementation.
    :param a: ndarray/tensor
    :param axis: axis to sum over; default (None) sums over everything; use negative number to specify axis in reverse
    (last to first) order (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html)
    :return:
    """
    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    p = a - lse
    p = np.exp(p)
    return p


def weighted_feature_fun(feature_fun, weight):
    # return lambda vs: weight * feature_fun(vs)
    def wf(args):
        return weight * feature_fun(args)

    return wf


def get_log_potential_fun_from_MLNPotential(potential):
    """
    Extract the log potential function from a given MLNPotential object
    :param potential: an instance of MLNPotential
    :return:
    """
    f = weighted_feature_fun(potential.formula, potential.w)
    f.potential_obj = potential  # ref for convenience
    return f


def get_log_potential_fun_from_GaussianPotential(potential):
    """
    Extract the log potential function from a given GaussianPotential object
    :param potential:
    :return:
    """
    mu = potential.mu
    sig_inv = potential.inv
    log_coef = np.log(potential.coefficient)  # optional; can be absorbed into log Z
    p = len(mu)  # num args

    def f(args):
        """
        Gaussian log potential fun; -0.5 ((x-mu)^T * Sig_inv * (x-mu)) + log_coef
        :param args:
        :return:
        """
        quad_form = 0
        diffs = [None] * p
        for i in range(p):
            diffs[i] = args[i] - mu[i]

        for i in range(p):
            for j in range(i, p):
                sig_inv_coef = sig_inv[i, j]
                if i == j:  # diagonal terms only counted once
                    quad_form -= 0.5 * sig_inv_coef * diffs[i] ** 2
                else:  # strictly upper triangular terms counted twice
                    quad_form -= sig_inv_coef * (diffs[i] * diffs[j])

        res = quad_form + log_coef
        return res

    f.potential_obj = potential  # ref for convenience
    return f


def get_log_potential_fun_from_Potential(potential):
    """
    Extract the log potential function (callable) from a given Potential object
    :param potential:
    :return:
    """
    from MLNPotential import MLNPotential
    from Potential import GaussianPotential
    allowed_Potentials = [MLNPotential, GaussianPotential]
    assert type(potential) in allowed_Potentials, 'currenctly only support %s' % (str(allowed_Potentials))
    if type(potential) == MLNPotential:
        log_potential_fun = get_log_potential_fun_from_MLNPotential(potential)
    elif type(potential) == GaussianPotential:
        log_potential_fun = get_log_potential_fun_from_GaussianPotential(potential)
    return log_potential_fun


def outer_prod_einsum_equation(ndim, common_first_ndims=0):
    """
    Get the einsum equation for n-dimensional outer/tensor product, of n vectors v1, v2, ..., vn
    Extended to handle tensors (thought of as containers of vectors) for efficient simultaneous evaluations; input
    tensors can differ in shapes only in the last dimension, over which products will be taken, e.g., if arrs have
    shapes [K x V1, K x V2, ..., K x Vn], common_first_ndims should be set to 1, and the result einsum will be of
    shape K x V1 x V2 x ... x Vn
    :param ndim: number of tensors to perform einsum over
    :param common_first_ndims: default=0, will get the tensor product of vectors; if = 1, will get the tensor-product of
    matrices with shared 0th dimension; similarly for more than 1 common_first_ndims
    :return:
    """
    # TODO: allow arbitrary number of shared first p dimensions (currently p=1 for mats), like with expand_dims_for_grid
    prefix_indices = 'abcdefg'
    indices = 'ijklmnopqrstuvwxyz'
    assert ndim < len(indices) and common_first_ndims < len(prefix_indices), "Ran out of letters for einsum indices!"
    prefix_indices = prefix_indices[:common_first_ndims]  # empty string if common_first_ndims==0
    indices = indices[:ndim]

    lhs = ','.join([prefix_indices + c for c in indices])  # "abi,abj,abk,..."
    rhs = prefix_indices + indices  # "abijk..."
    return lhs + '->' + rhs


def expand_dims_for_grid(arrs, first_ndims_to_keep=0):
    """
    https://stackoverflow.com/a/22778484/4115369
    Extended to tensors with arbitrary number of dimensions (but same rank); e.g., if arrs are matrices of shapes
    K x V1, K x V2, ..., K x Vn, then use first_ndims_to_keep=1, and arrs will be reshaped to be K x V1 x 1 x ... x 1,
    K x 1 x V2 x ... x 1, ..., K x 1 x 1 x ... x Vn, each shape of length (n+1)
    :param arrs:
    :return:
    """
    # return [x[(None,) * i + (slice(None),) + (None,) * (len(arrs) - i - 1)] for i, x in enumerate(arrs)]

    first_ndims_slices = (slice(None),) * first_ndims_to_keep  # these dims won't be expanded (will get full slices :)
    return [x[first_ndims_slices + (None,) * i + (slice(None),) + (None,) * (len(arrs) - i - 1)] for i, x in
            enumerate(arrs)]


def eval_fun_grid(fun, arrs, sep_args=False):
    """
    Evaluate a function R^n -> R on the tensor (outer) product of vectors [v1, v2, ..., vn];
    Extended to tensors (thought of as containers of vectors) for efficient simultaneous evaluations, e.g., if arrs have
    shapes [K x V1, K x V2, ..., K x Vn], the result will be of shape K x V1 x V2 x ... x Vn, and should be equivalent
    to concatenating the results of evaluating on K ndgrids each of shape V1 x V2 ... x Vn (such that the kth grid is
    formed by taking the kth rows of all the arrs).
    :param fun: scalar fun that takes an iterable of n args
    :param arrs: iterable (tuple/list) of tf/np arrays, length n
    :param sep_args: whether fun takes iterable (instead of *args) as arguments, i.e., f(xyz) vs f(x,y,z)
    :return:
    """
    arrs_shapes_except_last = [a.shape[:-1] for a in arrs]
    arrs_shapes_except_last = [tuple(s.as_list()) if isinstance(s, tf.TensorShape) else s
                               for s in arrs_shapes_except_last]  # convert tf tensor shapes (np shapes already tuples)
    assert len(set(arrs_shapes_except_last)) == 1, 'Shapes of input tensors can only differ in the last dimension!'
    common_first_ndims = len(arrs_shapes_except_last[0])  # form grid based on the last dimension
    expanded_arrs = expand_dims_for_grid(arrs, common_first_ndims)

    if sep_args:
        res = fun(*expanded_arrs)
    else:
        res = fun(expanded_arrs)  # should evaluate on the Cartesian product (ndgrid) of axes by broadcasting
    return res


def calc_num_grad(vs, obj, sess, delta=1e-4):
    """
    Compute numerical gradient using tensorflow.python.ops.gradient_checker.
    e.g., calc_num_grad(Mu, bfe, sess) gives gradient of the scalar bfe objective w.r.t. Mu, at Mu's current value
    :param vs: list of tf vars, or a single tf var
    :param obj:
    :param sess:
    :param delta:
    :return:
    """
    from tensorflow.python.ops import gradient_checker
    if not isinstance(vs, list):
        vs = [vs]
    scalar_obj = np.prod(obj.shape) == 1
    grads = []
    with sess.as_default():
        for v in vs:
            val = sess.run(v)
            grad = gradient_checker._compute_numeric_jacobian(v, v.shape, val, obj, obj.shape, delta, {v.name: val})
            if scalar_obj:
                grad = grad.reshape(v.shape)
            grads.append(grad)

    if len(grads) == 1:
        out = grads[0]
    else:
        out = grads
    return out
