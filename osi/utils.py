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
    Compute exp(a)/sumexp(a); relying on scipy logsumexp implementation to avoid numerical issues.
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


def get_unique_subsets(items, key):
    unique_properties = list(set(key(item) for item in items))  # order may change, might be an issue
    subsets_with_unique_properties = [None] * len(unique_properties)
    for i, unique_prop in enumerate(unique_properties):
        subset = list(filter(lambda item: key(item) == unique_prop, items))
        subsets_with_unique_properties[i] = subset
    return subsets_with_unique_properties, unique_properties


def set_log_potential_funs(factors):
    """
    Set the log_potential_fun attribute of factors, ensuring that factors with the same potentials will have the same
    log potentials. This is a pre-processing step for OSI, as the graph construction/compression code typically work
    with potential objects instead of log potentials.
    :param factors: list of factors (whose log_potential_fun will be modified if not already set (i.e., is None))
    :return:
    """
    # unique_potentials = set(f.potential for f in factors)
    # factors_with_unique_potentials = [None] * len(unique_potentials)  # list of lists of factors
    # for i, unique_potential in enumerate(unique_potentials):
    #     like_factors = list(filter(lambda f: f.potential == unique_potential, factors))
    #     factors_with_unique_potentials[i] = like_factors
    #
    #     if any(factor.log_potential_fun is None
    #            for factor in like_factors):  # skip if log_potential_fun already defined
    #         unique_log_potential_fun = unique_potential.to_log_potential()
    #         for factor in like_factors:
    #             factor.log_potential_fun = unique_log_potential_fun  # reference the same object

    factors_with_unique_potentials, unique_potentials = get_unique_subsets(factors, lambda f: f.potential)
    for i, unique_potential in enumerate(unique_potentials):
        like_factors = factors_with_unique_potentials[i]
        if any(factor.log_potential_fun is None for factor in
               like_factors):  # skip if log_potential_fun already defined
            unique_log_potential_fun = unique_potential.to_log_potential()
            for factor in like_factors:
                factor.log_potential_fun = unique_log_potential_fun  # reference the same object


def get_partial_function(fun, n, partial_args_vals):
    """

    :param fun: a function that takes n args
    :param n: number of arguments of fun
    :param partial_args_vals: a dict mapping indices of a subset of args to their values.
    Example: fun takes (x0, x1, x2) as args; if we let pfun = get_partial_function(fun, {2: 0.9}), then pfun defines a
    new function over (x0, x1) alone, s.t. pfun([x0, x1]) = fun([x0, x1, 0.9]).
    :return:
    """

    def pfun(args):
        assert len(args) + len(partial_args_vals) == n
        orig_args = [None] * n
        j = 0
        for i in range(n):
            if i in partial_args_vals:
                orig_args[i] = partial_args_vals[i]
            else:
                orig_args[i] = args[j]
                j += 1
        return fun(orig_args)

    return pfun


def get_conditional_gaussian(mu, Sig, obs_args_vals):
    """
    Get the mean and covariance matrix of a conditional Gaussian distribution p(x_a | x_b), given the joint distribution
    over (x_a, x_b). Based on section 2.3.1, p 87 of PRML textbook.
    :param mu:  joint mean
    :param Sig: joint covmat
    :param obs_args_vals: a dict mapping indices of variables in the observed subector (x_b) to their obs values.
    :return:
    """
    Sig = np.asarray(Sig)
    n = len(mu)
    b_ind = np.array(list(obs_args_vals.keys()), dtype=int)
    b_val = np.array([obs_args_vals[i] for i in b_ind])
    a_ind = np.setdiff1d(np.arange(n), b_ind)
    mu_b = mu[b_ind]
    mu_a = mu[a_ind]
    Sig_bb = Sig[np.ix_(b_ind, b_ind)]
    Sig_bb_inv = np.linalg.inv(Sig_bb)
    Sig_aa = Sig[np.ix_(a_ind, a_ind)]
    Sig_ab = Sig[np.ix_(a_ind, b_ind)]
    Sig_ba = Sig_ab.T

    cond_mu = mu_a + Sig_ab @ (Sig_bb_inv @ (b_val - mu_b))  # (2.81)
    cond_Sig = Sig_aa - Sig_ab @ Sig_bb_inv @ Sig_ba  # (2.82)
    return cond_mu, cond_Sig


def condition_factors_on_evidence(factors, evidence):
    """

    :param factors: an iterable of factors
    :param evidence: a dict mapping rvs (Graph.RV objects) to numerical values
    :return: a new list of factors (original factors won't be modified) reduced to the context of given evidence
    """
    from MLNPotential import MLNPotential
    from Potential import GaussianPotential
    from copy import copy
    cond_factors = []
    for factor in factors:
        if any(rv in evidence for rv in factor.nb):  # create new factor (and its potential/log_potential_fun)
            remaining_rvs = []
            n = len(factor.nb)
            partial_args_vals = {}
            for i, rv in enumerate(factor.nb):
                if rv in evidence:
                    partial_args_vals[i] = evidence[rv]
                else:
                    remaining_rvs.append(rv)
            f = copy(factor)  # may need to construct a new factor object instead
            f.uncond_factor = factor
            f.nb = remaining_rvs

            if len(f.nb) == 0:  # entire factor is observed, then it just reduces to a constant
                f.potential = None
                f.log_potential_fun = lambda x: factor.log_potential_fun([evidence[rv] for rv in factor.nb])
            else:
                potential = factor.potential
                if isinstance(potential, GaussianPotential):
                    mu, sig = get_conditional_gaussian(potential.mu, potential.sig, partial_args_vals)
                    pot = GaussianPotential(mu=mu, sig=sig)
                    log_pot = pot.to_log_potential()
                elif isinstance(potential, MLNPotential):
                    formula = get_partial_function(potential.formula, n, partial_args_vals)
                    pot = MLNPotential(formula=formula, w=potential.w)
                    log_pot = pot.to_log_potential()
                else:  # may not be able to construct the conditional potential in closed-form
                    pot = copy(potential)
                    pot.potential.get = get_partial_function(potential.get, n, partial_args_vals)
                    log_pot = get_partial_function(factor.log_potential_fun, n, partial_args_vals)

                if hasattr(potential, 'symmetric'):
                    pot.symmetric = potential.symmetric

                f.potential = pot
                f.log_potential_fun = log_pot
        else:
            f = factor
        cond_factors.append(f)
    return cond_factors


def get_conditional_mrf(factors, rvs, evidence):
    """

    :param factors: an iterable of factors
    :param rvs: an iterable of rvs
    :param evidence: a dict mapping rvs (Graph.RV objects) to numerical values
    :return: g, containing unobserved rvs and conditional factors with scopes reduced to the unobserved rvs
    """
    cond_factors = condition_factors_on_evidence(factors, evidence)
    cond_factors = list(filter(lambda f: len(f.nb) > 0, cond_factors))  # keep only non-empty factors
    remaining_rvs = [rv for rv in rvs if rv not in evidence]

    from Graph import Graph
    g = Graph()
    g.rvs = remaining_rvs
    g.factors = cond_factors
    g.init_nb()  # update .nb attributes of rvs
    return g


def get_prec_mat_from_gaussian_mrf(factors, rvs_list):
    """
    Get the precision (information) matrix (i.e., inverse of covmat) of a Gaussian MRF.
    :param factors:
    :param rvs_list: a list of N rvs (order is important; will determine the resulting matrix)
    :return:
    """
    from Potential import GaussianPotential
    N = len(rvs_list)
    rvs_idx = {rv: i for (i, rv) in enumerate(rvs_list)}
    prec_mat = np.zeros([N, N], dtype='float')
    for factor in factors:
        pot = factor.potential
        assert isinstance(pot, GaussianPotential)
        J = np.linalg.inv(pot.sig)  # local precision matrix
        n = len(factor.nb)
        rvs_pos = [rvs_idx[rv] for rv in factor.nb]
        for i in range(n):
            for j in range(n):
                prec_mat[rvs_pos[i], rvs_pos[j]] += J[i, j]
    return prec_mat, rvs_idx


def check_diagonal_dominance(mat, strict=True):
    shape = mat.shape
    assert shape[0] == shape[1], 'must input square matrix'
    tmp = np.copy(mat)
    np.fill_diagonal(tmp, 0)
    non_diag_abs_sums = np.sum(np.abs(tmp), axis=1)
    abs_diag = np.abs(np.diag(mat))
    if strict:
        res = np.all(abs_diag > non_diag_abs_sums)
    else:
        res = np.all(abs_diag >= non_diag_abs_sums)
    return res


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
    assert ndim > 0
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


def expand_dims_for_fun_grid(arrs):
    """
    Preprocessing helper for eval_fun_grid. arrs should be the list of arguments to a function R^n -> R.
    See eval_fun_grid for comments.
    :param arrs:
    :return:
    """
    arrs_shapes_except_last = [a.shape[:-1] for a in arrs]
    arrs_shapes_except_last = [tuple(s.as_list()) if isinstance(s, tf.TensorShape) else s
                               for s in arrs_shapes_except_last]  # convert tf tensor shapes (np shapes already tuples)
    assert len(set(arrs_shapes_except_last)) == 1, 'Shapes of input tensors can only differ in the last dimension!'
    common_first_ndims = len(arrs_shapes_except_last[0])  # form grid based on the last dimension
    expanded_arrs = expand_dims_for_grid(arrs, common_first_ndims)
    return expanded_arrs


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
    # arrs_shapes_except_last = [a.shape[:-1] for a in arrs]
    # arrs_shapes_except_last = [tuple(s.as_list()) if isinstance(s, tf.TensorShape) else s
    #                            for s in arrs_shapes_except_last]  # convert tf tensor shapes (np shapes already tuples)
    # assert len(set(arrs_shapes_except_last)) == 1, 'Shapes of input tensors can only differ in the last dimension!'
    # common_first_ndims = len(arrs_shapes_except_last[0])  # form grid based on the last dimension
    # expanded_arrs = expand_dims_for_grid(arrs, common_first_ndims)
    expanded_arrs = expand_dims_for_fun_grid(arrs)

    if sep_args:
        res = fun(*expanded_arrs)
    else:
        res = fun(expanded_arrs)  # should evaluate on the Cartesian product (ndgrid) of axes by broadcasting
    return res


def broadcast_arrs_to_common_shape(arrs):
    """
    Return a list of arrays that are all broadcasted to a common shape.
    e.g., if arrs have shapes [1, 3, 1] and [2, 1, 3], result will be a list of 2 arrays both with shape [2, 3, 3]
    :param arrs: list of arrays/tensors broadcastable to the same largest shape.
    :return:
    """
    # if len(arrs) == 1:
    #     return arrs
    arrs_shapes = [tuple(map(int, arg.shape)) for arg in arrs]
    if len(set(arrs_shapes)) == 1:  # all have the same shapes
        out = arrs
    else:
        if isinstance(arrs[0], (tf.Variable, tf.Tensor)):
            backend = tf
        else:
            backend = np

        assert len(set([len(s) for s in arrs_shapes])) == 1, 'arrs should have the same number of dimensions'
        arrs_shapes = np.array(arrs_shapes)
        # n, arrs_ndim = arrs_shapes.shape
        common_shape = tuple(np.max(arrs_shapes, axis=0))
        out = [backend.broadcast_to(arg, common_shape) for arg in arrs]  # now all have the same shape

    return out


def calc_numerical_grad(vs, obj, sess, delta=1e-4):
    """
    Compute numerical gradient using tensorflow.python.ops.gradient_checker.
    e.g., calc_numerical_grad(Mu, bfe, sess) gives gradient of the scalar bfe objective w.r.t. Mu, at Mu's current value
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
