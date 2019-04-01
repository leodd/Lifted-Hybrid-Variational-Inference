# from utils import outer_prod_einsum_equation
from scipy.special import roots_hermite
import tensorflow as tf
import numpy as np
import utils


def hfactor_belief_expectation_coefs_points(factor, T=None, dtype='float64'):
    """
    Get the coefficients and evaluation points needed for computing expectation of a scalar-valued function over the
    hybrid factor (for all K mixture components simultaneously)
    :param factor:
    :param T: number of quad points (only needed if factor contains cont node); for now assume the same for all cnodes
    :return: (coefs, axes): coefs, axes are lists of matricies, of shapes [[K, V1], [K, V2], ..., [K, Vn]], where n
    is the number of nodes in the factor, Vi = T if node i is cont, or is the number of states if node i is discrete;
    to compute expectation w.r.t. the kth component (fully factorized) distribution, define the tensor of coefficients
    C_k := \bigotimes_{i=1}^n coefs[i][k, :], the tensor of evaluation points E_k = \bigotimes_{i=1}^n axes[i][k, :],
    then the kth expectation is \langle vec(C_k), vec(f(E_k)) \rangle; the total expectation w.r.t. the mixture is
    obtained by taking a weighted mixture (by w_k) of K component expectations.
    The list of the kth rows of axes mats gives the axes needed to construct evaluation grid for computing the kth
    component-wise expectation
    """
    coefs = []
    axes = []
    if factor.domain_type in ('c', 'h'):
        weights, axes = roots_hermite(T)  # assuming Gaussian for now
        ghq_coef = (np.pi) ** (-0.5)  # from change-of-var
        weights = ghq_coef * weights  # let's fold ghq_coef into the quadrature weights, so no need to worry about it later
        # weights = tf.constant(weights, dtype=dtype)
        for rv in factor.nb:
            if rv.domain_type == 'c':
                K = rv.belief_params_[0].shape[0]
                # weights_KT = tf.tile(tf.reshape(weights, [1, -1]), [K, 1])  # K x T
                weights_KT = np.tile(np.reshape(weights, [1, -1]), [K, 1])  # K x T
                break

    for rv in factor.nb:
        if rv.domain_type == 'd':  # discrete
            c = rv.belief_params_[0]  # K x dstates[i] matrix (tf)
            p = np.tile(np.reshape(rv.values, [1, -1]), [K, 1])  # K x dstates
        elif rv.domain_type == 'c':  # cont, assuming Gaussian for now
            c = weights_KT
            mean, var = rv.belief_params_[0], rv.belief_params_[1]
            mean_K1 = tf.reshape(mean, [K, 1])
            var_K1 = tf.reshape(var, [K, 1])
            p = (2 * var_K1) ** 0.5 * axes + mean_K1  # K x T
        else:
            raise NotImplementedError
        coefs.append(c)
        axes.append(p)

    return coefs, axes


def eval_hfactor_belief(factor, axes, w):
    """

    :param factor:
    :param axes: list of mats [K x V1, K x V2, ..., K x Vn]; we allow the flexibility to evaluate on K > 1 ndgrids
    simultaneously
    :param w:
    :return: a [K x V1 x V2 x ... x Vn] tensor, whose (k, v1, ..., vn)th coordinate is the mixture belief evaluated on
    the point (axes[0][k,v1], axes[1][k,v2], ..., axes[n][k,vn])
    """
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), mats=True)

    res = []
    # TODO: 1. get rid of outer loop; 2. maybe replace multiplication with addition in log-domain? no need?
    K = np.prod(w.shape)
    for k in range(K):  # for mixture belief b(E_k) on the kth grid E_k = \bigotimes_{i=1}^n axes[i][k, :]
        comp_probs = []
        for n, rv in enumerate(factor.nb):
            if rv.domain_type == 'd':  # discrete
                comp_prob = rv.belief_params_[
                    'probs']  # assuming the states of Xn are sorted, so p_kn(rv.states) = p_kn
            elif rv.domain_type == 'c':  # cont, assuming Gaussian for now
                # mean, var = rv.belief_params_['mean'], rv.belief_params_['var']
                # mean_K1 = tf.reshape(mean, [-1, 1])
                # var_inv = 1 / var
                # var_K1 = tf.reshape(var, [-1, 1])
                # var_inv_K1 = tf.reshape(var_inv, [-1, 1])
                mean_K1 = rv.belief_params_['mean_K1']
                var_inv_K1 = rv.belief_params_['var_inv_K1']
                comp_prob = (2 * np.pi) ** (-0.5) * tf.sqrt(var_inv_K1) * \
                            tf.exp(-0.5 * (axes[n][k] - mean_K1) ** 2 * var_inv_K1)
            else:
                raise NotImplementedError
            comp_probs.append(comp_prob)

        # multiple all dimensions together, then weighted sum by w; this gives
        joint_comp_probs = tf.einsum(equation=einsum_eq, *comp_probs)  # K x V1 x V2 x ... Vn
        w_broadcast = tf.reshape(w, [-1] + [1] * len(comp_probs))  # K x 1 x 1 ... x 1
        res.append(tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0, keepdims=True))  # 1 x V1 x V2 x ... Vn

    return tf.concat(res, 0)  # K x V1 x V2 x ... Vn


def eval_dfactor_belief(factor, w):
    """
    Evaluate discrete factor belief on the ndgrid tensor formed by the Cartesian product of node states.
    :param factor:
    :return:
    """
    comp_probs = [rv.belief_params_['probs'] for rv in factor.nb]  # [K x V1, K x V2, ..., K x Vn]
    # multiple all dimensions together, all K components at once
    einsum_eq = utils.outer_prod_einsum_equation(len(comp_probs), mats=True)
    joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # K x V1 x V2 x ... Vn
    w_broadcast = tf.reshape(w, [-1] + [1] * len(comp_probs))  # K x 1 x 1 ... x 1
    belief = tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0)  # V1 x V2 x ... Vn
    return belief


def dfactor_belief_bfe(factor, w):
    """
    Get the contribution to the BFE from a discrete factor
    :param factor:
    :param w:
    :return: bfe, aux_obj
    """
    belief = eval_dfactor_belief(factor, w)  # V1 x V2 x ... Vn
    axes = [rv.values for rv in factor.nb]
    axes = utils.expand_dims_for_grid(axes)
    lpot = factor.log_potential.get(axes)  # should evaluate on the Cartesian product (ndgrid) of axes by broadcasting
    log_belief = tf.log(belief)
    F = - lpot + log_belief
    prod = tf.stop_gradient(belief * F)  # stop_gradient is needed for aux_obj
    bfe = tf.reduce_sum(prod)  # we really mean the free energy, which is to be minimized
    aux_obj = tf.reduce_sum(prod * log_belief)  # only differentiate w.r.t log_belief

    return bfe, aux_obj


def drv_belief_bfe(rv, w):
    """
    Get the contribution to the BFE from a discrete rv.
    :param rv:
    :param w:
    :return:
    """
    w_K1 = tf.reshape(w, [-1, 1])
    belief = tf.reduce_sum(w_K1 * rv.belief_params_['probs'], axis=0)  # K x 1 times K x S, then sum over axis 0
    log_belief = tf.log(belief)
    F = (1 - len(rv.nb)) * log_belief
    prod = tf.stop_gradient(belief * F)  # stop_gradient is needed for aux_obj
    bfe = tf.reduce_sum(prod)  # we really mean the free energy, which is to be minimized
    aux_obj = tf.reduce_sum(prod * log_belief)  # only differentiate w.r.t log_belief

    return bfe, aux_obj
