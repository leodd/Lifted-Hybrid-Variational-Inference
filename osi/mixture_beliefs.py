# from utils import outer_prod_einsum_equation
from scipy.special import roots_hermite
import tensorflow as tf
import numpy as np
import utils


def get_hfactor_expectation_coefs_points(factor, K, T=None, dtype='float64'):
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
    if factor.domain_type in ('c', 'h'):  # compute GHQ once (same for all cnodes) if factor is cont/hybrid
        ghq_points, ghq_weights = roots_hermite(T)  # assuming Gaussian for now
        ghq_coef = (np.pi) ** (-0.5)  # from change-of-var
        ghq_weights = ghq_coef * ghq_weights  # let's fold ghq_coef into the quadrature weights, so no need to worry about it later
        # ghq_weights_KT = np.tile(np.reshape(ghq_weights, [1, -1]), [K, 1])  # K x T (repeat for K identical rows)
        ghq_weights = tf.constant(ghq_weights, dtype=dtype)
        ghq_weights_KT = tf.tile(tf.reshape(ghq_weights, [1, -1]), [K, 1])  # K x T (repeat for K identical rows)

    for rv in factor.nb:
        if rv.domain_type == 'd':  # discrete
            c = rv.belief_params_['probs']  # K x dstates[i] matrix (tf)
            p = np.tile(np.reshape(rv.values, [1, -1]), [K, 1])  # K x dstates[i] (identical rows)
        elif rv.domain_type == 'c':  # cont, assuming Gaussian for now
            c = ghq_weights_KT
            mean_K1 = rv.belief_params_['mean_K1']
            var_K1 = rv.belief_params_['var_K1']
            p = (2 * var_K1) ** 0.5 * ghq_points + mean_K1  # K x T
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
    w_broadcast = tf.reshape(w, [-1] + [1] * len(factor.nb))  # K x 1 x 1 ... x 1
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
                            tf.exp(-0.5 * (axes[n][k] - mean_K1) ** 2 * var_inv_K1)  # eval pdf under all K scalar comps
            else:
                raise NotImplementedError
            comp_probs.append(comp_prob)

        # multiple all dimensions together, then weighted sum by w; this gives
        joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # K x V1 x V2 x ... Vn
        res.append(tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0, keepdims=True))  # 1 x V1 x V2 x ... Vn

    return tf.concat(res, 0)  # K x V1 x V2 x ... Vn


def hfactor_bfe_obj(factor, T, w):
    """
    Get the contribution to the BFE from a hybrid or continuous factor (use dfactor_bfe_obj for a discrete factor for
    efficiency).
    :param factor:
    :param T: num quad points
    :param w:
    :return:
    """
    K = np.prod(w.shape)
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), mats=True)

    coefs, axes = get_hfactor_expectation_coefs_points(factor, K, T)  # [[K, V1], [K, V2], ..., [K, Vn]]
    coefs = tf.einsum(einsum_eq, *coefs)  # K x V1 x V2 x ... Vn; K grids of Hadamard products
    belief = eval_hfactor_belief(factor, axes, w)  # K x V1 x V2 x ... Vn
    lpot = utils.eval_fun_grid(factor.log_potential_fun, arrs=axes)  # K x V1 x V2 x ... Vn
    log_belief = tf.log(belief)
    F = -lpot + log_belief
    prod = tf.stop_gradient(coefs * F)  # stop_gradient is needed for aux_obj
    bfe = tf.reduce_sum(
        w * tf.reduce_sum(prod, axis=tuple(range(1, len(factor.nb) + 1))))  # inner reduce over 1,2,...,n
    aux_obj = tf.reduce_sum(w * tf.reduce_sum(prod * log_belief, axis=tuple(range(1, len(factor.nb) + 1))))

    return bfe, aux_obj


def eval_dfactor_belief(factor, w):
    """
    Evaluate discrete factor belief on the ndgrid tensor formed by the Cartesian product of node states.
    :param factor:
    :return:
    """
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), mats=True)
    comp_probs = [rv.belief_params_['probs'] for rv in factor.nb]  # [K x V1, K x V2, ..., K x Vn]
    # multiple all dimensions together, all K components at once
    joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # K x V1 x V2 x ... Vn
    w_broadcast = tf.reshape(w, [-1] + [1] * len(comp_probs))  # K x 1 x 1 ... x 1
    belief = tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0)  # V1 x V2 x ... Vn
    return belief


def dfactor_bfe_obj(factor, w):
    """
    Get the contribution to the BFE from a discrete factor
    :param factor:
    :param w:
    :return: bfe, aux_obj
    """
    belief = eval_dfactor_belief(factor, w)  # V1 x V2 x ... Vn
    axes = [rv.values for rv in factor.nb]
    lpot = utils.eval_fun_grid(factor.log_potential_fun, axes)
    log_belief = tf.log(belief)
    F = - lpot + log_belief
    prod = tf.stop_gradient(belief * F)  # stop_gradient is needed for aux_obj
    bfe = tf.reduce_sum(prod)  # we really mean the free energy, which is to be minimized
    aux_obj = tf.reduce_sum(prod * log_belief)  # only differentiate w.r.t log_belief

    return bfe, aux_obj


def drv_bfe_obj(rv, w):
    """
    Get the contribution to the BFE from a discrete rv.
    :param rv:
    :param w:
    :return:
    """
    w_K1 = tf.reshape(w, [-1, 1])
    belief = tf.reduce_sum(w_K1 * rv.belief_params_['probs'], axis=0)  # K x 1 times K x S, then sum over axis 0
    log_belief = tf.log(belief)
    prod = tf.stop_gradient(belief * log_belief)  # stop_gradient is needed for aux_obj
    bfe = (1 - len(rv.nb)) * tf.reduce_sum(prod)  # we really mean the free energy, which is to be minimized
    aux_obj = (1 - len(rv.nb)) * tf.reduce_sum(prod * log_belief)  # only differentiate w.r.t log_belief

    return bfe, aux_obj


def drvs_bfe_obj(rvs, w, Pi):
    """
    :param rvs: list of discrete rvs; must "line up with" Pi; i.e., Pi[i] gives belief params for rvs[i]
    :param w:
    :param Pi: tensor of component belief probabilities, for N drvs sharing the same number of states; N x K x dstates
    :return:
    """
    w_1K1 = tf.reshape(w, [1, -1, 1])
    belief = tf.reduce_sum(w_1K1 * Pi, axis=1)  # Nd x shared_dstates
    log_belief = tf.log(belief)
    num_nbrs = np.array([len(rv.nb) for rv in rvs])
    prod = tf.stop_gradient(belief * log_belief)  # Nd x shared_dstates
    bfe = tf.reduce_sum((1 - num_nbrs) * tf.reduce_sum(prod, axis=1))
    aux_obj = tf.reduce_sum((1 - num_nbrs) * tf.reduce_sum(prod * log_belief, axis=1))

    return bfe, aux_obj


def eval_crvs_belief(x, w, Mu, Var):
    """

    :param x: shape N x ..., each nth slice evaluated by a different cnode
    :param w: shape K
    :param Mu: shape N x K, mixture means of N cnodes
    :param Var: shape N x K, mixture variances of N cnodes
    :return:
    """
    assert len(Mu.shape) > 1, '1D case to be added later'
    [N, K] = Mu.shape
    xmat = tf.reshape(tf.transpose(tf.reshape(x, [N, -1])), [-1, N, 1])  # d x N x 1
    v_inv = 1 / Var
    comp_probs = (2 * np.pi) ** (-0.5) * tf.sqrt(v_inv) * tf.exp(-0.5 * (xmat - Mu) ** 2 * v_inv)  # d x N x K
    out = tf.reshape(tf.transpose(tf.reduce_sum(w * comp_probs, 2)), x.shape)
    return out


def crvs_bfe_obj(rvs, T, w, Mu, Var):
    """

    :param rvs: rvs: list of cont rvs; must "line up with" params Mu and Var; i.e., Mu[i] and Var[i] give belief params
    for rvs[i]
    :param Mu:
    :param Var:
    :param w:
    :return:
    """
    [N, K] = Mu.shape
    w_col = tf.reshape(w, [K, 1])

    ghq_points, ghq_weights = roots_hermite(T)  # assuming Gaussian for now
    ghq_coef = (np.pi) ** (-0.5)  # from change-of-var
    ghq_weights *= ghq_coef
    QY = ghq_points * (2 * tf.reshape(Var, [N, K, 1])) ** 0.5 + tf.reshape(Mu, [N, K, 1])  # N x K x T; all eval points

    num_nbrs = np.array([len(rv.nb) for rv in rvs])
    num_nbrs = num_nbrs.reshape([1, N])
    log_belief = tf.log(eval_crvs_belief(QY, w, Mu, Var))  # N x K x T
    prod = tf.stop_gradient(ghq_weights * log_belief)  # N x K x T; component-wise Hadamard products

    grals = tf.reduce_sum(prod, axis=2)  # N x K
    bfe = (1 - num_nbrs) @ (grals @ w_col)
    bfe = tf.reshape(bfe, ())  # convert 1x1 mat to scalar

    grals = tf.reduce_sum(prod * log_belief, axis=2)  # N x K
    aux_obj = (1 - num_nbrs) @ (grals @ w_col)
    aux_obj = tf.reshape(aux_obj, ())  # convert 1x1 mat to scalar

    return bfe, aux_obj
