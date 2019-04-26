# from utils import outer_prod_einsum_equation
from scipy.special import roots_hermite
import tensorflow as tf
import numpy as np
import utils


def get_hfactor_expectation_coefs_points(factor, K, T, dtype='float64'):
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
    assert factor.domain_type in ('c', 'h'), \
        'Must input continuous/hybrid factor; use dfactor_bfe_obj directly for discrete factor for better performance'
    # compute GHQ once (same for all cnodes) if factor is cont/hybrid
    ghq_points, ghq_weights = roots_hermite(T)  # assuming Gaussian for now
    ghq_coef = (np.pi) ** (-0.5)  # from change-of-var
    ghq_weights = ghq_coef * ghq_weights  # let's fold ghq_coef into the quadrature weights, so no need to worry about it later
    # ghq_weights_KT = np.tile(np.reshape(ghq_weights, [1, -1]), [K, 1])  # K x T (repeat for K identical rows)
    ghq_weights = tf.constant(ghq_weights, dtype=dtype)
    ghq_weights_KT = tf.tile(tf.reshape(ghq_weights, [1, -1]), [K, 1])  # K x T (repeat for K identical rows)

    for rv in factor.nb:
        if rv.domain_type == 'd':  # discrete
            c = rv.belief_params_['pi']  # K x dstates[i] matrix (tf); will be put under stop_gradient later
            p = np.tile(np.reshape(rv.values, [1, -1]), [K, 1])  # K x dstates[i] (identical rows); currently not used
        elif rv.domain_type == 'c':  # cont, assuming Gaussian for now
            c = ghq_weights_KT
            mean_K1 = rv.belief_params_['mu_K1']
            var_K1 = rv.belief_params_['var_K1']
            p = (2 * var_K1) ** 0.5 * ghq_points + mean_K1  # K x T
            p = tf.stop_gradient(p)  # don't want to differentiate w.r.t. evaluation points
        else:
            raise NotImplementedError
        coefs.append(c)
        axes.append(p)

    return coefs, axes


def eval_hfactor_belief(factor, axes, w):
    """
    Evaluate hybrid/continuous factor's belief on grid(s).
    :param factor:
    :param axes: list of mats [M x V1, M x V2, ..., M x Vn]; we allow the flexibility to evaluate on M > 1 ndgrids
    simultaneously
    :param w:
    :return: a [M x V1 x V2 x ... x Vn] tensor, whose (m, v1, ..., vn)th coordinate is the mixture belief evaluated on
    the point (axes[0][m,v1], axes[1][m,v2], ..., axes[n][m,vn])
    """
    M = int(axes[0].shape[0])  # number of grids
    comp_probs = []
    for n, rv in enumerate(factor.nb):
        if rv.domain_type == 'd':  # discrete
            comp_prob = rv.belief_params_['pi']  # assuming the states of Xn are sorted, so p_kn(rv.states) = p_kn
            comp_prob = comp_prob[:, None, :]  # K x 1 x Vn
            comp_prob = tf.tile(comp_prob, [1, M, 1])  # K x M x Vn; same for all M axes
        elif rv.domain_type == 'c':  # cont, assuming Gaussian for now
            mean_K1 = rv.belief_params_['mu_K1']
            mean_K11 = mean_K1[:, :, None]
            var_inv_K1 = rv.belief_params_['var_inv_K1']
            var_inv_K11 = var_inv_K1[:, :, None]
            # eval pdf of axes[n] (M x Vn) under all K scalar comps of nth node in the clique; result is K x M x Vn
            comp_prob = (2 * np.pi) ** (-0.5) * tf.sqrt(var_inv_K11) * \
                        tf.exp(-0.5 * (axes[n] - mean_K11) ** 2 * var_inv_K11)
        else:
            raise NotImplementedError
        comp_probs.append(comp_prob)

    # multiply all dimensions together, then weigh by w
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), common_first_ndims=2)
    joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # K x M x V1 x V2 x ... Vn
    w_broadcast = tf.reshape(w, [-1] + [1] * (len(factor.nb) + 1))
    return tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0)  # M x V1 x V2 x ... Vn


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
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), common_first_ndims=1)
    w_broadcast = tf.reshape(w, [-1] + [1] * len(factor.nb))  # K x 1 x 1 ... x 1

    coefs, axes = get_hfactor_expectation_coefs_points(factor, K, T)  # [[K, V1], [K, V2], ..., [K, Vn]]
    coefs = tf.einsum(einsum_eq, *coefs)  # K x V1 x V2 x ... Vn; K grids of Hadamard products
    belief = eval_hfactor_belief(factor, axes, w)  # K x V1 x V2 x ... Vn
    lpot = utils.eval_fun_grid(factor.log_potential_fun, arrs=axes)  # K x V1 x V2 x ... Vn
    log_belief = tf.log(belief)
    F = -lpot + log_belief
    prod = tf.stop_gradient(w_broadcast * coefs * F)  # weighted component-wise Hadamard products for K expectations
    bfe = tf.reduce_sum(prod)
    aux_obj = tf.reduce_sum(prod * log_belief)

    return bfe, aux_obj


def eval_dfactor_belief(factor, w):
    """
    Evaluate discrete factor belief on the ndgrid tensor formed by the Cartesian product of node states.
    :param factor:
    :return:
    """
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), common_first_ndims=1)
    comp_probs = [rv.belief_params_['pi'] for rv in factor.nb]  # [K x V1, K x V2, ..., K x Vn]
    # multiply all dimensions together, all K components at once
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
    belief = tf.reduce_sum(w_K1 * rv.belief_params_['pi'], axis=0)  # K x 1 times K x S, then sum over axis 0
    log_belief = tf.log(belief)
    prod = tf.stop_gradient(belief * log_belief)  # stop_gradient is needed for aux_obj
    bfe = (1 - len(rv.nb)) * tf.reduce_sum(prod)  # we really mean the free energy, which is to be minimized
    aux_obj = (1 - len(rv.nb)) * tf.reduce_sum(prod * log_belief)  # only differentiate w.r.t log_belief

    return bfe, aux_obj


def drvs_bfe_obj(rvs, w, Pi, rvs_counts=None):
    """
    Get the contribution to the BFE from multiple discrete rvs sharing the same number of states
    :param rvs: list of discrete rvs; must "line up with" Pi; i.e., Pi[i] gives belief params for rvs[i]
    :param w:
    :param Pi: tensor of component belief probabilities, for N drvs sharing the same number of states; N x K x dstates
    :param rvs_counts: an iterable of non-negative ints, such that the bfe contribution from rvs[i] will be multiplied
    by rvs_counts[i]; by default the contribution from each rv is only counted once
    :return:
    """
    w_1K1 = tf.reshape(w, [1, -1, 1])
    belief = tf.reduce_sum(w_1K1 * Pi, axis=1)  # Nd x common_dstates
    log_belief = tf.log(belief)
    prod = tf.stop_gradient(belief * log_belief)  # Nd x common_dstates
    num_nbrs = np.array([len(rv.nb) for rv in rvs])
    if rvs_counts is None:
        rvs_counts = np.ones(len(rvs), dtype='int')
    else:
        rvs_counts = np.array(rvs_counts).astype('int')
    expect_coefs = rvs_counts * (1 - num_nbrs)
    bfe = tf.reduce_sum(expect_coefs * tf.reduce_sum(prod, axis=1))
    aux_obj = tf.reduce_sum(expect_coefs * tf.reduce_sum(prod * log_belief, axis=1))

    return bfe, aux_obj


def crvs_bfe_obj(rvs, T, w, Mu, Var, rvs_counts=None):
    """
    Get the contribution to the BFE from multiple cont (Gaussian) rvs
    :param rvs: rvs: list of cont rvs; must "line up with" params Mu and Var; i.e., Mu[i] and Var[i] give belief params
    for rvs[i]
    :param T:
    :param w:
    :param Mu:
    :param Var:
    :param rvs_counts: an iterable of non-negative ints, such that the bfe contribution from rvs[i] will be multiplied
    by rvs_counts[i]; by default the contribution from each rv is only counted once
    :return:
    """
    [N, K] = Mu.shape
    w_1K1 = w[None, :, None]

    ghq_points, ghq_weights = roots_hermite(T)  # assuming Gaussian for now
    ghq_coef = (np.pi) ** (-0.5)  # from change-of-var
    ghq_weights *= ghq_coef
    QY = ghq_points * (2 * tf.reshape(Var, [N, K, 1])) ** 0.5 + tf.reshape(Mu, [N, K, 1])  # N x K x T; all eval points
    QY = tf.stop_gradient(QY)  # don't want to differentiate w.r.t. quad points

    log_belief = tf.log(eval_crvs_belief(QY, w, Mu, Var))  # N x K x T
    prod = tf.stop_gradient(w_1K1 * ghq_weights * log_belief)  # N x K x T; weighted component-wise Hadamard products

    num_nbrs = np.array([len(rv.nb) for rv in rvs])
    if rvs_counts is None:
        rvs_counts = np.ones(len(rvs), dtype='int')
    else:
        rvs_counts = np.array(rvs_counts).astype('int')
    expect_coefs = rvs_counts * (1 - num_nbrs)

    bfe = tf.reduce_sum(expect_coefs * tf.reduce_sum(prod, axis=[1, 2]))
    aux_obj = tf.reduce_sum(expect_coefs * tf.reduce_sum(prod * log_belief, axis=[1, 2]))

    return bfe, aux_obj


def eval_crvs_belief(X, w, Mu, Var):
    """
    Evaluate beliefs on tensor data, for multiple continuous (Gaussian) rvs simultaneously.
    :param X: shape N x ..., each nth slice evaluated by the nth cnode, with params Mu[n], Var[n]
    :param w: shape K
    :param Mu: shape N x K, mixture means of N cnodes
    :param Var: shape N x K, mixture variances of N cnodes
    :return:
    """
    assert len(Mu.shape) > 1, '1D case to be added later'
    comp_log_probs = eval_crvs_comp_log_prob(X, Mu, Var, backend=tf)
    w_broadcast = tf.reshape(w, [-1] + [1] * len(X.shape))  # K x 1 x 1...
    out = tf.reduce_sum(w_broadcast * tf.exp(comp_log_probs), axis=0)
    return out


def eval_crvs_comp_log_prob(X, Mu, Var, backend=np):
    """
    Evaluate component-wise log probabilities on tensor data, for multiple continuous (Gaussian) rvs
    simultaneously.
    :param X: shape N x ..., each nth slice evaluated by the nth cnode, with params Mu[n], Var[n]
    :param Mu: shape N x K, mixture means of N cnodes
    :param Var: shape N x K, mixture variances of N cnodes
    :param backend: tf/np, for symbolic/eager computation
    :return: a tensor whose first dimension is K, the rest dimensions have the same shape as x, so that the kth slice
    gives the log pdf of x under the kth component
    """
    assert len(Mu.shape) > 1, '1D case to be added later'
    b = backend
    Mu, Var = b.transpose(Mu), b.transpose(Var)  # K x N
    ind = (...,) + (None,) * (len(X.shape) - 1)
    Mu, Var = Mu[ind], Var[ind]  # K x N x 1 x 1 ... 1, to line up with x for broadcasting
    Var_inv = 1 / Var
    comp_log_probs = -0.5 * np.log(2 * np.pi) + 0.5 * b.log(Var_inv) - 0.5 * (X - Mu) ** 2 * Var_inv  # K by x.shape
    return comp_log_probs


def eval_drvs_comp_prob(X, Pi):
    """
    Evaluate component-wise probabilities on matrix X, for multiple discrete rvs simultaneously.
    Currently only supports numpy arrays.
    :param X: shape N x M integer type, each nth slice evaluated by the nth dnode, with params Pi[n]
    :param Pi: probability params, Pi[n] should be K x dstates_of_nth_dnode
    :return: K x N x M, kth slice gives p_k(x)
    """
    N, M = X.shape
    K, _ = Pi[0].shape
    out = np.empty([K, N, M], dtype=Pi[0].dtype)  # maybe work out better shapes
    for n in range(N):
        out[:, n, :] = Pi[n][:, X[n]]  # K x M

    return out


def calc_marg_comp_log_prob(g, X, obs_rvs, params):
    """

    :param g:
    :param X: M x N_o matrix of (partial) observations, where N_o is the number of obs nodes; alternatively a N_o vector
    :param obs_rvs: length N_o list of observed rvs
    :param params:
    :return:
    """
    single_example = False
    if len(X.shape) == 1:
        single_example = True
        X = X[None, :]  # [1, N_o]

    obs_c = np.array([rv in g.Vc for rv in obs_rvs])  # indicator vec of length N_o
    obs_d = np.array([rv in g.Vd for rv in obs_rvs])  # indicator vec of length N_o

    obs_crvs_idxs = [g.Vc_idx[rv] for (i, rv) in enumerate(obs_rvs) if obs_c[i]]
    obs_drvs_idxs = [g.Vd_idx[rv] for (i, rv) in enumerate(obs_rvs) if obs_d[i]]

    C = X[:, obs_c]  # M x (num of cont obs)
    D = X[:, obs_d]  # M x (num of disc obs)
    D_precast = D.copy()
    D = D.astype('int')  # for disc obs (will need to index into Pi)
    assert np.all(D == D_precast), 'Discrete observations must be integers!'

    comp_log_probs = 0

    if len(obs_crvs_idxs) > 0:
        Mu = params['Mu'][obs_crvs_idxs]  # N_oc x K
        Var = params['Var'][obs_crvs_idxs]
        all_comp_log_probs = eval_crvs_comp_log_prob(np.transpose(C), Mu=Mu, Var=Var, backend=np)  # K x N_oc x M
        comp_log_probs += np.sum(all_comp_log_probs, axis=1)

    if len(obs_drvs_idxs) > 0:
        Pi = params['Pi'][obs_drvs_idxs]  # [K x V1, K x V2, ... ], of length N_od
        all_comp_log_probs = np.log(eval_drvs_comp_prob(np.transpose(D), Pi=Pi))  # K x N_od x M
        comp_log_probs += np.sum(all_comp_log_probs, axis=1)

    comp_log_probs = comp_log_probs.transpose()  # M x K

    if single_example:
        comp_log_probs = comp_log_probs[0]  # K

    return comp_log_probs


def calc_marg_log_prob(g, X, obs_rvs, params):
    """
    Calculate marginal log probabilities of observations
    :param g:
    :param X: M x N_o matrix of (partial) observations, where N_o is the number of obs nodes; alternatively a N_o vector
    :param obs_rvs: obs_rvs: length N_o list of observed rvs
    :param params:
    :return:
    """
    comp_log_probs = calc_marg_comp_log_prob(g, X, obs_rvs, params)  # M x K
    w = params['w']
    out = utils.logsumexp(np.log(w) + comp_log_probs, axis=-1)  # reduce along the last dimension
    return out


def calc_cond_mixture_weights(g, X, obs_rvs, params):
    """
    Calculate mixture weights of the new mixture produced by conditioning on observations;
    i.e., given current weights w, calc w', such that p(x_h|x_o) = \sum_k w'[k] p_k(x_h) =
    = \sum_k {w[k]*p_k(x_o) / p(x_o)} p_k(x_h) = \sum_k {w[k]*p_k(x_o) / (\sum_j w[j]*p_j(x_o))} p_k(x_h)
    :param g:
    :param X:
    :param obs_rvs:
    :param params:
    :return:
    """
    comp_log_probs = calc_marg_comp_log_prob(g, X, obs_rvs, params)  # M x K
    w = params['w']
    cond_mixture_weights = utils.softmax(np.log(w) + comp_log_probs, axis=-1)  # M x K
    return cond_mixture_weights


def drv_belief_map(w, pi):
    """
    Calculate MAP configuration of a discrete node belief.
    :param w: w is allowed to be a M x K matrix, for simultaneous calculation for M different mixture weights
    :param pi: K x dstates matrix of params of categorical mixture
    :return:
    """
    state_probs = w @ pi  # M x dstates
    if state_probs.ndim == 1:  # output will be scalars
        map_states = np.argmax(state_probs)
        map_probs = state_probs[map_states]
    else:
        map_states = np.argmax(state_probs, axis=1)
        map_probs = state_probs[:, map_states]

    return map_states, map_probs


def crv_belief_map(w, mu, var, bds):
    """
    Find MAP estimate of a scalar continuous belief.
    :param w:
    :param mu:
    :param var:
    :param bds:
    :return:
    """
    # TODO: implement multivariate case and call it to handle special scalar case
    from scipy.optimize import minimize
    log_w = np.log(w)
    var_inv = 1 / var

    def neg_gmm_log_prob(x):
        comp_log_probs = -0.5 * np.log(2 * np.pi) + 0.5 * np.log(var_inv) - 0.5 * (x - mu) ** 2 * var_inv
        return -utils.logsumexp(log_w + comp_log_probs)

    res = []
    for m in mu:  # starting optimization from the K component modes and take the best solution
        x0 = m
        r = minimize(neg_gmm_log_prob, x0=x0, bounds=[bds])  # bounds kwarg needs to be a list of (lb, ub)
        res.append(r)
    best_res = min(res, key=lambda r: r.fun)
    x_opt = float(best_res.x)

    return x_opt
