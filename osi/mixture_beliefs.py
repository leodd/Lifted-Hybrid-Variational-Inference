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
        if rv.domain_type[0] == 'd':  # discrete
            c = rv.belief_params_['pi']  # K x dstates[i] matrix (tf); will be put under stop_gradient later
            a = np.tile(np.reshape(rv.values, [1, -1]), [K, 1])  # K x dstates[i] (last dimension repeated)
            a = tf.constant(a, dtype=dtype)  # otherwise tf complains about multiplying int tensor with float tensor
        elif rv.domain_type[0] == 'c':  # cont, assuming Gaussian for now
            c = ghq_weights_KT
            mean_K1 = rv.belief_params_['mu_K1']
            var_K1 = rv.belief_params_['var_K1']
            a = (2 * var_K1) ** 0.5 * ghq_points + mean_K1  # K x T
            a = tf.stop_gradient(a)  # don't want to differentiate w.r.t. evaluation points
        else:
            raise NotImplementedError
        coefs.append(c)
        axes.append(a)

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
    for i, rv in enumerate(factor.nb):
        if rv.domain_type[0] == 'd':  # discrete
            comp_prob = rv.belief_params_['pi']  # assuming the states of Xi are sorted, so p_ki(rv.states) = p_ki
            comp_prob = comp_prob[:, None, :]  # K x 1 x Vi
            comp_prob = tf.tile(comp_prob, [1, M, 1])  # K x M x Vi; same for all M axes
        elif rv.domain_type[0] == 'c':  # cont, assuming Gaussian for now
            mean_K1 = rv.belief_params_['mu_K1']
            mean_K11 = mean_K1[:, :, None]
            var_inv_K1 = rv.belief_params_['var_inv_K1']
            var_inv_K11 = var_inv_K1[:, :, None]
            # eval pdf of axes[i] (M x Vi) under all K scalar comps of ith node in the clique; result is K x M x Vi
            comp_prob = (2 * np.pi) ** (-0.5) * tf.sqrt(var_inv_K11) * \
                        tf.exp(-0.5 * (axes[i] - mean_K11) ** 2 * var_inv_K11)
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


def eval_hfactors_belief(factors, axes, w):
    """
    Evaluate multiple hybrid/continuous factors' belief on grid(s), assuming the factors have the same nb_domain_type
    (i.e., the same kinds of variables in their cliques) so the dimensions match.
    :param factors:
    :param axes: list of mats [C x M x V1, C x M x V2, ..., C x M x Vn]; we allow the flexibility to evaluate on M > 1
    ndgrids for each factor simultaneously
    :param w:
    :return: a [C x M x V1 x V2 x ... x Vn] tensor, whose (c, m, v1, ..., vn)th coordinate is the mixture belief of the
    cth factor (i.e., factors[c]) evaluated on the point (axes[0][c,m,v1], axes[1][c,m,v2], ..., axes[n][c,m,vn])
    """
    K = np.prod(w.shape)
    C = len(factors)
    M = int(axes[0].shape[1])  # number of grids for each factor
    comp_probs = []
    factor = factors[0]
    n = len(factor.nb)
    for i, domain_type in enumerate(factor.nb_domain_types):
        factors_ith_nb = [factor.nb[i] for factor in factors]  # the ith neighbor (rv in clique) across all factors
        if domain_type[0] == 'd':  # discrete
            comp_prob = tf.stack([rv.belief_params_['pi'] for rv in factors_ith_nb],
                                 axis=1)  # K x C x Vi, where Vi is the number of dstates of factor.nb[i]
            comp_prob = comp_prob[:, :, None, :]  # K x C x 1 x Vi
            comp_prob = tf.tile(comp_prob, [1, 1, M, 1])  # K x C x M x Vi; same for all M axes
        elif domain_type[0] == 'c':  # cont, assuming Gaussian for now
            # Mu = tf.stack([rv.belief_params_['mu'] for rv in factors_ith_nb], axis=0)  # C x K
            # Var_inv = tf.stack([rv.belief_params_['var_inv'] for rv in factors_ith_nb], axis=0)  # C x K
            # Mu_KC11 = tf.reshape(tf.transpose(Mu), [K, C, 1, 1])
            # Var_inv_KC11 = tf.reshape(tf.transpose(Var_inv), [K, C, 1, 1])
            Mu_KC11 = tf.stack([rv.belief_params_['mu_K1'] for rv in factors_ith_nb], axis=1)[:, :, None]
            Var_inv_KC11 = tf.stack([rv.belief_params_['var_inv_K1'] for rv in factors_ith_nb], axis=1)[:, :, None]
            # eval pdf of axes[i] under all K scalar comps of ith nodes in all the cliques; result is K x C x M x Vi
            comp_prob = (2 * np.pi) ** (-0.5) * tf.sqrt(Var_inv_KC11) * \
                        tf.exp(-0.5 * (axes[i] - Mu_KC11) ** 2 * Var_inv_KC11)
        else:
            raise NotImplementedError
        comp_probs.append(comp_prob)

    # multiply all dimensions together, then weigh by w
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), common_first_ndims=3)
    joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # K x C x M x V1 x V2 x ... Vn
    w_broadcast = tf.reshape(w, [K] + [1] * (len(factor.nb) + 2))
    return tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0)  # C x M x V1 x V2 x ... Vn


def hfactors_bfe_obj(factors, T, w, dtype='float64'):
    """
    Get the contribution to the BFE from multiple hybrid (or continuous) factors that have the same types of neighboring
    rvs.
    :param factors: length C list of factor objects that have the same nb_domain_type.
    :param T:
    :param w:
    :return:
    """
    # group factors with the same types of log potentials together for efficient evaluation later
    factors_with_unique_log_potential_fun_types, unique_log_potential_fun_types = \
        utils.get_unique_subsets(factors, key=lambda f: type(f.log_potential_fun))
    factors = sum(factors_with_unique_log_potential_fun_types, [])  # join together into flat list

    K = np.prod(w.shape)
    C = len(factors)
    factor = factors[0]
    n = len(factor.nb)

    ghq_points, ghq_weights = roots_hermite(T)  # assuming Gaussian for now
    ghq_coef = (np.pi) ** (-0.5)  # from change-of-var
    ghq_weights = ghq_coef * ghq_weights  # let's fold ghq_coef into the quadrature weights, so no need to worry about it later
    ghq_weights = tf.constant(ghq_weights, dtype=dtype)
    ghq_weights_CKT = tf.tile(tf.reshape(ghq_weights, [1, 1, T]), [C, K, 1])  # C x K x T

    coefs = [None] * n  # will be [[C, K, V1], [C, K, V2], ..., [C, K, Vn]]
    axes = [None] * n  # will be [[C, K, V1], [C, K, V2], ..., [C, K, Vn]]

    comp_probs = []  # for evaluating beliefs along the way
    for i, domain_type in enumerate(factor.nb_domain_types):
        factors_ith_nb = [factor.nb[i] for factor in factors]  # the ith neighbor (rv in clique) across all factors
        if domain_type[0] == 'd':
            rv = factor.nb[i]
            c = tf.stack([rv.belief_params_['pi'] for rv in factors_ith_nb],
                         axis=0)  # C x K x Vi, where Vi is the number of dstates of factor.nb[i]

            coefs[i] = c  # the prob params are exactly the inner-prod coefficients in expectations
            a = np.tile(np.reshape(rv.values, [1, 1, -1]),
                        [C, K, 1])  # C x K x dstates[i] (last dimension repeated)
            a = tf.constant(a, dtype=dtype)  # otherwise tf complains about multiplying int tensor with float tensor
            axes[i] = a

            # eval_hfactors_belief
            # comp_prob = tf.stack([rv.belief_params_['pi'] for rv in factors_ith_nb],
            #                      axis=1)  # K x C x Vi, where Vi is the number of dstates of factor.nb[i]
            comp_prob = tf.transpose(c, [1, 0, 2])  # K x C x Vi
            comp_prob = comp_prob[:, :, None, :]  # K x C x 1 x Vi
            comp_prob = tf.tile(comp_prob, [1, 1, K, 1])  # K x C x M(=K) x Vi; same for all M(=K) axes
        elif domain_type[0] == 'c':
            Mu_CK = tf.stack([rv.belief_params_['mu'] for rv in factors_ith_nb], axis=0)  # C x K
            Var_CK = tf.stack([rv.belief_params_['var'] for rv in factors_ith_nb], axis=0)  # C x K
            coefs[i] = ghq_weights_CKT
            a = (2 * Var_CK[:, :, None]) ** 0.5 * ghq_points + Mu_CK[:, :, None]  # C x K x T
            a = tf.stop_gradient(a)  # don't want to differentiate w.r.t. evaluation points
            axes[i] = a

            # eval_hfactors_belief
            Mu_KC11 = tf.transpose(Mu_CK)[:, :, None, None]  # K x C x 1 x 1
            Var_inv_KC11 = tf.stack([rv.belief_params_['var_inv_K1'] for rv in factors_ith_nb], axis=1)[:, :, None]
            # eval pdf of axes[i] under all K scalar comps of ith nodes in all the cliques; result is K x C x M(=K) x Vi
            comp_prob = (2 * np.pi) ** (-0.5) * tf.sqrt(Var_inv_KC11) * \
                        tf.exp(-0.5 * (axes[i] - Mu_KC11) ** 2 * Var_inv_KC11)
        else:
            raise NotImplementedError
        comp_probs.append(comp_prob)

    # eval_hfactors_belief
    # multiply all dimensions together, then weigh by w
    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), common_first_ndims=3)
    joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # K x C x M x V1 x V2 x ... Vn
    w_broadcast = tf.reshape(w, [K] + [1] * (len(factor.nb) + 2))
    belief = tf.reduce_sum(w_broadcast * joint_comp_probs, axis=0)  # C x M x V1 x V2 x ... Vn
    # above replaces the call belief = eval_hfactors_belief(factors, axes, w)  # C x K x V1 x V2 x ... Vn

    einsum_eq = utils.outer_prod_einsum_equation(n, common_first_ndims=2)
    coefs = tf.einsum(einsum_eq, *coefs)  # C x K x V1 x V2 x ... Vn; C x K grids of Hadamard products

    lpot = group_eval_log_potential_funs(factors_with_unique_log_potential_fun_types, unique_log_potential_fun_types,
                                         axes)  # C x K x V1 x V2 x ... Vn
    log_belief = tf.log(belief)
    F = -lpot + log_belief
    w_broadcast = tf.reshape(w, [-1] + [1] * n)  # K x 1 x 1 ... x 1
    prod = tf.stop_gradient(w_broadcast * coefs * F)  # weighted component-wise Hadamard products for C x K expectations
    factors_bfes = tf.reduce_sum(prod, axis=list(range(1, n + 2)))  # reduce the last (n+1) dimensions
    factors_aux_objs = tf.reduce_sum(prod * log_belief, axis=list(range(1, n + 2)))  # reduce the last (n+1) dimensions

    sharing_counts = np.array([factor.sharing_count for factor in factors], dtype='float')
    bfe = tf.reduce_sum(sharing_counts * factors_bfes)
    aux_obj = tf.reduce_sum(sharing_counts * factors_aux_objs)

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


def dfactors_bfe_obj(factors, w):
    """
    Get the contribution to the BFE from multiple discrete factors that have the same kinds of neighboring rvs (i.e.,
    the rvs in factor.nb should have the same number of dstates).
    :param factors: length C list of factor objects that have the same factor.nb_domain_types.
    :param w:
    :return:
    """
    # group factors with the same types of log potentials together for efficient evaluation later
    # unique_log_potential_types = set(type(factor.log_potential_fun) for factor in factors)
    # factors_with_unique_log_potential_types = [None] * len(unique_log_potential_types)  # list of lists of factors
    # for i, log_potential_type in unique_log_potential_types:
    #     like_factors = list(filter(lambda f: type(factor.log_potential_fun) == log_potential_type, factors))
    #     factors_with_unique_log_potential_types[i] = like_factors
    factors_with_unique_log_potential_fun_types, unique_log_potential_fun_types = \
        utils.get_unique_subsets(factors, key=lambda f: type(f.log_potential_fun))
    factors = sum(factors_with_unique_log_potential_fun_types, [])  # join together into flat list

    C = len(factors)
    factor = factors[0]
    n = len(factor.nb)

    einsum_eq = utils.outer_prod_einsum_equation(len(factor.nb), common_first_ndims=2)
    comp_probs = [tf.stack([f.nb[i].belief_params_['pi'] for f in factors], axis=0)
                  for i in range(n)]  # [C x K x V1, C x K x V2, ..., C x K x Vn]
    # multiply all dimensions together, all C factors and all K components at once
    joint_comp_probs = tf.einsum(einsum_eq, *comp_probs)  # C x K x V1 x V2 x ... Vn
    w_broadcast = tf.reshape(w, [-1] + [1] * len(comp_probs))  # K x 1 x 1 ... x 1
    belief = tf.reduce_sum(w_broadcast * joint_comp_probs, axis=1)  # C x V1 x V2 x ... Vn
    axes = [np.stack([f.nb[i].values for f in factors], axis=0) for i in range(n)]  # [C x V1, C x V2, ..., C x Vn]

    lpot = group_eval_log_potential_funs(factors_with_unique_log_potential_fun_types, unique_log_potential_fun_types,
                                         axes)  # C x V1 x V2 x ... Vn
    log_belief = tf.log(belief)
    F = - lpot + log_belief
    prod = tf.stop_gradient(belief * F)  # stop_gradient is needed for aux_obj

    factors_bfes = tf.reduce_sum(prod, axis=list(range(1, n + 1)))  # reduce the last n dimensions
    factors_aux_objs = tf.reduce_sum(prod * log_belief, axis=list(range(1, n + 1)))  # reduce the last n dimensions

    sharing_counts = np.array([factor.sharing_count for factor in factors], dtype='float')
    bfe = tf.reduce_sum(sharing_counts * factors_bfes)
    aux_obj = tf.reduce_sum(sharing_counts * factors_aux_objs)

    return bfe, aux_obj


def group_eval_log_potential_funs(factors_with_unique_log_potential_fun_types, unique_log_potential_fun_types, axes):
    """
    Evaluate multiple log potentials on corresponding slices of axes, grouping the evaluation of same types of log
    potentials together for efficiency.
    This extends the call of 'lpot = utils.eval_fun_grid(log_potential_fun, axes)  # C x ??? x V1 x V2 x ... Vn'
    to the case when log_potential_fun is no longer shared across all C slices of axes.
    :param factors_with_unique_log_potential_fun_types:
    :param unique_log_potential_fun_types:
    :param axes: [C x ??? x V1, C x ??? x V2, ... C x ??? x Vn], where ??? is any number of identical dimensions
    :return:
    """
    n = len(axes)  # all the factors have n args (nb) in scope
    from Potential import GaussianLogPotential
    from MLNPotential import MLNLogPotential
    lpots = [None] * len(unique_log_potential_fun_types)
    j = 0
    for i, log_potential_fun_type in enumerate(unique_log_potential_fun_types):
        like_factors = factors_with_unique_log_potential_fun_types[i]  # all have the same type of log_potential_funs
        like_log_potential_funs = [f.log_potential_fun for f in like_factors]
        c = len(like_factors)
        like_axes = [a[j:(j + c)] for a in axes]

        if log_potential_fun_type == GaussianLogPotential:
            like_axes = utils.broadcast_arrs_to_common_shape(utils.expand_dims_for_fun_grid(
                like_axes))  # length n list, having common shape [c x ??? x V1 x V2 x ... Vn]
            v = tf.stack(like_axes)  # n x c x ...
            mu = np.stack([f.mu for f in like_log_potential_funs], axis=-1)  # n x c
            mu = np.reshape(mu, [n, c] + [1] * (len(v.shape) - 2))  # n x c x ones
            sig_inv = np.stack([f.sig_inv for f in like_log_potential_funs], axis=-1)  # n x n x c
            sig_inv = np.reshape(sig_inv, [n, n, c] + [1] * (len(v.shape) - 2))  # n x n x c x ones

            diff = v - mu  # n x c x ... , same shape as v
            outer_prods = diff[None, ...] * diff[:, None, ...]  # n x n x c x ...
            quad_form = tf.reduce_sum(outer_prods * sig_inv, axis=[0, 1])
            lpot = -.5 * quad_form
        elif log_potential_fun_type == MLNLogPotential:
            formulas = [f.formula for f in like_log_potential_funs]
            unique_formulas = list(set(formulas))  # need a better way to check formula equality than object id
            assert len(unique_formulas) == 1, 'currently only support identical formula'
            shared_formula = unique_formulas[0]
            lpot = utils.eval_fun_grid(shared_formula, axes)  # c x ??? x V1 x V2 x ... Vn
            weights = np.array([f.w for f in like_log_potential_funs])
            weights = np.reshape(weights, [c] + [1] * (len(lpot.shape) - 1))
            # or, use weights[(slice(None),) + (None,)*len(lpot.shape-1)]
            lpot = lpot * weights  # c x ??? x V1 x V2 x ... Vn
        else:
            # lpot = [utils.eval_fun_grid(like_log_potential_funs[l], [a[l] for a in like_axes]) for l in range(c)]
            lpot = []
            for l in range(c):
                lpot.append(utils.eval_fun_grid(like_log_potential_funs[l], [a[l] for a in like_axes]))
            lpot = tf.stack(lpot, axis=0)

        lpots[i] = lpot
        j = j + c

    if len(lpots) == 1:
        lpot = lpots[0]
    else:
        lpot = tf.concat(lpots, 0)  # C x V1 x V2 x ... Vn

    return lpot


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


def get_obs_rvs_domain_types_and_params(obs_rvs, all_rvs_params=None, Vc_idx=None, Vd_idx=None):
    """
    Extract the needed obs_rvs_domain_types and obs_rvs_params for _calc_marg_comp_log_prob
    Will try extracting obs_rvs_params from all_rvs_params first; fall back to the .belief_params attributes
    of obs_rvs, if all_rvs_params not provided. Assuming all continuous rvs are Gaussian.
    :param obs_rvs:
    :param all_rvs_params:
    :param g:
    :return:
    """
    obs_rvs_params = {}
    # N_o = len(obs_rvs)
    obs_rvs_domain_types = [rv.domain_type for rv in obs_rvs]
    obs_c = np.array([t[0] == 'c' for t in obs_rvs_domain_types])  # indicator vec of length N_o
    obs_d = np.array([t[0] == 'd' for t in obs_rvs_domain_types])  # indicator vec of length N_o

    if all_rvs_params is not None:
        if np.sum(obs_c) > 0:
            assert Vc_idx is not None
            obs_crvs_idxs = [Vc_idx[rv] for (i, rv) in enumerate(obs_rvs) if obs_c[i]]
            obs_rvs_params['Mu'] = all_rvs_params['Mu'][obs_crvs_idxs]  # N_oc x K
            obs_rvs_params['Var'] = all_rvs_params['Var'][obs_crvs_idxs]
        if np.sum(obs_d) > 0:
            assert Vd_idx is not None
            obs_drvs_idxs = [Vd_idx[rv] for (i, rv) in enumerate(obs_rvs) if obs_d[i]]
            obs_rvs_params['Pi'] = all_rvs_params['Pi'][obs_drvs_idxs]  # [K x V1, K x V2, ... ], of length N_od
    else:  # assuming each rv has .belief_params attribute and we can simply aggregate them
        assert all([rv.belief_params for rv in obs_rvs]), 'obs_rvs must have belief_params'
        if np.sum(obs_c) > 0:
            obs_rvs_params['Mu'] = np.stack([rv.belief_params['mu'] for (i, rv) in enumerate(obs_rvs) if obs_c[i]])
            obs_rvs_params['Var'] = np.stack([rv.belief_params['var'] for (i, rv) in enumerate(obs_rvs) if obs_c[i]])
        if np.sum(obs_d) > 0:
            obs_rvs_params['Pi'] = [rv.belief_params['pi'] for (i, rv) in enumerate(obs_rvs) if obs_d[i]]

    return obs_rvs_domain_types, obs_rvs_params


def _calc_marg_comp_log_prob(X, obs_rvs_domain_types, obs_rvs_params):
    """
    A version of calc_marg_comp_log_prob that directly takes in necessary data (instead of container objects).
    :param X: M x N_o matrix of (partial) observations, where N_o is the number of obs nodes; alternatively a N_o vector
    :param obs_rvs_domain_types: length N_o list of the domain type ('c' or 'd') of observed rvs, same order as obs in X
    :param obs_rvs_params: dict of params of observed rvs for evaluating the marginal probabilities. If obs_rvs contains N_oc
    continuous rvs (assumed Gaussian), then params should contain 'Mu' and 'Var' each of shape N_oc x K, such that
    Mu[i], Var[i] are the params of the ith continuous observed rv; similarly if obs_rvs contains N_od discrete rvs,
    params should contain 'Pi', an iterable of shapes [K x V1, K x V2, ... ], of length N_od (which could be a numpy
    array of shape N_od x K x V in case all the observed drvs take on V states)
    :return: M x K matrix (or a K-vector if X is a N_o-vector)
    """
    single_example = False
    if len(X.shape) == 1:
        single_example = True
        X = X[None, :]  # [1, N_o]

    obs_c = np.array([t[0] == 'c' for t in obs_rvs_domain_types])  # indicator vec of length N_o
    obs_d = np.array([t[0] == 'd' for t in obs_rvs_domain_types])  # indicator vec of length N_o

    C = X[:, obs_c]  # M x (num of cont obs)
    D = X[:, obs_d]  # M x (num of disc obs)
    D_precast = D.copy()
    D = D.astype('int')  # for disc obs (will need to index into Pi)
    assert np.all(D == D_precast), 'Discrete observations must be integers!'

    comp_log_probs = 0

    if np.sum(obs_c) > 0:
        Mu = obs_rvs_params['Mu']  # N_oc x K
        Var = obs_rvs_params['Var']
        all_comp_log_probs = eval_crvs_comp_log_prob(np.transpose(C), Mu=Mu, Var=Var, backend=np)  # K x N_oc x M
        comp_log_probs += np.sum(all_comp_log_probs, axis=1)

    if np.sum(obs_d) > 0:
        Pi = obs_rvs_params['Pi']  # [K x V1, K x V2, ... ], of length N_od
        all_comp_log_probs = np.log(eval_drvs_comp_prob(np.transpose(D), Pi=Pi))  # K x N_od x M
        comp_log_probs += np.sum(all_comp_log_probs, axis=1)

    comp_log_probs = comp_log_probs.transpose()  # M x K

    if single_example:
        comp_log_probs = comp_log_probs[0]  # K

    return comp_log_probs


def calc_marg_comp_log_prob(X, obs_rvs, all_rvs_params=None, g=None):
    """
    Compute component-wise (marginal) log-probabilities of given observed rvs.
    :param X:
    :param obs_rvs:
    :param all_rvs_params:
    :param g:
    :return: M x K matrix (or a K-vector if X is a N_o-vector)
    """
    if all_rvs_params is not None:
        assert g is not None
        Vc_idx, Vd_idx = g.Vc_idx, g.Vd_idx
    else:
        Vc_idx, Vd_idx = None, None
    obs_rvs_domain_types, obs_rvs_params = get_obs_rvs_domain_types_and_params(obs_rvs,
                                                                               all_rvs_params=all_rvs_params,
                                                                               Vc_idx=Vc_idx, Vd_idx=Vd_idx)
    return _calc_marg_comp_log_prob(X, obs_rvs_domain_types, obs_rvs_params)


def calc_marg_log_prob(X, obs_rvs, w, all_rvs_params=None, g=None):
    """
    Calculate marginal log probabilities of observations
    :param g:
    :param X: M x N_o matrix of (partial) observations, where N_o is the number of obs nodes; alternatively a N_o vector
    :param obs_rvs: obs_rvs: length N_o list of observed rvs
    :param params:
    :return:
    """
    comp_log_probs = calc_marg_comp_log_prob(X, obs_rvs, all_rvs_params=all_rvs_params, g=g)  # M x K
    out = utils.logsumexp(np.log(w) + comp_log_probs, axis=-1)  # reduce along the last dimension
    return out


def calc_cond_mixture_weights(X, obs_rvs, w, all_rvs_params=None, g=None):
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
    comp_log_probs = calc_marg_comp_log_prob(X, obs_rvs, all_rvs_params=all_rvs_params, g=g)  # M x K
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


def marginal_map(X, obs_rvs, query_rv, w):
    """
    Calculate marginal MAP configuration of query_rv conditioned on a single observation X of obs_rvs.
    Assuming obs_rvs have been assigned belief_params.
    :param X: vector of observations
    :param obs_rvs:
    :param query_rv:
    :param w:
    :return:
    """
    if len(obs_rvs) == 0:  # no observation/conditioning
        cond_w = w
    else:
        cond_w = calc_cond_mixture_weights(X=X, obs_rvs=obs_rvs, w=w, all_rvs_params=None, g=None)
    if query_rv.domain_type[0] == 'd':
        map_state, map_prob = drv_belief_map(cond_w, query_rv.belief_params['pi'])
        # print(query_rv, map_state, map_prob)
        out = query_rv.values[map_state]
    else:
        assert query_rv.domain_type[0] == 'c'
        mu, var = query_rv.belief_params['mu'], query_rv.belief_params['var']
        bds = (query_rv.values[0], query_rv.values[1])
        out = crv_belief_map(cond_w, mu, var, bds)
    return out
