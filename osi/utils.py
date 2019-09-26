import tensorflow as tf
import numpy as np


def set_path(paths=('..',)):  # to facilitate importing within module
    import sys
    sys.path += list(paths)


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


def get_scalar_gm_log_prob(x, w, mu, var, get_fun=False):
    # convenience method; x can be tensor; mu, var, w should be len K vecs
    var_inv = 1 / var
    log_w = np.log(w)
    K = len(w)
    if isinstance(x, np.ndarray):  # tensor; reshape params for broadcasting
        mu = mu.reshape([K] + [1] * len(x.shape))
        var_inv = var_inv.reshape([K] + [1] * len(x.shape))
        log_w = log_w.reshape([K] + [1] * len(x.shape))
    else:
        # assert x must be scalar
        pass

    if not get_fun:
        comp_log_probs = -0.5 * np.log(2 * np.pi) + 0.5 * np.log(var_inv) - 0.5 * (x - mu) ** 2 * var_inv
        res = logsumexp(log_w + comp_log_probs, axis=0)
    else:  # curry the consts for speed; x was used for reshaping params
        const = -0.5 * np.log(2 * np.pi) + 0.5 * np.log(var_inv)
        coef = -0.5 * var_inv

        def f(arg):
            comp_log_probs = const + coef * (arg - mu) ** 2
            return logsumexp(log_w + comp_log_probs, axis=0)

        res = f

    return res


def get_scalar_gm_mode(w, mu, var, bds, best_log_pdf=False):
    """
    Find (approximately) the mode of a scalar gaussian mixture
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

    const = -0.5 * np.log(2 * np.pi) + 0.5 * np.log(var_inv)
    coef = -0.5 * var_inv

    def neg_gmm_log_prob(x):
        comp_log_probs = const + coef * (x - mu) ** 2
        return -logsumexp(log_w + comp_log_probs)

    res = []
    for m in set(mu):  # starting optimization from the K component modes and take the best solution
        x0 = m
        r = minimize(neg_gmm_log_prob, x0=x0, bounds=[bds])  # bounds kwarg needs to be a list of (lb, ub)
        res.append(r)
    best_res = min(res, key=lambda r: r.fun)
    x_opt = float(best_res.x)

    if not best_log_pdf:
        return x_opt
    else:
        return x_opt, -best_res.fun


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


def set_log_potential_funs(factors, skip_existing=True):
    """
    Set the log_potential_fun attribute of factors, ensuring that factors with the same potentials will have the same
    log potentials. This is a pre-processing step for OSI, as the graph construction/compression code typically work
    with potential objects instead of log potentials.
    :param factors: list of factors (whose log_potential_fun will be modified if not already set (i.e., is None))
    :return:
    """
    factors_with_unique_potentials, unique_potentials = get_unique_subsets(factors, lambda f: f.potential)
    for i, unique_potential in enumerate(unique_potentials):
        like_factors = factors_with_unique_potentials[i]
        if all(factor.log_potential_fun is not None for factor in like_factors) \
                and skip_existing:  # skip if all log_potential_fun already defined
            continue
        else:
            unique_log_potential_fun = unique_potential.to_log_potential()
            for factor in like_factors:
                factor.log_potential_fun = unique_log_potential_fun  # reference the same object


def convert_disc_MLNPotential_to_TablePotential(pot, nb):
    """
    :param pot:
    :return:
    """
    assert [rv.domain_type[0] == 'd' for rv in nb]
    all_nb_values = [rv.values for rv in nb]
    from itertools import product
    table = np.empty([rv.dstates for rv in nb])
    for config in product(*all_nb_values):
        table[config] = pot.get(config)
    from Potential import TablePotential
    return TablePotential(table, symmetric=np.all(table == table.T))


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


def get_conditional_quadratic(A, b, c, obs_args_vals):
    """
    Given a joint quadratic and values of a subset of the variables y, return parameters of the quadratic over the
    remaining variables x.
    [x y]^T A [x y] + b^T [x y] + c = x^T A_xx x + x^T A_xy y + y^T A_yx x + y^T A_yy y + b_x^T x + b_y^T y + c
    = x^T A_xx x + (A_xy y + y^T A_yx + b_x)^T x + y^T A_yy y + b_y^T y + c
    :param A:
    :param b:
    :param c:
    :param obs_args_vals:
    :return:
    """
    A = np.asarray(A)
    n = len(b)
    y_ind = np.array(list(obs_args_vals.keys()), dtype=int)
    y = np.array([obs_args_vals[i] for i in y_ind])
    x_ind = np.setdiff1d(np.arange(n), y_ind)
    b_y = b[y_ind]
    b_x = b[x_ind]
    A_yy = A[np.ix_(y_ind, y_ind)]
    A_xx = A[np.ix_(x_ind, x_ind)]
    A_xy = A[np.ix_(x_ind, y_ind)]
    A_yx = A[np.ix_(y_ind, x_ind)]

    A_cond = A_xx
    b_cond = A_xy @ y + (A_yx.T @ y).T + b_x
    c_cond = np.dot(y, A_yy @ y) + np.dot(b_y, y) + c
    return A_cond, b_cond, c_cond


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
    from Potential import QuadraticPotential, GaussianPotential, LinearGaussianPotential, X2Potential, XYPotential
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
                if isinstance(potential, (QuadraticPotential, GaussianPotential, LinearGaussianPotential, X2Potential,
                                          XYPotential)):
                    A, b, c = potential.get_quadratic_params()
                    A_cond, b_cond, c_cond = get_conditional_quadratic(A, b, c, partial_args_vals)
                    pot = QuadraticPotential(A_cond, b_cond, c_cond)
                    log_pot = pot.to_log_potential()
                elif isinstance(potential, MLNPotential):
                    formula = get_partial_function(potential.formula, n, partial_args_vals)
                    pot = MLNPotential(formula=formula, w=potential.w)
                    log_pot = pot.to_log_potential()
                else:  # may not be able to construct the conditional potential in closed-form
                    pot = copy(potential)
                    pot.get = get_partial_function(potential.get, n, partial_args_vals)
                    assert factor.log_potential_fun is not None, \
                        "factor.log_potential_fun hasn't been set in the original graph, don't know how to condition"
                    log_pot = get_partial_function(factor.log_potential_fun, n, partial_args_vals)

                if hasattr(potential, 'symmetric'):
                    pot.symmetric = potential.symmetric

                f.potential = pot
                f.log_potential_fun = log_pot
        else:
            f = factor
        cond_factors.append(f)
    return cond_factors


def get_unique_quadratic_params(quadratic_params):
    """
    Given a list of quadratic params [(A0, b0, c0), (A1, b1, c1), ... ], determine the unique subset of params
    and get the indices of corresponding params in quadratic_params.
    :param quadratic_params:
    :return: unique_subsets_ids, unique_params
    """
    N = len(quadratic_params)
    A0, b0, c0 = quadratic_params[0]
    # n = b0.size()
    all_params_flat = np.array([np.hstack((params[0].ravel(), params[1], params[2])) for params in quadratic_params])
    # will have shape N x (n^2+n+1)
    unique_params_flat, unique_params_idx = np.unique(all_params_flat, return_inverse=True, axis=0)

    unique_params = []
    for upf in unique_params_flat:
        A = upf[:A0.size].reshape(A0.shape)
        b = upf[A0.size:-1]
        c = upf[-1]
        unique_params.append((A, b, c))
    return unique_params, unique_params_idx


def condense_duplicate_factor_potentials(factors):
    """
    Given a list of factors, identify those potentials that are actually the same (easy for generic parametric
    potentials like Quadratic or Table; tricky for MLN b/c opaque formula) and assign them the same potential objects
    :param factors:
    :return:
    """
    from Potential import QuadraticPotential  # , GaussianPotential, LinearGaussianPotential, X2Potential, XYPotential
    # all_pots = [f.potential for f in factors]
    # currently only handle Quadratic

    quad_factors = [f for f in factors if isinstance(f.potential, QuadraticPotential)]
    assert np.all(np.array([len(f.nb) for f in quad_factors])
                  == np.array(
        [f.potential.dim() for f in quad_factors])), 'num nbrs in each potential must match potential.dim'
    quad_pots = [f.potential for f in quad_factors]

    # quad_factors_with_uniq_dims, uniq_dims = get_unique_subsets(quad_pots, lambda p: p.dim)

    quad_factors_with_uniq_dims, uniq_dims = get_unique_subsets(quad_factors, lambda f: len(f.nb))
    for fs in quad_factors_with_uniq_dims:  # all fs have the same number of nbr/dimensions
        fs_params = [f.potential.get_quadratic_params() for f in fs]
        unique_params, unique_idx = get_unique_quadratic_params(fs_params)
        unique_pots = [QuadraticPotential(*params) for params in unique_params]
        unique_lpots = [p.to_log_potential() for p in unique_pots]
        for i, f in enumerate(fs):
            f.potential = unique_pots[unique_idx[i]]
            f.log_potential_fun = unique_lpots[unique_idx[i]]


def set_nbrs_idx_in_factors(factors, Vd_idx, Vc_idx):
    # create factor.disc_nb_idx/cont_nb_idx attrs for convenience (mostly used in hybrid mln baseline)
    for factor in factors:
        disc_nb_idx = ()
        cont_nb_idx = ()
        for rv in factor.nb:
            if rv.domain_type[0] == 'd':
                disc_nb_idx += (Vd_idx[rv],)
            else:
                assert rv.domain_type[0] == 'c'
                cont_nb_idx += (Vc_idx[rv],)
        factor.disc_nb_idx = disc_nb_idx
        factor.cont_nb_idx = cont_nb_idx


def get_conditional_mrf(factors, rvs, evidence, update_rv_nbs=False):  # TODO: fix this hot OOP mess
    """

    :param factors: an iterable of factors
    :param rvs: an iterable of rvs
    :param evidence: a dict mapping rvs (Graph.RV objects) to numerical values
    :param update_rv_nbs: if True, will run g.init_nb on the returned graph g; this will modify the rvs passed in
    :return: g, containing unobserved rvs and conditional factors with scopes reduced to the unobserved rvs
    WARNING: the returned graph will share ref to the rv objects that were passed in, so use update_rv_nbs with caution,
    as this will propagate to the original graph which they belong to; default is to have update_rv_nbs=False, so the
    returned graph rvs will likely have the wrong .nb (neighboring factors).
    """
    cond_factors = condition_factors_on_evidence(factors, evidence)
    cond_factors = list(filter(lambda f: len(f.nb) > 0, cond_factors))  # keep only non-empty factors
    remaining_rvs = [rv for rv in rvs if rv not in evidence]

    from Graph import Graph
    g = Graph()
    g.rvs = remaining_rvs
    g.factors = cond_factors
    if update_rv_nbs:
        print('WARNING: updating rvs to use neighboring factors in the conditioned graph')
        g.init_nb()  # update .nb attributes of rvs
    return g


def get_joint_quadratic_params(factor_params, factor_scopes, N=None):
    """
    Get the parameters A, b, c of a joint quadratic function x^T A x + b^T x + c defined by a list of quadratic factors
    over sub-vectors of x.
    :param factors: list of tuples, like [(A1, b1, c1), (A2, b2, c2), ...]
    :param rvs_list: list of tuples of ints, which give ids of variables in each factor, like [(0,2,3), (3,0), ...]
    :param N: total num of vars
    :return:
    """
    from itertools import chain
    factor_scopes_flat = list(chain.from_iterable(factor_scopes))
    factor_scopes_flat = np.array(factor_scopes_flat)
    assert np.all(factor_scopes_flat >= 0)
    if N is None:
        N = np.max(factor_scopes_flat) + 1
    A = np.zeros([N, N], dtype='float')
    b = np.zeros(N, dtype='float')
    c = 0
    for params, scope in zip(factor_params, factor_scopes):
        A_, b_, c_ = params
        n = len(scope)
        for i in range(n):
            x_i = scope[i]
            for j in range(n):
                # A[scope[i], scope[j]] += A_[i, j]
                A[x_i, scope[j]] += A_[i, j]
            b[x_i] += b_[i]

        c += c_

    return A, b, c


def get_quadratic_params_from_factor_graph(factors, rvs_list):
    """
    Get the parameters A, b, c of a joint quadratic function x^T A x + b^T x + c defined by exp quadratic potentials
    :param factors:
    :param rvs_list:
    :return:
    """
    N = len(rvs_list)
    rvs_idx = {rv: i for (i, rv) in enumerate(rvs_list)}

    from Potential import QuadraticPotential, GaussianPotential, LinearGaussianPotential, X2Potential, XYPotential, \
        LogQuadratic
    factor_params = []
    factor_scopes = []
    for factor in factors:
        pot = factor.potential
        if hasattr(factor, 'log_potential_fun') and factor.log_potential_fun is not None:
            lpot_fun = factor.log_potential_fun
            assert isinstance(lpot_fun, LogQuadratic)
            params = lpot_fun.A, lpot_fun.b, lpot_fun.c
        else:
            assert isinstance(pot, (QuadraticPotential, GaussianPotential, LinearGaussianPotential, X2Potential,
                                    XYPotential))
            params = pot.get_quadratic_params()
        factor_params.append(params)

        scope = tuple(rvs_idx[rv] for rv in factor.nb)
        factor_scopes.append(scope)

    joint_params = get_joint_quadratic_params(factor_params, factor_scopes, N)

    return joint_params, rvs_idx


def get_gaussian_mean_params_from_quadratic_params(A, b, mu_only=True):
    """
    Get mu, Sig, such that -1/2 (x-mu)^T Sig^{-1} (x-mu) = x^T A x + b^T + const.
    If mu_only, only mu is computed (by solving linear equations; no matrix inversion)
    :param A:
    :param b:
    :param c:
    :return:
    """
    J = -2. * A  # precision matrix
    if mu_only:
        mu = np.linalg.solve(J, b)
        return mu
    else:
        Sig = np.linalg.inv(J)
        mu = Sig @ b
        return mu, Sig


def get_prec_mat_from_gaussian_mrf(factors, rvs_list):  # superseded by get_quadratic_params_from_factor_graph
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
    expanded_arrs = expand_dims_for_fun_grid(arrs)

    if sep_args:
        res = fun(*expanded_arrs)
    else:
        res = fun(expanded_arrs)  # should evaluate on the Cartesian product (ndgrid) of axes by broadcasting
    return res


def broadcast_arrs_to_common_shape(arrs, backend):
    """
    Return a list of arrays that are all broadcasted to a common shape.
    e.g., if arrs have shapes [1, 3, 1] and [2, 1, 3], result will be a list of 2 arrays both with shape [2, 3, 3]
    :param arrs: list of arrays/tensors broadcastable to the same largest shape.
    :return:
    """
    # if len(arrs) == 1:
    #     return arrs
    arrs_shapes = []
    for arr in arrs:
        if hasattr(arr, 'shape'):
            shape = tuple(map(int, arr.shape))
        else:
            shape = np.array(arr).shape
        arrs_shapes.append(shape)

    if len(set(arrs_shapes)) == 1:  # all have the same shapes
        out = arrs
    else:
        max_ndims = max([len(s) for s in arrs_shapes])
        for i in range(len(arrs)):  # prepend empty dims if necessary
            shape = arrs_shapes[i]
            prepend_ndims = max_ndims - len(shape)
            if prepend_ndims > 0:
                arrs[i] = backend.reshape(arrs[i], (1,) * prepend_ndims + shape)
        # assert len(set([len(s) for s in arrs_shapes])) == 1, 'arrs should have the same number of dimensions'
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


# miscellaneous
def curry_epbp_belief(bp, rv, log_belief=True):
    # capture environment variables to avoid dangling ref; may use too much mem
    def f(x):
        return bp.belief(x, rv, log_belief=log_belief)

    return f


def curry_normal_logpdf(mu, var):
    from scipy.stats import norm
    sig = var ** 0.5

    def f(x):
        return norm.logpdf(x, loc=mu, scale=sig)

    return f


# moved from /KLDivergence (triggered by commit 38ba95034)
def kl_discrete(p, q):
    """
    Compute KL(p||q)
    p, q are probability table (tensors) of the same shape
    :param p:
    :param q:
    :return:
    """

    return np.sum(p * (np.log(p) - np.log(q)))  # naive implementation


# def kl_continuous_no_add_const(p, q, a, b, *args, **kwargs):
#     """
#     Compute KL(p||q), 1D
#     :param p:
#     :param q:
#     :param a:
#     :param b:
#     :param kwargs:
#     :return:
#     """
#
#     def integrand(x):
#         px = p(x)
#         qx = q(x)
#         return np.log(px ** px) - np.log(qx ** px)
#
#     res = integrate.quad(integrand, a, b, *args, **kwargs)
#     if 'full_result' in kwargs and kwargs['full_result']:
#         return res
#     else:
#         return res[0]
#
#
# def kl_continuous(p, q, a, b, *args, **kwargs):
#     """
#     Compute KL(p||q), 1D
#     :param p:
#     :param q:
#     :param a:
#     :param b:
#     :param kwargs:
#     :return:
#     """
#
#     def integrand(x):
#         px = p(x) + 1e-100
#         qx = q(x) + 1e-100
#         return px * (np.log(px) - np.log(qx))
#
#     res = integrate.quad(integrand, a, b, *args, **kwargs)
#     if 'full_result' in kwargs and kwargs['full_result']:
#         return res
#     else:
#         return res[0]


def kl_continuous_logpdf(log_p, log_q, a, b, *args, **kwargs):
    """
    Compute KL(p||q), 1D
    :param p:
    :param q:
    :param a:
    :param b:
    :param kwargs:
    :return:
    """
    from scipy.integrate import quad
    def integrand(x):
        logpx = log_p(x)
        logqx = log_q(x)
        px = np.exp(logpx)
        return px * (logpx - logqx)

    res = quad(integrand, a, b, *args, **kwargs)
    if 'full_result' in kwargs and kwargs['full_result']:
        return res
    else:
        return res[0]


def kl_normal(mu1, mu2, sig1, sig2):
    """
    Compute KL(p||q), 1D
    p ~ N(mu1, sig1^2), q ~ N(mu2, sig2^2)
    $KL(p, q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}$
    :param mu1:
    :param mu2:
    :param sig1:
    :param sig2:
    :return:
    """
    res = np.log(sig2) - np.log(sig1) + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2) - 0.5
    return res
