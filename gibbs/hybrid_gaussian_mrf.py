# Do Gibbs sampling in a hybrid Gaussian MRF, in which potentials involving continuous nodes (for each fixed set of
# discrete node values) are all exp quadratic
# Also assume all discrete nodes have discrete states 0, 1, 2, ..., i.e., consecutive ints that can be used for indexing
import numpy as np
from Potential import LogQuadratic, LogTable, LogHybridQuadratic
from itertools import product
from copy import deepcopy

import utils


def convert_to_bn(factors, Vd, Vc):
    """
    Brute-force convert a hybrid Gaussian MRF into equivalent Bayesian network, p(x_d, x_c) = p(x_d) p(x_c | x_d)
    :param factors: list of Graph.F
    :param Vd: list of disc rvs
    :param Vc: list of cont rvs
    :return:
    """
    # Vd = [rv for rv in rvs if rv.domain_type[0] == 'd']  # list of of discrete rvs
    # Vc = [rv for rv in rvs if rv.domain_type[0] == 'c']  # list of cont rvs
    # Vd_idx = {n: i for (i, n) in enumerate(Vd)}
    # Vc_idx = {n: i for (i, n) in enumerate(Vc)}

    Nd = len(Vd)
    Nc = len(Vc)
    all_dstates = [rv.dstates for rv in Vd]  # [v1, v2, ..., v_Nd]
    all_disc_config = list(product(*[range(d) for d in all_dstates]))  # all joint configs, \prod_i v_i by Nd
    disc_marginal_table = np.empty(all_dstates, dtype=float)  # [v1, v2, ..., v_Nd]
    gaussian_means = np.empty(all_dstates + [Nc], dtype=float)  # [v1, v2, ..., v_Nd, Nc]
    gaussian_covs = np.empty(all_dstates + [Nc, Nc], dtype=float)  # [v1, v2, ..., v_Nd, Nc, Nc]

    # for factor in factors:  # pre-processing for efficiency; may be better done in Graph.py
    #     disc_nb_idx = ()
    #     cont_nb_idx = ()
    #     for rv in factor.nb:
    #         if rv.domain_type[0] == 'd':
    #             disc_nb_idx += (Vd_idx[rv],)
    #         else:
    #             assert rv.domain_type[0] == 'c'
    #             cont_nb_idx += (Vc_idx[rv],)
    #     factor.disc_nb_idx = disc_nb_idx
    #     factor.cont_nb_idx = cont_nb_idx

    for disc_config in all_disc_config:
        quadratic_factor_params = []  # each obtained from a (reduced) hybrid factor when given disc nb node values
        quadratic_factor_scopes = []
        disc_pot_prod = 0  # table/discrete potential's contribution to \prod_c \psi_c (product of all potentials)
        for factor in factors:
            # pot = factor.potential  # .potential is for interfacing with the code outside /osi
            assert hasattr(factor, 'log_potential_fun')  # actually I'll only work with log_potential_fun for simplicity
            lpot_fun = factor.log_potential_fun
            if isinstance(lpot_fun, LogQuadratic):
                quadratic_factor_params.append((lpot_fun.A, lpot_fun.b, lpot_fun.c))
                quadratic_factor_scopes.append(factor.cont_nb_idx)
            else:
                disc_vals_in_factor = tuple(disc_config[i] for i in factor.disc_nb_idx)
                if isinstance(lpot_fun, LogTable):
                    disc_pot_prod += lpot_fun(disc_vals_in_factor)
                else:
                    assert isinstance(lpot_fun, LogHybridQuadratic)
                    # plugging in disc values has the effect of selecting one log quadratic
                    quadratic_factor_params.append(lpot_fun.get_quadratic_params_given_x_d(disc_vals_in_factor))
                    quadratic_factor_scopes.append(factor.cont_nb_idx)

        joint_quadratic_params = utils.get_joint_quadratic_params(quadratic_factor_params, quadratic_factor_scopes, Nc)
        A, b, c = joint_quadratic_params
        mu, Sig = utils.get_gaussian_mean_params_from_quadratic_params(A, b, mu_only=False)
        gaussian_means[disc_config] = mu
        gaussian_covs[disc_config] = Sig

        # use the log partition function formula for Gaussian to figure out \int exp{x^T A x + x^T b} =
        # \int exp{-0.5 x^T J x + x^T J mu} = (int exp{-0.5 (x-mu)^T J (x-mu)} * exp{0.5 mu^T J mu}
        joint_quadratic_integral = (2 * np.pi) ** (Nc / 2) * np.linalg.det(Sig) ** 0.5 * \
                                   np.e ** (0.5 * np.dot(mu, b))
        joint_quadratic_integral *= np.e ** c  # integral of all the hybrid factors (with disc nodes substituted in)

        disc_pot_prod = np.e ** disc_pot_prod
        disc_marginal_table[disc_config] = \
            disc_pot_prod * joint_quadratic_integral  # prod of all potentials for this disc_config

    Z = np.sum(disc_marginal_table)
    disc_marginal_table /= Z

    return disc_marginal_table, gaussian_means, gaussian_covs


def get_crv_marg(disc_marginal_table, gaussian_means, gaussian_covs, Vc_idx, crv, flatten_params=True):
    """
    Find the parameters of the marginal distribution of a cont rv in a hybrid Gaussian MRF in Bayesian network
    parameterization; result is a univariate GMM whose number of components equals number of joint disc states
    :param disc_marginal_table: [v1, v2, ..., v_Nd]
    :param gaussian_means: [v1, v2, ..., v_Nd, Nc]
    :param gaussian_covs: [v1, v2, ..., v_Nd, Nc, Nc]
    :param Vc_idx:
    :param crv: rv object
    :return:
    """
    idx = Vc_idx[crv]
    weights = disc_marginal_table  # mixing weights
    cond_gaussian_means = gaussian_means[..., idx]  # [v1, v2, ..., v_Nd]
    cond_gaussian_vars = gaussian_covs[..., idx, idx]  # simply take diagonal; [v1, v2, ..., v_Nd]
    if flatten_params:
        weights, cond_gaussian_means, cond_gaussian_vars = \
            tuple(map(np.ravel, [weights, cond_gaussian_means, cond_gaussian_vars]))

    # might want to wrap things up in a GMM:
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    return weights, cond_gaussian_means, cond_gaussian_vars


def get_crv_marg_map(disc_marginal_table, gaussian_means, gaussian_covs, Vc_idx, crv, best_log_pdf=False):
    """

    :param disc_marginal_table:
    :param gaussian_means:
    :param gaussian_covs:
    :param Vc_idx:
    :param crv:
    :return:
    """

    weights, cond_gaussian_means, cond_gaussian_vars = get_crv_marg(disc_marginal_table, gaussian_means, gaussian_covs,
                                                                    Vc_idx, crv, flatten_params=True)
    bds = (crv.values[0], crv.values[1])
    return utils.get_scalar_gm_mode(w=weights, mu=cond_gaussian_means, var=cond_gaussian_vars, bds=bds,
                                    best_log_pdf=best_log_pdf)


def get_drv_marg(disc_marginal_table, Vd_idx, drv):
    """

    :param disc_marginal_table:
    :param drv:
    :return:
    """
    idx = Vd_idx[drv]
    Nd = len(disc_marginal_table.shape)
    all_axes_except_idx = list(range(Nd))
    all_axes_except_idx.remove(idx)
    all_axes_except_idx = tuple(all_axes_except_idx)
    return np.sum(disc_marginal_table, axis=all_axes_except_idx)


def get_drv_marg_map(disc_marginal_table, Vd_idx, drv, best_prob):
    """

    :param disc_marginal_table:
    :param Vd_idx:
    :param drv:
    :param best_prob:
    :return:
    """
    marg = get_drv_marg(disc_marginal_table, Vd_idx, drv)
    map_state = np.argmax(marg)
    if not best_prob:
        return map_state
    else:
        return map_state, marg[map_state]


# def get_mmap(disc_marginal_table, gaussian_means, gaussian_covs, rvs):


import disc_mrf


def block_gibbs_sample(factors, rvs, Vd, Vc, num_burnin, num_samples, init_x_d=None, disc_block_its=100):
    """

    :param factors:
    :param rvs:
    :param Vd:
    :param Vc:
    :return:
    """
    int_type = int

    Nd = len(Vd)
    Nc = len(Vc)

    all_dstates = [rv.dstates for rv in Vd]  # [v1, v2, ..., v_Nd]
    if init_x_d is None:  # will start by sampling p(x_c | x_d)
        init_x_d = np.array([np.random.randint(0, dstates) for dstates in all_dstates], dtype=int_type)
    x_d = init_x_d

    disc_samples = np.empty([num_samples, Nd], dtype=int_type)
    cont_samples = np.empty([num_samples, Nc], dtype=float)

    # pre-processing for sampling in cond MRF
    cont_factors = []
    disc_factors = []
    hybrid_factors = []  # strictly hybrid, i.e., contains both disc and cont
    for j, factor in enumerate(factors):
        lpot_fun = factor.log_potential_fun
        if isinstance(lpot_fun, LogQuadratic):
            cont_factors.append(factor)
        elif isinstance(lpot_fun, LogHybridQuadratic):
            if len(np.shape(lpot_fun.c)) >= 1:
                hybrid_factors.append(factor)
            else:
                cont_factors.append(factor)
        elif isinstance(lpot_fun, LogTable):
            disc_factors.append(factor)
        else:
            raise NotImplementedError

    contf_params = []
    contf_scopes = []
    for factor in cont_factors:
        lpot_fun = factor.log_potential_fun
        contf_params.append((lpot_fun.A, lpot_fun.b, lpot_fun.c))
        contf_scopes.append(factor.cont_nb_idx)
    contf_joint_quadratic_params = utils.get_joint_quadratic_params(contf_params, contf_scopes, Nc)

    discf_lpot_tables = []
    discf_scopes = []
    discf_nbr_factor_ids = [[] for _ in range(Nd)]
    for j, factor in enumerate(disc_factors):
        lpot_fun = factor.log_potential_fun
        discf_lpot_tables.append(lpot_fun.table)
        discf_scopes.append(factor.disc_nb_idx)

        for i in factor.disc_nb_idx:
            discf_nbr_factor_ids[i].append(j)
    num_disc_factors = len(disc_factors)

    for it in range(num_burnin + num_samples):
        # 1. sample p(x_c | x_d) in the conditional Gaussian MRF
        # basically copied from convert_to_bn:
        cond_quadratic_factor_params = []  # each obtained from a (reduced) hybrid factor when given disc nb node values
        cond_quadratic_factor_scopes = []
        # for factor in factors:  # TODO: speed up by precomputing joint_quadratic_params over cont_factors outside loop
        #     lpot_fun = factor.log_potential_fun
        #     if isinstance(lpot_fun, LogHybridQuadratic):
        #         disc_vals_in_factor = tuple(x_d[i] for i in factor.disc_nb_idx)  # () is OK
        #         cond_quadratic_factor_params.append(lpot_fun.get_quadratic_params_given_x_d(disc_vals_in_factor))
        #         cond_quadratic_factor_scopes.append(factor.cont_nb_idx)
        for factor in hybrid_factors:
            lpot_fun = factor.log_potential_fun
            disc_vals_in_factor = tuple(x_d[i] for i in factor.disc_nb_idx)
            cond_quadratic_factor_params.append(lpot_fun.get_quadratic_params_given_x_d(disc_vals_in_factor))
            cond_quadratic_factor_scopes.append(factor.cont_nb_idx)

        cond_joint_quadratic_params = utils.get_joint_quadratic_params(cond_quadratic_factor_params,
                                                                       cond_quadratic_factor_scopes, Nc)
        A = contf_joint_quadratic_params[0] + cond_joint_quadratic_params[0]
        b = contf_joint_quadratic_params[1] + cond_joint_quadratic_params[1]
        # c = contf_joint_quadratic_params[2] + cond_joint_quadratic_params[2]
        mu, Sig = utils.get_gaussian_mean_params_from_quadratic_params(A, b, mu_only=False)
        x_c = np.random.multivariate_normal(mean=mu, cov=Sig)

        # 2. sample p(x_d | x_c) in the conditional discrete MRF with table factors
        # get contribution from strictly hybrid factors
        cond_lpot_tables = []
        cond_scopes = []
        cond_disc_nbr_factor_ids = deepcopy(discf_nbr_factor_ids)  # TODO: ugly, somehow get rid of

        for factor in hybrid_factors:
            lpot_fun = factor.log_potential_fun
            cont_vals_in_factor = x_c[list(
                factor.cont_nb_idx)]  # note the index is a list (not a tuple) so we get another array instead of single element
            cond_lpot_tables.append(lpot_fun.get_table_params_given_x_c(cont_vals_in_factor))
            cond_scopes.append(factor.disc_nb_idx)
        for j, scope in enumerate(cond_scopes):
            for i in scope:
                cond_disc_nbr_factor_ids[i].append(j + num_disc_factors)
        cond_lpot_tables = discf_lpot_tables + cond_lpot_tables
        cond_scopes = discf_scopes + cond_scopes
        x_d = disc_mrf.gibbs_sample_one(cond_lpot_tables, cond_scopes, cond_disc_nbr_factor_ids, all_dstates, x_d,
                                        its=disc_block_its)

        sample_it = it - num_burnin
        if sample_it >= 0:
            disc_samples[sample_it] = x_d
            cont_samples[sample_it] = x_c

    return disc_samples, cont_samples
