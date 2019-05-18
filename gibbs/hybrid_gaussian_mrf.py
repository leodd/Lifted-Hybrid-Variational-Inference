# Do Gibbs sampling in a hybrid Gaussian MRF, in which potentials involving continuous nodes (for each fixed set of
# discrete node values) are all exp quadratic
# Also assume all discrete nodes have discrete states 0, 1, 2, ..., i.e., consecutive ints that can be used for indexing
import numpy as np
from Potential import QuadraticPotential, GaussianPotential, LinearGaussianPotential, X2Potential, XYPotential, \
    LogQuadratic, LogTable, LogHybridQuadratic
import math

import utils


def convert_to_bn(factors, rvs):
    """
    Brute-force convert a hybrid Gaussian MRF into equivalent Bayesian network, p(x_d, x_c) = p(x_d) p(x_c | x_d)
    :param factors: list of Graph.F
    :param rvs: list of Graph.RV
    :return:
    """
    Vd = [rv for rv in rvs if rv.domain_type[0] == 'd']  # list of of discrete rvs
    Vc = [rv for rv in rvs if rv.domain_type[0] == 'c']  # list of cont rvs
    Vd_idx = {n: i for (i, n) in enumerate(Vd)}
    Vc_idx = {n: i for (i, n) in enumerate(Vc)}

    Nd = len(Vd)
    Nc = len(Vc)
    from itertools import product
    all_dstates = [rv.dstates for rv in Vd]  # [v1, v2, ..., v_Nd]
    all_disc_config = np.array(list(product(*all_dstates)))  # all joint configurations, \prod_i v_i by Nd
    disc_marginal_table = np.empty(all_dstates, dtype=float)  # [v1, v2, ..., v_Nd]
    gaussian_means = np.empty(all_dstates + [Nc], dtype=float)  # [v1, v2, ..., v_Nd, Nc]
    gaussian_covs = np.empty(all_dstates + [Nc, Nc], dtype=float)  # [v1, v2, ..., v_Nd, Nc, Nc]

    for factor in factors:  # pre-processing for efficiency; may be better done in Graph.py
        disc_nb_idx = ()
        cont_nb_idx = ()
        for rv in factor.nb:
            if rv.domain_type[0] == 'd':
                disc_nb_idx += Vd_idx[rv]
            else:
                assert rv.domain_type[0] == 'c'
                cont_nb_idx += Vc_idx[rv]
        factor.disc_nb_idx = disc_nb_idx
        factor.cont_nb_idx = cont_nb_idx

    for disc_config in all_disc_config:
        quadratic_factor_params = []  # each obtained from a (reduced) hybrid factor when given disc nb node values
        quadratic_factor_scopes = []
        disc_pot_prod = 0  # table/discrete potential's contribution to \prod_c \psi_c (product of all potentials)
        for factor in factors:
            # pot = factor.potential  # .potential is for interfacing with the code outside /osi
            assert hasattr(factor, 'log_potential_fun')  # actually I'll only work with log_potential_fun for simplicity
            lpot_fun = factor.log_potential_fun
            disc_vals_in_factor = tuple(disc_config[i] for i in factor.disc_nb_idx)
            if isinstance(lpot_fun, LogTable):
                disc_pot_prod += lpot_fun[disc_vals_in_factor]
            else:
                assert isinstance(lpot_fun, LogHybridQuadratic)
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
