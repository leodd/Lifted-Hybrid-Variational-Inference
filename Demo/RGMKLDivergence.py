from RelationalGraph import *
from Potential import GaussianPotential
from EPBPLogVersion import EPBP
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from GaBP import GaBP
import numpy as np
import time
from Demo.Data.RGM.Generator import generate_rel_graph, load_data
from KLDivergence import KL
import osi.utils as utils


def norm_pdf(x, mu, sig):
    u = (x - mu) / sig
    y = np.exp(-u * u * 0.5) / (2.506628274631 * sig)
    return y


rel_g = generate_rel_graph()
key_list = rel_g.key_list()

num_test = 5
evidence_ratio = 0.01

print('number of vr', len(key_list))
print('number of evidence', int(len(key_list) * evidence_ratio))

kl = list()
map_err = list()

for i in range(num_test):
    data = load_data('Data/RGM/' + str(evidence_ratio) + '_' + str(i))
    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    # ans = GaBP(g)
    # ans.run(15, log_enable=False)

    obs_rvs = [v for v in g.rvs if v.value is not None]
    evidence = {rv: rv.value for rv in obs_rvs}
    cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs, evidence)  # this will also condition log_potential_funs
    quadratic_params, rvs_idx = utils.get_quadratic_params_from_factor_graph(cond_g.factors, cond_g.rvs_list)

    mu, sig = utils.get_gaussian_mean_params_from_quadratic_params(A=quadratic_params[0], b=quadratic_params[1],
                                                              mu_only=False)

    infer = C2FVI(g, num_mixtures=1, num_quadrature_points=3)
    infer.run(200, lr=0.1)

    kl_temp = list()
    map_err_temp = list()

    for rv in g.rvs:
        if rv.value is None:
            rv_idx = rvs_idx[rv]
            kl_temp.append(KL(
                lambda x: infer.belief(x, rv),
                lambda x: norm_pdf(x, mu[rv_idx], sig[rv_idx]),
                rv.domain
            ))

            map_err_temp.append(
                abs(infer.map(rv) - mu[rv_idx])
            )

    kl.extend(kl_temp)
    map_err.extend(map_err_temp)

    print('average KL:', np.average(kl_temp))
    print('average MAP error:', np.average(map_err_temp), '±', np.std(map_err_temp))

print('average KL:', np.average(kl))
print('average MAP error:', np.average(map_err), '±', np.std(map_err))
