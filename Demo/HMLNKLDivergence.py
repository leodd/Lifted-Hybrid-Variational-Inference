from RelationalGraph import *
from EPBPLogVersion import EPBP
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
import numpy as np
import time
from Demo.Data.HMLN.Generator import generate_rel_graph, load_data
from KLDivergence import kl_continuous
import osi.utils as utils
from scipy.integrate import quad


rel_g = generate_rel_graph()
key_list = rel_g.key_list()

num_test = 5
evidence_ratio = 0.01

print('number of vr', len(key_list))
print('number of evidence', int(len(key_list) * evidence_ratio))

kl = list()
map_err = list()

for i in range(num_test):
    data = load_data('Data/HMLN/' + str(i))
    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
    infer.run(100, lr=0.2)

    # infer = EPBP(g, n=20, proposal_approximation='simple')
    # infer.run(15)

    kl_temp = list()
    map_err_temp = list()

    for rv in g.rvs:
        if rv.value is None:
            # print(quad(lambda x: infer.belief(x, rv), rv.domain.values[0], rv.domain.values[1]))

            rv_idx = rvs_idx[rv]
            kl_temp.append(kl_continuous(
                lambda x: infer.belief(x, rv),
                # lambda x: ans.belief(x, rv),
                lambda x: norm_pdf(x, mu[rv_idx], sig[rv_idx, rv_idx]),
                rv.domain.values[0],
                rv.domain.values[1]
            ))

            map_err_temp.append(
                abs(infer.map(rv) - mu[rv_idx])
            )

    kl.extend(kl_temp)
    map_err.extend(map_err_temp)

    print('average KL:', np.average(kl_temp), '±', np.std(kl_temp))
    print('average MAP error:', np.average(map_err_temp), '±', np.std(map_err_temp))

print('average KL:', np.average(kl), '±', np.std(kl))
print('average MAP error:', np.average(map_err), '±', np.std(map_err))
