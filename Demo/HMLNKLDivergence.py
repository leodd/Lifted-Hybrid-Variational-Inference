from RelationalGraph import *
from EPBPLogVersion import EPBP
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
import numpy as np
import time
from Demo.Data.HMLN.Generator import generate_rel_graph, load_data
from KLDivergence import kl_continuous
from scipy.integrate import quad


rel_g = generate_rel_graph()

num_test = 5

kl = list()
map_err = list()

for i in range(num_test):
    data = load_data('Data/HMLN/' + str(i))
    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()
    print('number of vr', len(g.rvs))
    print('number of evidence', sum([0 if rv.value is None else 1 for rv in g.rvs]))

    infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
    infer.run(100, lr=0.2)

    # infer = EPBP(g, n=20, proposal_approximation='simple')
    # infer.run(15)

    kl_temp = list()
    map_err_temp = list()

    for rv in g.rvs:
        if rv.domain.continuous and rv.value is None:
            kl_temp.append(kl_continuous(
                lambda x: infer.belief(x, rv),
                lambda x: ans.belief(x, rv),
                rv.domain.values[0],
                rv.domain.values[1]
            ))

            map_err_temp.append(
                abs(infer.map(rv) - ans.map(rv))
            )

    kl.extend(kl_temp)
    map_err.extend(map_err_temp)

    print('average KL:', np.average(kl_temp), '±', np.std(kl_temp))
    print('average MAP error:', np.average(map_err_temp), '±', np.std(map_err_temp))

print('average KL:', np.average(kl), '±', np.std(kl))
print('average MAP error:', np.average(map_err), '±', np.std(map_err))
