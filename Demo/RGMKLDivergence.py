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


rel_g = generate_rel_graph()
key_list = rel_g.key_list()

num_test = 5
evidence_ratio = 0.2

print('number of vr', len(key_list))
print('number of evidence', int(len(key_list) * evidence_ratio))

res = list()

for i in range(num_test):
    data = load_data('Data/RGM/' + str(evidence_ratio) + '_' + str(i))
    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    ans = GaBP(g)
    ans.run(15, log_enable=False)

    infer = C2FVI(g, num_mixtures=1, num_quadrature_points=3)
    infer.run(200, lr=0.2)

    for rv in g.rvs:
        if rv.value is None:
            res.append(KL(
                lambda x: infer.belief(x, rv),
                lambda x: ans.belief(x, rv),
                rv.domain
            ))

print('average KL', np.average(res))
