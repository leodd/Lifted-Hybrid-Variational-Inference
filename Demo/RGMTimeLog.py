from RelationalGraph import *
from Potential import GaussianPotential
import numpy as np
import time
from Demo.Data.RGM.Generator import generate_rel_graph, load_data


rel_g = generate_rel_graph()
key_list = rel_g.key_list()


from EPBPLogVersion import EPBP
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from GaBP import GaBP


data = load_data('Data/RGM/' + str(0.2) + '_' + str(0))
rel_g.data = data
g, rvs_table = rel_g.grounded_graph()

time_log = dict()

# # infer = EPBP(g, n=10, proposal_approximation='simple')
# infer = VI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(200, lr=0.2)
# time_log['VI'] = infer.time_log
#
# infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(200, lr=0.2)
# time_log['LVI'] = infer.time_log
#
# infer = C2FVI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(200, lr=0.2)
# time_log['C2FVI'] = infer.time_log


import json


with open('Data/RGM/time_log', 'r') as file:
    s = file.read()
    time_log = json.loads(s)

# infer = C2FVI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(200, lr=0.2)
# time_log['C2FVI'] = infer.time_log
#
# with open('Data/RGM/time_log', 'w+') as file:
#     file.write(json.dumps(time_log))


import matplotlib.pyplot as plt


color = {
    'VI': 'r',
    'LVI': 'g',
    'C2FVI': 'b'
}
for name, t_log in time_log.items():
    x = list()
    y = list()
    for t, fe in t_log:
        if t > 350: break
        x.append(t)
        y.append(fe)
    plt.plot(x, y, color=color[name])

plt.show()
