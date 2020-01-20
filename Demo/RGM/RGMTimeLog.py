from RelationalGraph import *
from Potential import GaussianPotential
import numpy as np
import time
from Demo.Data.RGM.Generator import generate_rel_graph, load_data
import json
from EPBPLogVersion import EPBP
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from GaBP import GaBP


rel_g = generate_rel_graph()

data = load_data('Demo/Data/RGM/time_log_20percent')
rel_g.ground_graph()
g, rvs_table = rel_g.add_evidence(data)

time_log = dict()

with open('Demo/Data/RGM/time_log_20_result', 'r') as file:
    s = file.read()
    time_log = json.loads(s)

# infer = EPBP(g, n=10, proposal_approximation='simple')

# infer = VI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(200, lr=0.2)
# time_log['VI'] = infer.time_log

infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
infer.run(200, lr=0.2)
time_log['LVI'] = infer.time_log

infer = C2FVI(g, num_mixtures=1, num_quadrature_points=3)
infer.run(200, lr=0.2)
time_log['C2FVI'] = infer.time_log


# with open('Demo/Data/RGM/time_log_20_result', 'w+') as file:
#     file.write(json.dumps(time_log))


import matplotlib.pyplot as plt


# ax = plt.subplot(111)
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(15)

color = {
    'VI': 'r',
    'LVI': 'g',
    'C2FVI': 'b'
}

dash = {
    'VI': [6, 0],
    'LVI': [6, 2],
    'C2FVI': [1, 1]
}

for name, t_log in time_log.items():
    x = list()
    y = list()
    for t, fe in t_log:
        if t > 125: break
        x.append(t)
        y.append(fe)
    plt.plot(x, y, color=color[name], dashes=dash[name])

# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)

# font = {'family': 'normal',
#         # 'weight': 'bold',
#         'size': 15}
#
# plt.rc('font', **font)

plt.legend(['BVI', 'Lifted BVI', 'C2F BVI'], )
plt.xlabel('time (second)')
plt.ylabel('free energy')
plt.show()
