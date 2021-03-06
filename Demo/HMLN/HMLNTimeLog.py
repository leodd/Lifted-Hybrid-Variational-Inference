from utils import log_likelihood
# from Demo.Data.HMLN.GeneratorRobotMapping import generate_rel_graph, load_raw_data
from Demo.Data.HMLN.GeneratorPaperPopularity import generate_rel_graph, load_data
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from EPBPLogVersion import EPBP
from HybridMaxWalkSAT import HybridMaxWalkSAT as HMWS
import json


rel_g = generate_rel_graph()
_, rvs_dict = rel_g.ground_graph()

# query = dict()
# for key, rv in rvs_dict.items():
#     if key[0] == 'SegType' or key[0] == 'PartOf' or key[0] == 'Length' or key[0] == 'Depth':
#         query[key] = rv
#
# data = load_raw_data('Demo/Data/HMLN/robot-map')
# for key, rv in rvs_dict.items():
#     if key not in data and key not in query and not rv.domain.continuous:
#         data[key] = 0  # closed world assumption

query = dict()
for key, rv in rvs_dict.items():
    if key[0] == 'PaperPopularity' or key[0] == 'TopicPopularity':
        query[key] = rv

data = load_data('Demo/Data/HMLN/0')

# data = dict()

g, rvs_dict = rel_g.add_evidence(data)
print(len(rvs_dict))
print(len(g.factors))

time_log = dict()

with open('Demo/Data/HMLN/paper-popularity-time-log', 'r') as file:
    s = file.read()
    time_log = json.loads(s)

# infer = HMWS(g)
# infer.run(max_tries=1, max_flips=10000, epsilon=0.7, noise_std=0.3)
# time_log['HMWS'] = infer.time_log

# infer = VI(g, num_mixtures=2, num_quadrature_points=3)
# infer.run(100, lr=0.2, log_fe=False)
# time_log['VI'] = infer.time_log

# infer = LVI(g, num_mixtures=2, num_quadrature_points=3)
# infer.run(100, lr=0.2, log_fe=False)
# time_log['LVI'] = infer.time_log

# infer = C2FVI(g, num_mixtures=2, num_quadrature_points=3)
# infer.run(100, lr=0.2, log_fe=False)
# time_log['C2FVI'] = infer.time_log


# with open('Demo/Data/HMLN/paper-popularity-time-log', 'w+') as file:
#     file.write(json.dumps(time_log))


import matplotlib.pyplot as plt


# ax = plt.subplot(111)
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(15)

color = {
    'HMWS': 'y',
    'VI': 'r',
    'LVI': 'g',
    'C2FVI': 'b',
}

dash = {
    'HMWS': [6, 0],
    'VI': [6, 1],
    'LVI': [3, 3],
    'C2FVI': [1, 1],
}

max_t = 80

for name, t_log in time_log.items():
    x = list()
    y = list()
    for t, ll in t_log:
        if t > max_t:
            break
        x.append(t)
        y.append(ll)
    x.append(max_t)
    y.append(y[-1])
    plt.plot(x, y, color=color[name], dashes=dash[name])

# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)

# font = {'family': 'normal',
#         # 'weight': 'bold',
#         'size': 15}
#
# plt.rc('font', **font)

plt.legend(['HMWS', 'BVI', 'Lifted BVI', 'C2F BVI'], )
plt.xlabel('time (second)')
plt.ylabel('negative log probability')
plt.show()
