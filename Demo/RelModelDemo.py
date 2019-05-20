from RelationalGraph import *
from Potential import GaussianPotential
from EPBPLogVersion import EPBP
from GaBP import GaBP
import numpy as np
import time
from Demo.Data.RGM.Generator import generate_rel_graph, load_data


rel_g = generate_rel_graph()
key_list = rel_g.key_list()

avg_err = dict()
max_err = dict()
err_var = dict()
time_cost = dict()

num_test = 5
evidence_ratio = 0.01

print('number of vr', len(key_list))
print('number of evidence', int(len(key_list) * evidence_ratio))

for i in range(num_test):
    data = load_data('Data/RGM/' + str(evidence_ratio) + '_' + str(i))
    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    ans = dict()

    name = 'GaBP'
    bp = GaBP(g)
    start_time = time.process_time()
    bp.run(15, log_enable=False)
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    for key in key_list:
        if key not in data:
            ans[key] = bp.map(rvs_table[key])

    name = 'EPBP'
    bp = EPBP(g, n=20)
    start_time = time.process_time()
    bp.run(15, log_enable=False)
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    err = []
    for key in key_list:
        if key not in data:
            err.append(abs(bp.map(rvs_table[key]) - ans[key]))
    avg_err[name] = np.average(err) / num_test + avg_err.get(name, 0)
    max_err[name] = np.max(err) / num_test + max_err.get(name, 0)
    err_var[name] = np.average(err) ** 2 / num_test + err_var.get(name, 0)
    print(name, f'avg err {np.average(err)}')
    print(name, f'max err {np.max(err)}')

print('######################')
for name, v in time_cost.items():
    print(name, f'avg time {v}')
for name, v in avg_err.items():
    print(name, f'avg err {v}')
    print(name, f'err std {np.sqrt(err_var[name] - v ** 2)}')
for name, v in max_err.items():
    print(name, f'max err {v}')

