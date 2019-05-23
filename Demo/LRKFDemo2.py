from KalmanFilter import KalmanFilter
from Graph import *
from EPBPLogVersion import EPBP
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from GaBP import GaBP
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt


cluster_mat = scipy.io.loadmat('Data/RKF/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/RKF/well_t.mat')['well_t']
ans = scipy.io.loadmat('Data/RKF/LRKF_tree.mat')['res']
param = scipy.io.loadmat('Data/RKF/LRKF_tree.mat')['param']
print(well_t.shape)

well_t = well_t[:, 199:]
well_t[well_t[:, 0] == 5000, 0] = 0
well_t[well_t == 5000] = 1
t = 20

cluster_id = [1]

rvs_id = []
for i in cluster_id:
    rvs_id.append(np.where(cluster_mat[:, i] == 1)[0])

rvs_id = np.concatenate(rvs_id, axis=None)
n_sum = len(rvs_id)
data = well_t[rvs_id, :t]

domain = Domain((-4, 4), continuous=True, integral_points=linspace(-4, 4, 30))

num_test = param.shape[1]

time_cost = list()

err = list()
err2 = list()

for i in range(num_test):
    kmf = KalmanFilter(domain,
                       np.eye(n_sum) * param[2, i],
                       param[0, i],
                       np.eye(n_sum),
                       param[1, i])

    g, rvs_table = kmf.grounded_graph(t, data)
    # infer = EPBP(g, n=50, proposal_approximation='simple')
    infer = C2FVI(g, 1, 3)
    print('number of vr', len(g.rvs))
    num_evidence = 0
    for rv in g.rvs:
        if rv.value is not None:
            num_evidence += 1
    print('number of evidence', num_evidence)

    start_time = time.process_time()
    # infer.run(20, log_enable=False)
    infer.run(200, 0.1)
    time_cost.append(time.process_time() - start_time)
    print('time lapse', time.process_time() - start_time)

    ans2 = GaBP(g)
    ans2.run(20, log_enable=False)

    temp_err = list()
    temp_err2 = list()

    for idx, rv in enumerate(rvs_table[t - 1]):
        res = infer.map(rv)
        temp_err.append(abs(res - ans[idx, i]))
        temp_err2.append(abs(res - ans2.map(rv)))

    err.extend(temp_err)
    err2.extend(temp_err2)

    print(f'avg err {np.average(temp_err)}')
    print(f'avg err2 {np.average(temp_err2)}')


print('########################')
print(f'avg time {np.average(time_cost)}')
print(f'avg err {np.average(err)}')
print(f'err std {np.std(err)}')
print(f'avg err2 {np.average(err2)}')
print(f'err std2 {np.std(err2)}')
