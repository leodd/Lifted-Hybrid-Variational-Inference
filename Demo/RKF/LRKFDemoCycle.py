from KalmanFilter import KalmanFilter
from Graph import *
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from EPBPLogVersion import EPBP
from HybridLBPLogVersion import HybridLBP
from GaBP import GaBP
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
from utils import kl_continuous


cluster_mat = scipy.io.loadmat('Demo/Data/RKF/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Demo/Data/RKF/well_t.mat')['well_t']
ans = scipy.io.loadmat('Demo/Data/RKF/LRKF_cycle.mat')['res']
param = scipy.io.loadmat('Demo/Data/RKF/LRKF_cycle.mat')['param']
print(well_t.shape)

idx = np.where(cluster_mat[:, 1] == 1)[0]
cluster_mat[idx[3:], 1] = 0
idx = np.where(cluster_mat[:, 2] == 1)[0]
cluster_mat[idx[:49], 2] = 0
cluster_mat[idx[52:], 2] = 0

well_t = well_t[:, 199:]
well_t[well_t[:, 0] == 5000, 0] = 0
well_t[well_t == 5000] = 1
t = 20

cluster_id = [1, 2]

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
kl = list()

for i in range(num_test):
    kmf = KalmanFilter(domain,
                       np.eye(n_sum) * param[2, i] + 0.01,
                       param[0, i],
                       np.eye(n_sum),
                       param[1, i])

    g, rvs_table = kmf.grounded_graph(t, data)
    # infer = EPBP(g, n=20, proposal_approximation='simple')
    # infer = HybridLBP(g, n=20, proposal_approximation='simple')
    infer = C2FVI(g, 1, 3)
    print('number of vr', len(g.rvs))
    num_evidence = 0
    for rv in g.rvs:
        if rv.value is not None:
            num_evidence += 1
    print('number of evidence', num_evidence)

    start_time = time.process_time()
    # infer.run(20, log_enable=False)
    # infer.run(20, c2f=0, log_enable=False)
    infer.run(200, 0.1)
    time_cost.append(time.process_time() - start_time)
    print('time lapse', time.process_time() - start_time)

    ans2 = GaBP(g)
    ans2.run(20, log_enable=False)

    temp_err = list()
    temp_err2 = list()
    temp_kl = list()

    for idx, rv in enumerate(rvs_table[t - 1]):
        res = infer.map(rv)
        temp_err.append(abs(res - ans[idx, i]))
        temp_err2.append(abs(res - ans2.map(rv)))
        temp_kl.append(kl_continuous(
            lambda x: infer.belief(x, rv),
            lambda x: ans2.belief(x, rv),
            rv.domain.values[0] - 20,
            rv.domain.values[1] + 20
        ))

    err.extend(temp_err)
    err2.extend(temp_err2)
    kl.extend(temp_kl)

    print(f'avg err {np.average(temp_err)}')
    print(f'avg err2 {np.average(temp_err2)}')
    print(f'avg kl {np.average(temp_kl)}')

print('########################')
print(f'avg time {np.average(time_cost)}')
print(f'avg err {np.average(err)}')
print(f'err std {np.std(err)}')
print(f'avg err2 {np.average(err2)}')
print(f'err std2 {np.std(err2)}')
print(f'avg kl {np.average(kl)}')
print(f'err kl std {np.std(kl)}')

