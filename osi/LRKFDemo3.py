import utils

utils.set_path()
seed = 0
utils.set_seed(seed=seed)

from KalmanFilter import KalmanFilter
from Graph import Domain
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaLBP import GaLBP
from GaBP import GaBP
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
from OneShot import OneShot, LiftedOneShot
from CompressedGraphSorted import CompressedGraphSorted

cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']
ans = scipy.io.loadmat('Data/LRKF_cycle.mat')['res']
param = scipy.io.loadmat('Data/LRKF_cycle.mat')['param']
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

domain = Domain((-4, 4), continuous=True, integral_points=np.linspace(-4, 4, 30))

num_test = param.shape[1]

result = np.zeros([n_sum, num_test])
ans2 = np.zeros([n_sum, num_test])
time_cost = list()
for i in range(num_test):
    utils.set_seed(seed=seed + i)

    kmf = KalmanFilter(domain,
                       np.eye(n_sum) * param[2, i] + 0.01,
                       param[0, i],
                       np.eye(n_sum),
                       param[1, i])

    g, rvs_table = kmf.grounded_graph(t, data)
    print('num rvs in g', len(g.rvs))
    print('num factors in g', len(g.factors))

    algo = 'LOSI'
    # algo = 'EPBP'
    if algo in ('OSI', 'LOSI'):
        utils.set_log_potential_funs(g.factors_list)  # OSI assumes factors have callable .log_potential_fun
        K = 3
        T = 20
        # lr = 1e-1
        lr = 5e-1
        its = 1000
        # fix_mix_its = int(its * 0.1)
        fix_mix_its = int(its * 1.0)
        # fix_mix_its = 500
        logging_itv = 50
        obs_rvs = [v for v in g.rvs if v.value is not None]
        evidence = {rv: rv.value for rv in obs_rvs}
        # cond = True
        cond = True
        if cond:
            cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs,
                                               evidence)  # this will also condition log_potential_funs

    if algo == 'EPBP':
        bp = EPBP(g, n=50, proposal_approximation='simple')
        print('number of vr', len(g.rvs))
        num_evidence = 0
        for rv in g.rvs:
            if rv.value is not None:
                num_evidence += 1
        print('number of evidence', num_evidence)

        start_time = time.process_time()
        bp.run(20, log_enable=False)
        time_cost.append(time.process_time() - start_time)
        print('time lapse', time.process_time() - start_time)

        for idx, rv in enumerate(rvs_table[t - 1]):
            result[idx, i] = bp.map(rv)

    elif algo == 'OSI':
        if cond:
            osi = OneShot(g=cond_g, K=K, T=T, seed=seed)
        else:
            osi = OneShot(g=g, K=K, T=T, seed=seed)
        start_time = time.process_time()
        osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
        time_cost.append(time.process_time() - start_time)
        # print('Mu =\n', osi.params['Mu'], '\nVar =\n', osi.params['Var'])
        print(algo, f'time {time_cost[-1]}')

        for idx, rv in enumerate(rvs_table[t - 1]):
            if cond:
                result[idx, i] = osi.map(obs_rvs=[], query_rv=rv)
            else:
                result[idx, i] = osi.map(obs_rvs=obs_rvs, query_rv=rv)

    elif algo == 'LOSI':
        if cond:
            cg = CompressedGraphSorted(cond_g)
        else:
            cg = CompressedGraphSorted(g)  # technically incorrect; currently we should run LOSI on the conditional MRF
        cg.run()
        print('number of rvs in cg', len(cg.rvs))
        print('number of factors in cg', len(cg.factors))
        osi = LiftedOneShot(g=cg, K=K, T=T, seed=seed)
        start_time = time.process_time()
        osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
        time_cost.append(time.process_time() - start_time)
        # print('Mu =\n', osi.params['Mu'], '\nVar =\n', osi.params['Var'])
        print(algo, f'time {time_cost[-1]}')

        for idx, rv in enumerate(rvs_table[t - 1]):
            if cond:
                result[idx, i] = osi.map(obs_rvs=[], query_rv=rv)
            else:
                result[idx, i] = osi.map(obs_rvs=obs_rvs, query_rv=rv)

    # bp = GaLBP(g)
    bp = GaBP(g)
    bp.run(20, log_enable=False)

    for idx, rv in enumerate(rvs_table[t - 1]):
        ans2[idx, i] = bp.map(rv)

    print(f'avg err {np.average(result[:, i] - ans[:, i])}')
    print(f'avg err2 {np.average(result[:, i] - ans2[:, i])}')

err = abs(result - ans)
err = np.average(err, axis=0)

err2 = abs(result - ans2)
err2 = np.average(err2, axis=0)

print('########################')
print(algo)
print(f'avg time {np.average(time_cost)}')
print(f'avg err {np.average(err)}')
print(f'err std {np.std(err)}')
print(f'avg err2 {np.average(err2)}')
print(f'err std2 {np.std(err2)}')
