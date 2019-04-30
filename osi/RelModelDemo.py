import utils

utils.set_path()
from RelationalGraph import *
from Potential import GaussianPotential
from EPBPLogVersion import EPBP
from GaBP import GaBP
import numpy as np
import time
from OneShot import OneShot, LiftedOneShot
from CompressedGraphSorted import CompressedGraphSorted

seed = 0
utils.set_seed(seed=seed)

instance_category = []
instance_bank = []
for i in range(100):
# for i in range(50):
# for i in range(10):
    instance_category.append(f'c{i}')
for i in range(5):
    instance_bank.append(f'b{i}')

d = Domain((-50, 50), continuous=True, integral_points=linspace(-50, 50, 30))

# p1 = GaussianPotential([0., 0.], [[10., -7.], [-7., 10.]])
# p2 = GaussianPotential([0., 0.], [[10., 5.], [5., 10.]])
# p3 = GaussianPotential([0., 0.], [[10., 7.], [7., 10.]])

p1 = GaussianPotential([-20., 0.], [[10., -7.], [-7., 10.]])
p2 = GaussianPotential([0., 0.], [[10., 5.], [5., 10.]])
p3 = GaussianPotential([0., 20.], [[10., 7.], [7., 10.]])

lv_recession = LV(('all',))
lv_category = LV(instance_category)
lv_bank = LV(instance_bank)

atom_recession = Atom(d, logical_variables=(lv_recession,), name='recession')
atom_market = Atom(d, logical_variables=(lv_category,), name='market')
atom_loss = Atom(d, logical_variables=(lv_category, lv_bank), name='loss')
atom_revenue = Atom(d, logical_variables=(lv_bank,), name='revenue')

f1 = ParamF(p1, nb=(atom_recession, atom_market))
f2 = ParamF(p2, nb=(atom_market, atom_loss))
f3 = ParamF(p3, nb=(atom_loss, atom_revenue))

rel_g = RelationalGraph()
rel_g.atoms = (atom_recession, atom_revenue, atom_loss, atom_market)
rel_g.param_factors = (f1, f2, f3)
rel_g.init_nb()

key_list = rel_g.key_list()

data = dict()

avg_err = dict()
max_err = dict()
err_var = dict()
time_cost = dict()

num_test = 3
# evidence_ratio = 0.01
# evidence_ratio = 0.2
evidence_ratio = 0.

print('number of vr', len(key_list))
print('number of evidence', int(len(key_list) * evidence_ratio))

for _ in range(num_test):
    data.clear()
    idx_evidence = np.random.choice(len(key_list), int(len(key_list) * evidence_ratio), replace=False)
    for i in idx_evidence:
        key = key_list[i]
        data[key] = np.random.uniform(-30, 30)

    # data[('recession', 'all')] = np.random.uniform(-30, 30)

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    print('checking the validity of the Gaussian MRF...')
    J, _ = utils.get_info_mat_from_gaussian_mrf(g.factors, g.rvs_list)
    print('precision mat is diagonally dominant?', utils.check_diagonally_dominant(J))  # sufficient for normalizability
    print('determinant of the cov mat = ', 1 / np.linalg.det(J))

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
    print(ans)
    name = 'OSI'
    K = 5
    T = 20
    # lr = 1e-1
    lr = 5e-1
    its = 600
    # fix_mix_its = int(its * 0.1)
    # fix_mix_its = int(its * 1.0)
    fix_mix_its = 500
    logging_itv = 50
    osi = OneShot(g=g, K=K, T=T, seed=seed)
    start_time = time.process_time()
    osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
    print('Mu =\n', osi.params['Mu'], '\nVar =\n', osi.params['Var'])
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    err = []
    pred = {}
    for key in key_list:
        if key not in data:
            pred[key] = osi.map(rvs_table[key])
            err.append(abs(osi.map(rvs_table[key]) - ans[key]))
    print(pred)
    avg_err[name] = np.average(err) / num_test + avg_err.get(name, 0)
    max_err[name] = np.max(err) / num_test + max_err.get(name, 0)
    err_var[name] = np.average(err) ** 2 / num_test + err_var.get(name, 0)
    print(name, f'avg err {np.average(err)}')
    print(name, f'max err {np.max(err)}')

    name = 'LOSI'
    cg = CompressedGraphSorted(g)
    cg.run()
    print('number of rvs in cg', len(cg.rvs))
    print('number of factors in cg', len(cg.factors))
    osi = LiftedOneShot(g=cg, K=K, T=T, seed=seed)
    start_time = time.process_time()
    osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
    print('Mu =\n', osi.params['Mu'], '\nVar =\n', osi.params['Var'])
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    err = []
    for key in key_list:
        if key not in data:
            pred[key] = osi.map(rvs_table[key])
            err.append(abs(osi.map(rvs_table[key]) - ans[key]))
    print(pred)
    avg_err[name] = np.average(err) / num_test + avg_err.get(name, 0)
    max_err[name] = np.max(err) / num_test + max_err.get(name, 0)
    err_var[name] = np.average(err) ** 2 / num_test + err_var.get(name, 0)
    print(name, f'avg err {np.average(err)}')
    print(name, f'max err {np.max(err)}')

    run = False
    if run:
        name = 'EPBP'
        bp = EPBP(g, n=20)
        start_time = time.process_time()
        bp.run(15, log_enable=False)
        time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
        print(name, f'time {time.process_time() - start_time}')
        err = []
        for key in key_list:
            if key not in data:
                pred[key] = bp.map(rvs_table[key])
                err.append(abs(bp.map(rvs_table[key]) - ans[key]))
        print(pred)
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
