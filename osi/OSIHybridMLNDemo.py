import utils

utils.set_path()
from RelationalGraph import *
from MLNPotential import *
from EPBPLogVersion import EPBP
from OneShot import OneShot
import numpy as np
import time

seed = 9
utils.set_seed(seed)

# num_x = 10
# num_y = 2
# num_s = 5

num_x = 2
num_y = 2
num_s = 2

X = []
for x in range(num_x):
    X.append(f'x{x}')
Y = []
for y in range(num_y):
    Y.append(f'y{y}')
S = []
for s in range(num_s):
    S.append(f's{s}')

domain_bool = Domain((0, 1))
domain_real = Domain((-15, 15), continuous=True, integral_points=linspace(-15, 15, 20))

lv_x = LV(X)
lv_y = LV(Y)
lv_s = LV(S)

atom_A = Atom(domain_real, logical_variables=(lv_y,), name='A')
atom_B = Atom(domain_real, logical_variables=(lv_x,), name='B')
atom_C = Atom(domain_bool, logical_variables=(lv_x, lv_y), name='C')
atom_D = Atom(domain_bool, logical_variables=(lv_x, lv_s), name='D')
atom_E = Atom(domain_bool, logical_variables=(lv_y, lv_s), name='E')

f1 = ParamF(
    MLNPotential(lambda x: imp_op(x[0] * x[1], x[2]), w=1), nb=(atom_D, atom_E, atom_C)
)
f2 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=0.01), nb=(atom_C, atom_A, atom_B)
)

f3 = ParamF(
    MLNPotential(lambda atom: -0.1 * atom[0] ** 2),  # just to help ensure normalizability
    nb=[atom_A]
)

rel_g = RelationalGraph()
rel_g.atoms = (atom_A, atom_B, atom_C, atom_D, atom_E)
rel_g.param_factors = (f1, f2, f3)
rel_g.init_nb()

num_tests = 2  # num rounds with different queries
num_runs = 3

avg_diff = dict()
err_var = dict()
time_cost = dict()

data = dict()

for _ in range(num_tests):
    data.clear()

    X_ = np.random.choice(num_x, int(num_x * 0.2), replace=False)
    for x_ in X_:
        data[('B', f'x{x_}')] = np.clip(np.random.normal(0, 5), -10, 10)

    X_ = np.random.choice(num_x, int(num_x * 1), replace=False)
    for x_ in X_:
        # S_ = np.random.choice(num_s, 2, replace=False)
        S_ = np.random.choice(num_s, int(num_s * 1), replace=False)
        for s_ in S_:
            data[('D', f'x{x_}', f's{s_}')] = np.random.choice([0, 1])

    for y_ in Y:
        # S_ = np.random.choice(num_s, 5, replace=False)
        S_ = np.random.choice(num_s, int(num_s * 1), replace=False)
        for s_ in S_:
            data[('E', y_, f's{s_}')] = np.random.choice([0, 1])

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    print(rvs_table)

    print('number of rvs', len(g.rvs))
    print('number of factors', len(g.factors))
    print('number of evidence', len(data))

    key_list = list()
    for y_ in Y:
        key_list.append(('A', y_))
    for x_ in X:
        if ('B', x_) not in data:
            key_list.append(('B', x_))

    ans = dict()

    name = 'EPBP'
    res = np.zeros((len(key_list), num_runs))
    for j in range(num_runs):
        bp = EPBP(g, n=20, proposal_approximation='simple')
        start_time = time.process_time()
        bp.run(10, log_enable=False)
        time_cost[name] = (time.process_time() - start_time) / num_runs / num_tests + time_cost.get(name, 0)
        print(name, f'time {time.process_time() - start_time}')
        for i, key in enumerate(key_list):
            res[i, j] = bp.map(rvs_table[key])
    for i, key in enumerate(key_list):
        ans[key] = np.average(res[i, :])
    for i, key in enumerate(key_list):
        res[i, :] -= ans[key]
    avg_diff[name] = np.average(np.average(abs(res), axis=1)) / num_tests + avg_diff.get(name, 0)
    err_var[name] = np.average(np.average(res ** 2, axis=1)) / num_tests + err_var.get(name, 0)
    print(name, 'diff', np.average(np.average(abs(res), axis=1)))
    print(name, 'var', np.average(np.average(res ** 2, axis=1)) - np.average(np.average(abs(res), axis=1)) ** 2)

    name = 'OSI'
    K = 5
    T = 20
    lr = 0.1
    its = 200
    fix_mix_its = int(its * 0.)
    res = np.zeros((len(key_list), num_runs))
    osi = OneShot(g=g, K=K, T=T, seed=seed)  # can be moved outside of all loops, as the ground MRF doesn't change
    for j in range(num_runs):
        start_time = time.process_time()
        osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its)
        time_cost[name] = (time.process_time() - start_time) / num_runs / num_tests + time_cost.get(name, 0)
        print(name, f'time {time.process_time() - start_time}')
        for i, key in enumerate(key_list):
            res[i, j] = osi.map(rvs_table[key])
    for i, key in enumerate(key_list):
        res[i, :] -= ans[key]
    avg_diff[name] = np.average(np.average(abs(res), axis=1)) / num_tests + avg_diff.get(name, 0)
    err_var[name] = np.average(np.average(res ** 2, axis=1)) / num_tests + err_var.get(name, 0)
    print(name, 'diff', np.average(np.average(abs(res), axis=1)))
    print(name, 'var', np.average(np.average(res ** 2, axis=1)) - np.average(np.average(abs(res), axis=1)) ** 2)

print('######################')
for name, v in time_cost.items():
    print(name, f'avg time {v}')
for name, v in avg_diff.items():
    print(name, f'diff {v}')
    print(name, f'std {np.sqrt(err_var[name] - v ** 2)}')
