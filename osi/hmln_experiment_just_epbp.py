import utils

utils.set_path(('..', '../gibbs'))
from RelationalGraph import *
from MLNPotential import *
from Potential import QuadraticPotential, TablePotential, HybridQuadraticPotential
from EPBPLogVersion import EPBP
# from OneShot import OneShot, LiftedOneShot
# from CompressedGraphSorted import CompressedGraphSorted
import numpy as np
import time

# from copy import copy

seed = 3
utils.set_seed(seed)

from KLDivergence import kl_continuous_logpdf

num_x = 4
num_y = 2

X = []
for x in range(num_x):
    X.append(f'x{x}')
Y = []
for y in range(num_y):
    Y.append(f'y{y}')
S = ['T1', 'T2', 'T3']

domain_bool = Domain((0, 1))
domain_real = Domain((-15, 15), continuous=True, integral_points=linspace(-15, 15, 20))

lv_x = LV(X)
lv_y = LV(X)
lv_s = LV(S)
lv_s2 = LV(S)

atom_A = Atom(domain_real, logical_variables=(lv_x,), name='A')
atom_B = Atom(domain_real, logical_variables=(lv_s,), name='B')
atom_C = Atom(domain_bool, logical_variables=(lv_x, lv_s), name='C')
atom_C2 = Atom(domain_bool, logical_variables=(lv_y, lv_s2), name='C')
atom_D = Atom(domain_bool, logical_variables=(lv_x, lv_y), name='D')

f1 = ParamF(  # disc
    MLNPotential(lambda x: imp_op(x[0] * x[1], x[2]), w=0.1),
    nb=(atom_D, atom_C, atom_C2),
    constrain=lambda sub: (sub[lv_s] == 'T1' and sub[lv_s2] == 'T1') or (sub[lv_s] == 'T1' and sub[lv_s2] == 'T2')
)

w_h = 1  # the stronger the more multi-modal things tend to be
f2 = ParamF(  # hybrid
    MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=w_h),
    nb=(atom_C, atom_A, atom_B)
)
equiv_hybrid_pot = HybridQuadraticPotential(
    A=w_h * np.array([np.array([[0., 0], [0, 0]]), np.array([[-1., 1.], [1., -1.]])]),
    b=w_h * np.array([[0., 0.], [0., 0.]]),
    c=w_h * np.array([0., 0.])
)  # equals 0 if x[0]==0, equals -(x[1]-x[1])^2 if x[0]==1

prior_strength = 0.01
f3 = ParamF(  # cont
    QuadraticPotential(A=-prior_strength * (np.eye(2)), b=np.array([0., 0.]), c=0.),
    nb=[atom_A, atom_B]
)  # needed to ensure normalizability; model will be indefinite when all discrete nodes are 0

rel_g = RelationalGraphSorted()
rel_g.atoms = (atom_A, atom_B, atom_C, atom_D)
rel_g.param_factors = (f1, f2, f3)
rel_g.init_nb()

data = dict()
num_tests = 1

import matplotlib.pyplot as plt

plt.figure()
xs = np.linspace(domain_real.values[0], domain_real.values[1], 100)

for test_num in range(num_tests):
    test_seed = seed + test_num

    # regenerate/reload evidence
    data.clear()
    B_vals = np.random.normal(loc=0, scale=5, size=len(S))  # special treatment for the story
    # B_vals = np.random.uniform(low=domain_real.values[0], high=domain_real.values[1], size=len(S))
    # B_vals = [-14, 2, 20]
    for i, s in enumerate(S):
        data[('B', s)] = B_vals[i]

    for x in X:
        for y in X:
            if x != y:
                data[('D', x, y)] = np.random.randint(2)

    evidence_ratio = 0.5
    x_idx = np.random.choice(len(X), int(len(X) * evidence_ratio), replace=False)
    for i in x_idx:
        data['C', X[i], 'T1'] = np.random.randint(2)

    x_idx = np.random.choice(len(X), int(len(X) * evidence_ratio), replace=False)
    for i in x_idx:
        data['C', X[i], 'T2'] = np.random.randint(2)

    x_idx = np.random.choice(len(X), int(len(X) * evidence_ratio), replace=False)
    for i in x_idx:
        data['C', X[i], 'T3'] = np.random.randint(2)

    print(data)

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    print('factors with duplicate nbrs:', )
    print([f.nb for f in g.factors_list if len(f.nb) != len(set(f.nb))])

    # labels of query nodes
    key_list = list()
    for x_ in X:
        if ('A', x_) not in data:
            key_list.append(('A', x_))
    query_rvs = [rvs_table[key] for key in key_list]

    print('number of rvs', len(g.rvs))
    print('num drvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'd']))
    print('num crvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'c']))
    print('number of factors', len(g.factors))
    print('number of evidence', len(data))
    #
    # obs_rvs = [v for v in g.rvs if v.value is not None]
    # evidence = {rv: rv.value for rv in obs_rvs}
    # cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs,
    #                                    evidence)  # this will also condition log_potential_funs
    #
    # print('cond number of rvs', len(cond_g.rvs))
    # print('cond num drvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'd']))
    # print('cond num crvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'c']))

    algo_name = 'EPBP'
    bp = EPBP(g, n=20, proposal_approximation='simple')
    start_time = time.process_time()
    start_wall_time = time.time()
    bp.run(10, log_enable=False)
    cpu_time = time.process_time() - start_time
    wall_time = time.time() - start_wall_time

    # temp storage
    mmap = np.zeros(len(query_rvs))
    obj = -1

    for i, rv in enumerate(query_rvs):
        mmap[i] = bp.map(rv)
        marg_logpdf = lambda x: bp.belief(x, rv, log_belief=True)  # probly slightly faster if not plotting
        # marg_logpdf = utils.curry_epbp_belief(bp, rv, log_belief=True)
        assert rv.domain_type[0] == 'c', 'only looking at kl for cnode queries for now'
        # lb, ub = -np.inf, np.inf
        lb, ub = -np.inf, np.inf
        # marg_kl = kl_continuous_logpdf(log_p=baseline_margs[i], log_q=margs[i], a=lb, b=ub)

        plt.plot(xs, [bp.belief(x, rv) for x in xs], label=f'{algo_name} for crv{i}')

    print('mmap pred', mmap)

plt.legend(loc='best')
plt.title('crv marginals')
# plt.show()
save_name = __file__.split('.py')[0]
plt.savefig('%s.png' % save_name)

print('######################')
from collections import OrderedDict
#
# avg_records = OrderedDict()
# for algo_name in algo_names:
#     record = records[algo_name]
#     avg_record = OrderedDict()
#     for record_field in record_fields:
#         avg_record[record_field] = (np.mean(record[record_field]), np.std(record[record_field]))
#     avg_records[algo_name] = avg_record
#
# from pprint import pprint
#
# for key, value in avg_records.items():
#     print(key + ':')
#     pprint(dict(value))
# # import json
# # output = json.dumps(avg_records, indent=0, sort_keys=True)
