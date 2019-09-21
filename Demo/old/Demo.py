from RelationalGraph import *
from MLNPotential import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from C2FVarInference import VarInference
import numpy as np
import time


num_x = 100
num_y = 2
num_s = 5

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

rel_g = RelationalGraph()
rel_g.atoms = (atom_A, atom_B, atom_C, atom_D, atom_E)
rel_g.param_factors = (f1, f2)
rel_g.init_nb()

num_test = 5

avg_diff = dict()
err_var = dict()
time_cost = dict()

data = dict()

for _ in range(num_test):
    data.clear()

    X_ = np.random.choice(num_x, int(num_x * 0.2), replace=False)
    for x_ in X_:
        data[('B', f'x{x_}')] = np.clip(np.random.normal(0, 5), -10, 10)

    X_ = np.random.choice(num_x, int(num_x * 1), replace=False)
    for x_ in X_:
        S_ = np.random.choice(num_s, 2, replace=False)
        for s_ in S_:
            data[('D', f'x{x_}', f's{s_}')] = np.random.choice([0, 1])

    for y_ in Y:
        S_ = np.random.choice(num_s, 5, replace=False)
        for s_ in S_:
            data[('E', y_, f's{s_}')] = np.random.choice([0, 1])

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    print(rvs_table)

    print('number of vr', len(g.rvs))
    print('number of evidence', len(data))

    key_list = list()
    for y_ in Y:
        key_list.append(('A', y_))
    for x_ in X:
        if ('B', x_) not in data:
            key_list.append(('B', x_))
