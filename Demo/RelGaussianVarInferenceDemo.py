from RelationalGraph import *
from Potential import GaussianPotential
import numpy as np
import time


X = np.load('Data/rel_model_instances.npy')
D = np.load('Data/rel_model_data.npy')

instance_category = []
instance_bank = []
for row in X:
    if row[0] == 'category':
        instance_category.append(row[1])
    else:
        instance_bank.append(row[1])

instance_category = instance_category[:100]
instance_bank = instance_bank[:5]

data = dict()
for row in D:
    key = tuple([x.strip() for x in row[0].split(',')])
    if key[0] == 'revenue': continue
    data[key] = float(row[1])
# data[('recession', 'all')] = 50

domain_percentage = Domain((-100, 100), continuous=True, integral_points=linspace(-100, 100, 30))
domain_billion = Domain((-50, 50), continuous=True, integral_points=linspace(-50, 50, 30))

p1 = GaussianPotential([0., 0.], [[100., -70.], [-70., 100.]])
p2 = GaussianPotential([0., 0.], [[100., 50.], [50., 100.]])
p3 = GaussianPotential([0., 0.], [[100., 70.], [70., 100.]])

lv_recession = LV(('all',))
lv_category = LV(instance_category)
lv_bank = LV(instance_bank)

atom_recession = Atom(domain_percentage, logical_variables=(lv_recession,), name='recession')
atom_market = Atom(domain_percentage, logical_variables=(lv_category,), name='market')
atom_loss = Atom(domain_billion, logical_variables=(lv_category, lv_bank), name='loss')
atom_revenue = Atom(domain_billion, logical_variables=(lv_bank,), name='revenue')

f1 = ParamF(p1, nb=(atom_recession, atom_market))
f2 = ParamF(p2, nb=(atom_market, atom_loss))
f3 = ParamF(p3, nb=(atom_loss, atom_revenue))

rel_g = RelationalGraph()
rel_g.atoms = (atom_recession, atom_revenue, atom_loss, atom_market)
rel_g.param_factors = (f1, f2, f3)
rel_g.data = data

rel_g.init_nb()
g, rvs_table = rel_g.grounded_graph()

key_table = []
j = 0
for key in rvs_table:
    key_table.append(key)
    j += 1
num_test = 1
result_table = np.zeros((len(rvs_table), num_test))
time_table = []

from EPBPLogVersion import EPBP
from C2FVarInference import VarInference
from GaBP import GaBP

for i in range(num_test):
    # infer = EPBP(g, n=10, proposal_approximation='simple')
    infer = VarInference(g, num_mixtures=1, num_quadrature_points=3)
    start_time = time.clock()
    infer.run(200, lr=0.2)
    time_table.append(time.clock() - start_time)

    j = 0
    for key, rv in rvs_table.items():
        result_table[j, i] = infer.map(rv)
        j += 1

print('average time', np.mean(time_table))

for i in range(len(rvs_table)):
    key = key_table[i]
    mean = np.mean(result_table[i])
    variance = np.var(result_table[i])
    print(key, mean, variance)

bp = GaBP(g)
bp.run(20, log_enable=False)

for key, rv in rvs_table.items():
    print(key, bp.map(rv))
