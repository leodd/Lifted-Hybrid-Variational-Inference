from Graph import *
from RelationalGraph import *
from MLNPotential import *

instance = [
    'Joey',
    'Rachel',
    'Tim',
]

data = {
    ('Friend', 'Joey', 'Rachel'): 1,
    ('Friend', 'Joey', 'Tim'): 1,
    ('Friend', 'Rachel', 'Joey'): 1,
    ('Friend', 'Rachel', 'Tim'): 0,
    ('Friend', 'Tim', 'Joey'): 1,
    ('Friend', 'Tim', 'Rachel'): 0,
    ('Smoke', 'Tim'): 1,
}

domain_bool = Domain((0, 1))

lv_x = LV(instance)
lv_y = LV(instance)

# formulas:
# Friend(X, Y) => [Smoke(X) <=> Smoke(Y)]
# Smoke(X) => Cancer(X)

atom_friend = Atom(domain=domain_bool, logical_variables=[lv_x, lv_y], name='Friend')
atom_smoke_x = Atom(domain=domain_bool, logical_variables=[lv_x], name='Smoke')
atom_smoke_y = Atom(domain=domain_bool, logical_variables=[lv_y], name='Smoke')
atom_cancer = Atom(domain=domain_bool, logical_variables=[lv_x], name='Cancer')

f1 = ParamF(
    MLNPotential(lambda atom: imp_op(atom[0], bic_op(atom[1], atom[2])), 0.1),
    nb=[atom_friend, atom_smoke_x, atom_smoke_y],
    constrain=lambda sub: sub[lv_x] > sub[lv_y]
)
f2 = ParamF(
    MLNPotential(lambda atom: imp_op(atom[0], atom[1]), 1),
    nb=[atom_smoke_x, atom_cancer]
)

rel_g = RelationalGraph()
rel_g.atoms = (atom_friend, atom_smoke_x, atom_cancer)
rel_g.param_factors = (f1, f2)
rel_g.init_nb()

rel_g.data = data
g, rvs_table = rel_g.grounded_graph()

print(rvs_table)

# from osi.OneShot import OneShot
# import osi.utils as utils
# seed = 0
# utils.set_seed(seed=seed)
# K = 2
# T = 8
# osi = OneShot(g=g, K=K, T=T, seed=seed)
# res = osi.run(lr=0.2, its=200)
# for key, rv in sorted(rvs_table.items()):
#     if rv.value is None:  # only test non-evidence nodes
#         print(key, osi.map(rv))

from OneShot import OneShot

# np.random.seed(9)
osi = OneShot(g, num_mixtures=10, num_quadrature_points=8)

# osi.init_param()
# print(osi.gradient_w_tau())
# old_energy = osi.free_energy()
# print(osi.w)
# osi.w_tau += [0.01, 0]
# osi.w = osi.softmax(osi.w_tau)
# print(osi.w)
# new_energy = osi.free_energy()
# print(old_energy, new_energy, (new_energy-old_energy)/0.01)

osi.run(200, lr=5)

print(osi.free_energy())

for key, rv in sorted(rvs_table.items()):
    if rv.value is None:  # only test non-evidence nodes
        p = dict()
        for x in rv.domain.values:
            p[x] = osi.belief(x, rv)
        print(key, p)

# EPBP inference
from EPBPLogVersion import EPBP

bp = EPBP(g, n=50, proposal_approximation='simple')
bp.run(20, log_enable=False)

for key, rv in sorted(rvs_table.items()):
    if rv.value is None:  # only test non-evidence nodes
        p = dict()
        for x in rv.domain.values:
            p[x] = bp.belief(x, rv)
        print(key, p)

