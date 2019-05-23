from RelationalGraph import *
from Potential import GaussianPotential
from GaBP import GaBP


instance = [
    'Joey',
    'Rachel',
    'Tim',
]

data = {
    ('X', 'Joey'): 20,
    ('X', 'Tim'): 40,
    # ('X', 'Rachel'): -20,
    # ('Y', 'Tim'): 0,
    # ('Y', 'Joey'): 1,
    # ('Y', 'Rachel'): 0,
}

d = Domain((-50, 50), continuous=True, integral_points=linspace(-50, 50, 30))

lv_x = LV(instance)
lv_y = LV(instance)

atom_X = Atom(domain=d, logical_variables=[lv_x], name='X')
atom_Y = Atom(domain=d, logical_variables=[lv_y], name='Y')

f1 = ParamF(GaussianPotential([0., 0.], [[10., -7.], [-7., 10.]]), [atom_X, atom_Y])

rel_g = RelationalGraph()
rel_g.atoms = [atom_X, atom_Y]
rel_g.param_factors = [f1]
rel_g.init_nb()

rel_g.data = data
g, rvs_table = rel_g.grounded_graph()

print(rvs_table)

from LiftedVarInference import VarInference

# np.random.seed(9)
vi = VarInference(g, num_mixtures=3, num_quadrature_points=5)

vi.run(500, lr=0.1)

print(vi.free_energy())

for key, rv in sorted(rvs_table.items()):
    if rv.value is None:  # only test non-evidence nodes
        # p = dict()
        # for x in rv.domain.values:
        #     p[x] = osi.belief(x, rv)
        # print(key, p)
        print(key, vi.map(rv))

# EPBP inference
from EPBPLogVersion import EPBP

bp = EPBP(g, n=20, proposal_approximation='simple')
bp.run(20, log_enable=False)

for key, rv in sorted(rvs_table.items()):
    if rv.value is None:  # only test non-evidence nodes
        # p = dict()
        # for x in rv.domain.values:
        #     p[x] = bp.belief(x, rv)
        # print(key, p)
        print(key, bp.map(rv))

# GaBP inference
from GaBP import GaBP

bp = GaBP(g)
bp.run(20, log_enable=False)

for key, rv in sorted(rvs_table.items()):
    if rv.value is None:  # only test non-evidence nodes
        # p = dict()
        # for x in rv.domain.values:
        #     p[x] = bp.belief(x, rv)
        # print(key, p)
        print(key, bp.map(rv))

