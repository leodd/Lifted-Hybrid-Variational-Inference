from Potential import TablePotential
from Graph import *


domain = Domain((0, 1))

p1 = TablePotential({
    (0, 0): 1,
    (0, 1): 0.1,
    (1, 0): 0.1,
    (1, 1): 1
})

# A-B-C-D-E

A = RV(domain)
B = RV(domain, 1)
C = RV(domain)
D = RV(domain)
E = RV(domain)

f_AB = F(p1, [A, B])
f_ED = F(p1, [E, D])
f_BC = F(p1, [B, C])
f_DC = F(p1, [D, C])

g = Graph()
g.rvs = [A, B, C, D, E]
g.factors = [f_AB, f_BC, f_DC, f_ED]
g.init_nb()

from OneShot import OneShot

# np.random.seed(9)
osi = OneShot(g, num_mixtures=10, num_quadrature_points=8)

osi.run(200, lr=2)

print(osi.free_energy())

for rv in g.rvs:
    if rv.value is None:  # only test non-evidence nodes
        p = dict()
        for x in rv.domain.values:
            p[x] = osi.belief(x, rv)
        print(rv, p)

# EPBP inference
from EPBPLogVersion import EPBP

bp = EPBP(g, n=50, proposal_approximation='simple')
bp.run(20, log_enable=False)

for rv in g.rvs:
    if rv.value is None:  # only test non-evidence nodes
        p = dict()
        for x in rv.domain.values:
            p[x] = bp.belief(x, rv)
        print(rv, p)
