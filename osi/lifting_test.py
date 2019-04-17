import tensorflow as tf
import numpy as np
import utils

seed = 0
utils.set_seed(seed=seed)
utils.set_path()

from CompressedGraphSorted import *
from Graph import Domain, RV, F, Graph

from numpy import Inf
from Potential import TablePotential

p1 = TablePotential({
    (True, True): 4,
    (True, False): 1,
    (False, True): 1,
    (False, False): 3
}, symmetric=True)


# p1 = TablePotential({
#     (True, True): 4,
#     (True, False): 1,
#     (False, True): 0.5,
#     (False, False): 3
# }, symmetric=False)  # in this example there'll be no compression if symmetric=False, so lifing makes no difference


# currently extract the log_potential_fun manually
def log_potential_fun(xs):
    x0, x1 = xs[0], xs[1]
    return 4 * (x0 == 1) * (x1 == 1) + 1 * (x0 == 1) * (x1 == 0) + 1 * (x0 == 0) * (x1 == 1) + 3 * (x0 == 0) * (
        x1 == 0)
    # return 4 * x0 * x1 + 3 * (1 - x0) * (1 - x1) + 1 * (x0 != x1)  # being clever in this case


# from MLNPotential import MLNPotential
# p1 = MLNPotential(formula=lambda xs: xs[0] * xs[1], w=1)

# d = Domain([True, False])
d = Domain([0, 1], continuous=False)

N = 4
# rvs = []
# for i in range(N):
#     rvs.append(RV(d))
rvs = [RV(d) for n in range(N)]

# fs = []
# for i in range(N - 1):
#     fs.append(F(potential=p1, log_potential_fun=log_potential_fun, nb=(rvs[i], rvs[i + 1])))
fs = [F(potential=p1, log_potential_fun=log_potential_fun, nb=(rvs[i], rvs[i + 1])) for i in range(N - 1)]
g = Graph()
g.rvs = rvs
g.factors = fs
g.init_nb()

cg = CompressedGraphSorted(g)
cg.run()

print(len(g.rvs), len(g.factors))
print(len(cg.rvs), len(cg.factors))

K = 2
T = 0
lr = 1e-1
its = 300
from OneShot import OneShot, LiftedOneShot

print('with lifting')
osi = LiftedOneShot(g=cg, K=K, T=T, seed=seed)
res = osi.run(lr=lr, its=its)
w = res['w']
w_row = w[None, :]
for rv in sorted(cg.rvs):
    print(rv)
    print(rv.rvs)
    params = rv.belief_params
    print(params)
    if 'pi' in params:
        print(w @ params['pi'])
    if 'mu' in params:
        print(w @ params['mu'])

import matplotlib.pyplot as plt

record = res['record']
# for key in record:
#     plt.plot(record[key], label=key)
plt.plot(record['bfe'], label='bfe (with lifting)')

print('no lifting')

osi = OneShot(g=g, K=K, T=T, seed=seed)
res = osi.run(lr=lr, its=its)
w = res['w']
w_row = w[None, :]
for rv in sorted(g.rvs):
    print(rv)
    params = rv.belief_params
    print(params)
    if 'pi' in params:
        print(w @ params['pi'])
    if 'mu' in params:
        print(w @ params['mu'])

import matplotlib.pyplot as plt

record = res['record']
# for key in record:
#     plt.plot(record[key], label=key)
plt.plot(record['bfe'], label='bfe (no lifting)')

plt.legend(loc='best')

# plt.show()
save_name = __file__.split('.py')[0]
plt.savefig('%s.png' % save_name)
