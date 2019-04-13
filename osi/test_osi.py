import tensorflow as tf
import numpy as np

seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)
dtype = 'float64'

import utils

utils.set_path()
from MLNPotential import and_op, or_op, neg_op, imp_op, bic_op, eq_op
from Graph import Domain, Potential, RV, F, Graph

test = 'd'  # disc
# test = 'd2'  # disc
# test = 'c'  # cont
# test = 'h'  # hybrid

N = 2
nodes = np.arange(N)

if test == 'd':
    rvs = [RV(domain=Domain(values=np.array([0, 1]), continuous=False)) for n in nodes]


    def lphi_0(xs):
        x = xs[0]
        return -0.2 * (xs[0] == 0) + 1 * (x == 1)


    def lphi_1(xs):
        x = xs[0]
        return 0.1 * (x == 0) + 0.2 * (x == 1)


    def lpsi(xs):
        x0 = xs[0]
        x1 = xs[1]
        return -1 * (x0 == 0) * (x1 == 0) + 1 * (x0 == 0) * (x1 == 1) + \
               2 * (x0 == 1) * (x1 == 0) + -1 * (x0 == 1) * (x1 == 1)


    factors = [
        F(log_potential_fun=lphi_0, nb=[rvs[0]]),
        F(log_potential_fun=lphi_1, nb=[rvs[1]]),
        F(log_potential_fun=lpsi, nb=[rvs[0], rvs[1]]),
    ]
    Z = 0
    for s0 in rvs[0].values:
        for s1 in rvs[1].values:
            Z += np.exp(lphi_0([s0]) + lphi_1([s1]) + lpsi([s0, s1]))
    print('true -log Z =', -np.log(Z))

elif test == 'd2':
    rvs = [RV(domain=Domain(values=np.array([0, 1]), continuous=False)),
           RV(domain=Domain(values=np.array([0, 1, 2]), continuous=False))]


    def lphi_0(xs):
        x = xs[0]
        return -0.2 * (xs[0] == 0) + 1 * (x == 1)


    def lphi_1(xs):
        x = xs[0]
        return 0.1 * (x == 0) + 0.2 * (x == 1) + 0.3 * (x == 2)


    def lpsi(xs):
        x0 = xs[0]
        x1 = xs[1]
        return -1 * (x0 == 0) * (x1 == 0) + 1 * (x0 == 0) * (x1 == 1) + 3 * (x0 == 0) * (x1 == 2) + \
               2 * (x0 == 1) * (x1 == 0) + -1 * (x0 == 1) * (x1 == 1) + -3 * (x0 == 1) * (x1 == 2)


    factors = [
        F(log_potential_fun=lphi_0, nb=[rvs[0]]),
        F(log_potential_fun=lphi_1, nb=[rvs[1]]),
        F(log_potential_fun=lpsi, nb=[rvs[0], rvs[1]]),
    ]

    Z = 0
    for s0 in rvs[0].values:
        for s1 in rvs[1].values:
            Z += np.exp(lphi_0([s0]) + lphi_1([s1]) + lpsi([s0, s1]))
    print('true -log Z =', -np.log(Z))

g = Graph()
g.rvs = rvs
g.factors = factors
g.init_nb()
g.init_rv_indices()

from OneShot import OneShot

K = 4
T = 8
lr = 1e-1
osi = OneShot(g=g, K=K, T=T, seed=seed)
res = osi.run(lr=lr, its=200)
w = res['w']
w_row = w[None, :]
for rv in sorted(g.rvs):
    params = rv.belief_params
    print(params)
    if 'probs' in params:
        print(w @ params['probs'])
    if 'mu' in params:
        print(w @ params['mu'])

import matplotlib.pyplot as plt

record = res['record']
for key in record:
    plt.plot(record[key], label=key)
plt.legend(loc='best')

plt.show()
