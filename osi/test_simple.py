import tensorflow as tf
import numpy as np

seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)
dtype = 'float64'

import utils

utils.set_path()
from Graph import Domain, RV, F, Graph

# test = 'd'  # disc
# test = 'd2'  # disc
# test = 'c'  # cont
test = 'h'  # hybrid

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

elif test == 'c':
    B = 10
    rvs = [RV(domain=Domain(values=[-B, B], continuous=True)),
           RV(domain=Domain(values=[-B, B], continuous=True))]


    def lphi_0(xs):
        x = xs[0]
        return -2 + 1 * x + 0.1 * x ** 2


    def lphi_1(xs):
        x = xs[0]
        return 0.3 + 0.2 * x + -1 * x ** 2


    def lpsi(xs):
        x0 = xs[0]
        x1 = xs[1]
        return 0.1 + 0.2 * x1 + 0.3 * x1 ** 2 + \
               0.4 * x0 + 0.5 * x0 * x1 + 0.6 * x0 * x1 ** 2 + \
               -0.7 * x0 ** 2 + -0.8 * x0 ** 2 * x1 + -0.9 * x0 ** 2 * x1 ** 2


    factors = [
        F(log_potential_fun=lphi_0, nb=[rvs[0]]),
        F(log_potential_fun=lphi_1, nb=[rvs[1]]),
        F(log_potential_fun=lpsi, nb=[rvs[0], rvs[1]]),
    ]

    from scipy.integrate import dblquad

    Z, err = dblquad(lambda x1, x0: np.exp(lphi_0([x0]) + lphi_1([x1]) + lpsi([x0, x1])),
                     -B, B, lambda x: -B, lambda x: B)
    print('true -log Z =', -np.log(Z))

elif test == 'h':
    B = 10
    rvs = [RV(domain=Domain(values=[0, 1], continuous=False)),
           RV(domain=Domain(values=[-B, B], continuous=True))]


    def lphi_0(xs):
        x = xs[0]
        return -0.2 * (xs[0] == 0) + 1 * (x == 1)


    def lphi_1(xs):
        x = xs[0]
        return -2 + 1 * x + 0.1 * x ** 2


    def lpsi(xs):
        x0 = xs[0]
        x1 = xs[1]
        return (x0 == 0) * (0.1 + 0.2 * x1 - 0.3 * x1 ** 2) + \
               (x0 == 1) * (0.4 + 0.5 * x1 - 0.6 * x1 ** 2)


    factors = [
        F(log_potential_fun=lphi_0, nb=[rvs[0]]),
        F(log_potential_fun=lphi_1, nb=[rvs[1]]),
        F(log_potential_fun=lpsi, nb=[rvs[0], rvs[1]]),
    ]

    from scipy.integrate import quad

    Z = 0
    for x0 in rvs[0].values:
        Z_, err = quad(lambda x1: np.exp(lphi_0([x0]) + lphi_1([x1]) + lpsi([x0, x1])), -B, B)
        Z += Z_
    print('true -log Z =', -np.log(Z))

g = Graph()
g.rvs = rvs
g.factors = factors
g.init_nb()
g.init_rv_indices()

from OneShot import OneShot

grad_check = True
if not grad_check:
    K = 4
    T = 30
    lr = 5e-1
    osi = OneShot(g=g, K=K, T=T, seed=seed)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    res = osi.run(lr=lr, optimizer=optimizer, its=1000)
    w = res['w']
    w_row = w[None, :]
    for rv in sorted(g.rvs):
        params = rv.belief_params
        print(params)
        if 'pi' in params:
            print(w @ params['pi'])
        if 'mu' in params:
            print(w @ params['mu'])

    import matplotlib.pyplot as plt

    record = res['record']
    for key in record:
        plt.plot(record[key], label=key)
    plt.legend(loc='best')

    plt.show()

else:
    K = 4
    its = 1
    lr = 5e-1
    for T in [10, 20, 50, 100, 200]:
        print('grad check, T =', T)
        utils.set_seed(seed)
        osi = OneShot(g=g, K=K, T=T, seed=seed)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        res = osi.run(lr=lr, optimizer=optimizer, its=its, grad_check=True)
        print()
