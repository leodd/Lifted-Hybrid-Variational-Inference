# classic example from MLN: An Interface Layer for AI; here we manually ground the network with 2 constants and run OSI
import tensorflow as tf
import numpy as np

seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)
dtype = 'float64'

import utils

utils.set_path()
from MLNPotential import and_op, or_op, neg_op, imp_op, bic_op, eq_op, MLNLogPotential
from Graph import Domain, Potential, RV, F, Graph

# KB = [
#     (lambda fxy, fyz, fxz: imp_op(and_op(fxy, fyz), fxz), 0.7),
#     (lambda sx, cx: imp_op(sx, cx), 1.5),
#     (lambda fxy, sx, sy: imp_op(and_op(fxy, sx), sy), 1.1),
#     # same as clausal form below
#     # (lambda fxy, fyz, fxz: or_op(or_op(neg_op(fxy), neg_op(fyz)), fxz), 0.7),
#     # (lambda sx, cx: or_op(neg_op(sx), cx), 1.5),
#     # (lambda fxy, sx, sy: or_op(or_op(neg_op(fxy), neg_op(sx)), sy), 1.1)
# ]
# use tuple args instead, to make Potential work
KB = [
    # (lambda vs: imp_op(and_op(vs[0], vs[1]), vs[2]), 0.7),  # 'friends of friends are friends; not used here
    (lambda vs: imp_op(vs[0], vs[1]), 1.5),  # 'smoking causes cancer'
    (lambda vs: imp_op(and_op(vs[0], vs[1]), vs[2]), 1.1),  # 'friend of smoker also smokes
    # same as clausal form below
    # (lambda fxy, fyz, fxz: or_op(or_op(neg_op(fxy), neg_op(fyz)), fxz), 0.7),
    # (lambda sx, cx: or_op(neg_op(sx), cx), 1.5),
    # (lambda fxy, sx, sy: or_op(or_op(neg_op(fxy), neg_op(sx)), sy), 1.1)
]

# log_potential_funs = [lambda vs: f(vs) * w for (f, w) in KB]    # subtle bug due to linger ref to temp f?
log_potential_funs = [utils.weighted_feature_fun(f, w) for (f, w) in KB]

# create the MN (ground MLN for constants={A, B})
N = 8
nodes = np.arange(N)  # node ids; 0:F(A,A), 1:F(A,B), 2:S(A), 3:C(A), 4:F(B,B), 5:F(B,A), 6:S(B), 7:C(B)
rvs = [RV(domain=Domain(values=np.array([0, 1]), continuous=False), id=n) for n in nodes]
factors = [
    F(log_potential_fun=log_potential_funs[1], nb=[rvs[i] for i in [0, 2, 2]]),
    F(log_potential_fun=log_potential_funs[0], nb=[rvs[i] for i in [2, 3]]),
    F(log_potential_fun=log_potential_funs[1], nb=[rvs[i] for i in [1, 2, 6]]),
    F(log_potential_fun=log_potential_funs[1], nb=[rvs[i] for i in [5, 6, 2]]),
    F(log_potential_fun=log_potential_funs[0], nb=[rvs[i] for i in [6, 7]]),
    F(log_potential_fun=log_potential_funs[1], nb=[rvs[i] for i in [4, 6, 6]]),
]
for i, f in enumerate(factors):
    f.id = i

g = Graph()
g.rvs = rvs
g.factors = factors
g.init_nb()
g.init_rv_indices()

# from FactorGraph import FactorGraph

# define BFE


K = 3
shared_dstates = set(rv.dstates for rv in g.Vd)
if len(shared_dstates) == 1:
    shared_dstates = shared_dstates.pop()
else:
    shared_dstates = -1

if shared_dstates > 0:  # all discrete rvs have the same number of states
    # Rho = tf.Variable(tf.zeros([g.Nd, K, shared_dstates], dtype=dtype), trainable=True,
    #                   name='Rho')  # dnode categorical prob logits
    Rho = tf.Variable(tf.random_normal([g.Nd, K, shared_dstates], dtype=dtype), trainable=True,
                      name='Rho')  # dnode categorical prob logits
    Pi = tf.nn.softmax(Rho, name='Pi')
else:  # general case when each dnode can have different num states
    Rho = [tf.Variable(tf.zeros([K, rv.dstates], dtype=dtype), trainable=True, name='Rho_%d' % i) for (i, rv) in
           enumerate(g.Vd)]  # dnode categorical prob logits
    Pi = [tf.nn.softmax(rho, name='Pi_%d' % i) for (i, rho) in enumerate(Rho)]  # convert to probs

# assign symbolic belief vars to rvs
for rv in g.Vd:
    i = g.Vd_idx[rv]  # ith disc node
    rv.belief_params_ = {'probs': Pi[i]}  # K x dstates[i] matrix

# for rv in g.Vc:
#     i = g.Vd_idx[rv]  # ith disc node
#     rv.belief_params_ = {
#         'mean': Mu[i], 'var': Var[i], 'var_inv': 1 / Var[i], 'mean_K1': tf.reshape(Mu[i], [K, 1]),
#         'var_K1': tf.reshape(Var[i], [K, 1]), 'var_inv_K1': tf.reshape(1 / Var[i], [K, 1])
#     }

tau = tf.Variable(tf.zeros(K, dtype=dtype), trainable=True, name='tau')  # mixture weights logits
# tau = tf.Variable(tf.random_normal([K], dtype=dtype), trainable=True, name='tau')  # mixture weights logits
w = tf.nn.softmax(tau, name='w')  # mixture weights

from mixture_beliefs import dfactor_belief_bfe, drv_belief_bfe

bfe = aux_obj = 0
for factor in g.factors:
    bfe_, aux_obj_ = dfactor_belief_bfe(factor, w)
    bfe += bfe_
    aux_obj += aux_obj_

if shared_dstates > 0:  # all discrete rvs have the same number of states
    w_1K1 = tf.reshape(w, [1, K, 1])
    belief = tf.reduce_sum(w_1K1 * Pi, axis=1)  # Nd x shared_dstates
    log_belief = tf.log(belief)
    num_nbrs = np.array([len(rv.nb) for rv in g.Vd])[:, np.newaxis]  # Nd x 1
    F = (1 - num_nbrs) * log_belief  # Nd x shared_dstates
    prod = tf.stop_gradient(belief * F)  # stop_gradient is needed for aux_obj
    bfe += tf.reduce_sum(prod)  # we really mean the free energy, which is to be minimized
    aux_obj += tf.reduce_sum(prod * log_belief)  # only differentiate w.r.t log_belief
else:
    for rv in g.Vd:
        bfe_, aux_obj_ = drv_belief_bfe(rv, w)
        bfe += bfe_
        aux_obj += aux_obj_

print('set up training')
# train
lr = 0.02
its = 200
# optimizer = tf.train.GradientDescentOptimizer(lr)
optimizer = tf.train.AdamOptimizer(lr)
# optimizer = tf.train.MomentumOptimizer(lr, 0.9)
inference_params = tf.trainable_variables()
print('trainable params', inference_params)
grads_and_vars = optimizer.compute_gradients(aux_obj, var_list=inference_params)
# clip gradients if needed
grads_update = optimizer.apply_gradients(grads_and_vars)

vnames = ['g' + v.name.split(':')[0] for v in inference_params]
record = {n: [] for n in vnames + ['bfe']}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for it in range(its):
        bfe_, grads_and_vars_, _ = sess.run([bfe, grads_and_vars, grads_update])
        # sess.run(clip_op)  # does nothing if no cont nodes
        avg_grads = []
        for i in range(len(grads_and_vars_)):
            grad = grads_and_vars_[i][0]
            if not isinstance(grad, np.ndarray):
                grad = grad.values  # somehow gMu is a IndexedSlicesValue
            avg_grads.append(np.mean(np.abs(grad)))
        it_record = dict(zip(vnames, avg_grads))
        it_record['bfe'] = bfe_
        it_record['t'] = it
        for key in sorted(it_record.keys()):
            print('%s: %g, ' % (key, it_record[key]), end='')
        print(sess.run(w))
        print()
        for key in record:
            record[key].append(it_record[key])
    Pi = sess.run(Pi)
    w = sess.run(w)
print(np.sum(w[None, :, None] * Pi, axis=1))

import matplotlib.pyplot as plt

plt.figure()
for key in record:
    plt.plot(record[key], label=key)
plt.legend(loc='best')
save_name = 'smoke_example'
plt.savefig('%s.png' % save_name)
