import tensorflow as tf
import numpy as np
# from Graph import Graph
import networkx as nx
from utils import build_grid_graph, update_nodes_info

# multithreading
from configs import tfconfig

seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)
dtype = 'float64'

img = np.load('exsquareNoisy.npy')
img_dim = img.shape[0]
# currently only model the pixels (continuous nodes only)
g, coords_to_ids = build_grid_graph(img_dim, img_dim)
update_nodes_info(g, [], [])
Nc = g.Nc

assert Nc == img.size
obs = np.empty(Nc)  # obs[n] gives noisy observation (y_n) for the nth node (x_n)
for c, n in coords_to_ids.items():
    obs[n] = img[c]


def node_log_pot(n, x):
    sig2 = 0.01
    obs_broadcast_shape = [1] * len(x.shape)
    obs_broadcast_shape[0] = x.shape[0]
    obs_ = obs.reshape(obs_broadcast_shape)
    out = - 0.5 * np.log(2 * np.pi * sig2) - 0.5 * (x - obs_) ** 2 / sig2
    return out


def edge_log_pot(n1, n2, x1, x2):
    range = 0.2
    trunc_const = - np.log(0.06) - range / 0.03
    diff = tf.abs(x1 - x2)
    leq = tf.cast(diff <= range, dtype)
    gt = 1 - leq
    out = (-np.log(0.06) - diff / 0.03) * leq + trunc_const * gt
    return out


K = 2
T = 12  # num quad points

Mu_bds = [-0.1, 1.1]
Sigs_bds = [1e-3, 10]
lSigs_bds = np.log(Sigs_bds)

# build computation graph / objective
from mixture_beliefs.gaussian import get_quad_bfe

# declare belief params
tau = tf.Variable(tf.zeros(K, dtype=dtype), trainable=True, name='tau')
w = tf.nn.softmax(tau)
Mu = tf.Variable(tf.random_uniform([g.Nc, K], minval=Mu_bds[0], maxval=Mu_bds[1], dtype=dtype), dtype=dtype,
                 trainable=True, name='Mu')
# log of Sigma squared, for numeric stability
lSigs = tf.Variable(tf.random_uniform([g.Nc, K], minval=lSigs_bds[0], maxval=lSigs_bds[1], dtype=dtype),
                    dtype=dtype, trainable=True, name='lSigs')
Sigs = tf.exp(lSigs)

clip_op = tf.group(tf.assign(Mu, tf.clip_by_value(Mu, *Mu_bds)),
                   tf.assign(lSigs, tf.clip_by_value(lSigs, *lSigs_bds)))
# clip_op = tf.no_op()
bfe, aux_obj = get_quad_bfe(g, w, Mu, Sigs, T, node_log_pot, edge_log_pot)

print('set up training')
# train
lr = 0.08
its = 1000
# optimizer = tf.train.GradientDescentOptimizer(lr)
optimizer = tf.train.AdamOptimizer(lr)
# optimizer = tf.train.MomentumOptimizer(lr, 0.9)
inference_params = tf.trainable_variables()
print('trainable params', inference_params)
grads_and_vars = optimizer.compute_gradients(-aux_obj, var_list=inference_params)
# clip gradients if needed
grads_update = optimizer.apply_gradients(grads_and_vars)

vnames = ['g' + v.name.split(':')[0] for v in inference_params]
record = {n: [] for n in vnames + ['bfe']}
with tf.Session(config=tfconfig) as sess:
    sess.run(tf.global_variables_initializer())

    for it in range(its):
        bfe_, grads_and_vars_, _ = sess.run([bfe, grads_and_vars, grads_update])
        sess.run(clip_op)  # does nothing if no cont nodes
        avg_grads = []
        for i in range(len(grads_and_vars_)):
            grad = grads_and_vars_[i][0]
            if not isinstance(grad, np.ndarray):
                grad = grad.values  # somehow gMu is a IndexedSlicesValue
            avg_grads.append(np.mean(np.abs(grad)))
        it_record = dict(zip(vnames, avg_grads))
        it_record['bfe'] = bfe_
        it_record['t'] = it
        for k in sorted(it_record.keys()):
            print('%s: %g, ' % (k, it_record[k]), end='')
        print(sess.run(w))
        print()
        for key in record:
            record[key].append(it_record[key])

    Mu = sess.run(Mu)
    w = sess.run(w)

import matplotlib.pyplot as plt

plt.figure()
for key in record:
    plt.plot(record[key], label=key)
plt.legend(loc='best')
save_name = 'denoising_bfe'
plt.savefig('%s.png' % save_name)

mix_mean = Mu @ w
plt.figure()
plt.imshow(np.reshape(mix_mean, [img_dim, img_dim]), cmap='gray', interpolation='none')
save_name = 'denoised_bfe'
plt.savefig('%s.png' % save_name)
