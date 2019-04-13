from Graph import *

import tensorflow as tf
import numpy as np

dtype = 'float64'
from mixture_beliefs import hfactor_bfe_obj, dfactor_bfe_obj, drv_bfe_obj, drvs_bfe_obj, crvs_bfe_obj, \
    calc_cond_mixture_weights, drv_belief_map, crv_belief_map
import utils

utils.set_path()
from MLNPotential import MLNPotential


class OneShot:
    """
    A wrapper class that defines and optimizes the BFE over mixture beliefs, given the factor graph of a (hybrid) MRF;
    also maintains all the local variables/results created along the way.
    """

    def __init__(self, g, K, T, seed=None):
        """
        Define symbolic BFE and auxiliary objective expression to be optimized by tensorflow, given a factor graph.
        We'll use the one default tensorflow computation graph; to make sure we don't redefine it, everytime it'll
        be cleared/reset whenever a new instance of OneShot is created.
        :param g: a grounded graph corresponding to a plain old MLN; its factors must have .log_potential_fun callable
        on tf tensors
        :param K: num mixture comps
        :param T: num quad points
        """

        # pre-process graph to make things easier
        g.init_rv_indices()  # will create attributes like Vc, Vc_idx, etc.
        g.init_nb()  # in case neighbors aren't already sorted
        for f in g.factors_list:
            if f.log_potential_fun is None:
                assert isinstance(f.potential,
                                  MLNPotential), 'currently can only get log_potential_fun from MLNPotential'
                f.log_potential_fun = utils.get_log_potential_fun_from_MLNPotential(f.potential)

        tf.reset_default_graph()  # clear existing
        if seed is not None:  # note that seed that has been set prior to tf.reset_default_graph will be invalidated
            tf.set_random_seed(seed)  # thus we have to reseed after reset_default_graph
        tau = tf.Variable(tf.zeros(K, dtype=dtype), trainable=True, name='tau')  # mixture weights logits
        # tau = tf.Variable(tf.random_normal([K], dtype=dtype), trainable=True, name='tau')  # mixture weights logits
        w = tf.nn.softmax(tau, name='w')  # mixture weights

        bfe = aux_obj = 0
        if g.Nd > 0:
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
                Rho = [tf.Variable(tf.zeros([K, rv.dstates], dtype=dtype), trainable=True, name='Rho_%d' % i) for
                       (i, rv) in
                       enumerate(g.Vd)]  # dnode categorical prob logits
                Pi = [tf.nn.softmax(rho, name='Pi_%d' % i) for (i, rho) in enumerate(Rho)]  # convert to probs

            # assign symbolic belief vars to rvs
            for rv in g.Vd:
                i = g.Vd_idx[rv]  # ith disc node
                rv.belief_params_ = {'probs': Pi[i]}  # K x dstates[i] matrix

            # get discrete nodes' contributions to the objective
            if shared_dstates > 0:  # all discrete rvs have the same number of states
                delta_bfe, delta_aux_obj = drvs_bfe_obj(rvs=g.Vd, w=w, Pi=Pi)
                bfe += delta_bfe
                aux_obj += delta_aux_obj
            else:
                for rv in g.Vd:
                    delta_bfe, delta_aux_obj = drv_bfe_obj(rv, w)
                    bfe += delta_bfe
                    aux_obj += delta_aux_obj

        clip_op = tf.no_op()  # will be replaced with real clip op if Nc > 0
        if g.Nc > 0:  # assuming Gaussian
            # hard-coded for now
            Mu_bds = [-10, 10]
            Var_bds = [5e-3, 10]
            lVar_bds = np.log(Var_bds)

            Mu = tf.Variable(np.random.uniform(low=Mu_bds[0] / 2, high=Mu_bds[1] / 2, size=[g.Nc, K]),
                             dtype=dtype, trainable=True, name='Mu')

            # log of Var (sigma squared), for numeric stability
            lVar = tf.Variable(np.random.uniform(low=lVar_bds[0], high=lVar_bds[1], size=[g.Nc, K]),
                               dtype=dtype, trainable=True, name='lVar')
            Var = tf.exp(lVar)

            clip_op = tf.group(tf.assign(Mu, tf.clip_by_value(Mu, *Mu_bds)),
                               tf.assign(lVar, tf.clip_by_value(lVar, *lVar_bds)))

            for rv in g.Vc:
                i = g.Vc_idx[rv]  # ith cont node
                rv.belief_params_ = {
                    'mean': Mu[i], 'var': Var[i], 'var_inv': 1 / Var[i], 'mean_K1': tf.reshape(Mu[i], [K, 1]),
                    'var_K1': tf.reshape(Var[i], [K, 1]), 'var_inv_K1': tf.reshape(1 / Var[i], [K, 1])
                }

            # get continuous nodes' contribution to the objectives (assuming all Gaussian for now)
            delta_bfe, delta_aux_obj = crvs_bfe_obj(rvs=g.Vc, T=T, w=w, Mu=Mu, Var=Var)
            bfe += delta_bfe
            aux_obj += delta_aux_obj

        # get factors' contribution to the objectives
        for factor in g.factors_list:
            if factor.domain_type == 'd':
                # delta_bfe, delta_aux_obj = dfactor_bfe_obj(factor, w)
                delta_bfe, delta_aux_obj = hfactor_bfe_obj(factor, T, w)
            else:
                assert factor.domain_type in ('c', 'h')
                delta_bfe, delta_aux_obj = hfactor_bfe_obj(factor, T, w)
            bfe += delta_bfe
            aux_obj += delta_aux_obj

        self.__dict__.update(**locals())

    def run(self, its=100, lr=5e-2, tf_session=None, optimizer=None, trainable_params=None):
        """
        Launch tf training session and optimize the BFE, to get optimal mixture belief params.
        :param its:
        :param lr:
        :param tf_session:
        :param optimizer:
        :param trainable_params:
        :return:
        """
        g = self.g
        bfe, aux_obj, clip_op = self.bfe, self.aux_obj, self.clip_op
        w = self.w
        if not optimizer:
            optimizer = tf.train.AdamOptimizer(lr)
        if trainable_params is None:  # means all params are trainable
            trainable_params = tf.trainable_variables()

        grads_and_vars = optimizer.compute_gradients(aux_obj, var_list=trainable_params)
        # clip gradients if needed
        grads_update = optimizer.apply_gradients(grads_and_vars)

        vnames = ['g' + v.name.split(':')[0] for v in trainable_params]
        record = {n: [] for n in vnames + ['bfe']}

        if not tf_session:
            sess = tf.Session()  # session configs maybe
        else:
            sess = tf_session

        sess.run(tf.global_variables_initializer())
        for it in range(its):
            # bfe_, grads_and_vars_, _ = sess.run([bfe, grads_and_vars, grads_update])
            bfe_, grads_and_vars_ = sess.run([bfe, grads_and_vars])
            sess.run(grads_update)

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
            for key in sorted(it_record.keys()):
                print('%s: %g, ' % (key, it_record[key]), end='')
            print(sess.run(w))
            print()
            for key in record:
                record[key].append(it_record[key])

        params = {}
        params['w'] = sess.run(w)
        if g.Nd > 0:
            Rho, Pi = self.Rho, self.Pi

            shared_dstates = set(rv.dstates for rv in g.Vd)
            if len(shared_dstates) == 1:
                shared_dstates = shared_dstates.pop()
            else:
                shared_dstates = -1

            if shared_dstates > 0:  # all discrete rvs have the same number of states
                params['Rho'] = sess.run(Rho)
                params['Pi'] = sess.run(Pi)
            else:
                params['Rho'] = [sess.run(p) for p in Rho]
                params['Pi'] = [sess.run(p) for p in Pi]

            for rv in g.Vd:
                i = g.Vd_idx[rv]  # ith disc node
                rv.belief_params = {'probs': params['Pi'][i]}  # K x dstates[i] matrix

        if g.Nc > 0:
            Mu, Var = self.Mu, self.Var
            params['Mu'] = sess.run(Mu)
            params['Var'] = sess.run(Var)

            for rv in g.Vc:
                i = g.Vc_idx[rv]  # ith cont node
                rv.belief_params = {
                    'mu': params['Mu'][i], 'var': params['Var'][i],
                }

        self.params = params  # these are numeric values, not tf tensors
        result = {'record': record, **params}
        return result

    def map(self, rv):
        """
        Toy method for testing marginal MAP. Highly inefficient b/c cond_w recomputed every time
        :param rv:
        :return:
        """
        if rv.value is None:
            g = self.g
            params = self.params
            obs_rvs = [x for x in g.rvs if x.value is not None]
            X = np.array([x.value for x in obs_rvs])

            cond_w = calc_cond_mixture_weights(g, X, obs_rvs, params)
            if rv.domain_type == 'd':
                map_state, map_prob = drv_belief_map(cond_w, rv.belief_params['probs'])
                print(rv, map_state, map_prob)
                out = rv.values[map_state]
            else:
                assert rv.domain_type == 'c'
                mu, var = rv.belief_params['mu'], rv.belief_params['var']
                bds = (rv.values[0], rv.values[1])
                out = crv_belief_map(cond_w, mu, var, bds)
        else:
            out = rv.value
        return out
