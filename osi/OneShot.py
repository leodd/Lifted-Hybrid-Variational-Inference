from Graph import *

import tensorflow as tf

import numpy as np

dtype = 'float64'
from mixture_beliefs import hfactor_bfe_obj, dfactor_bfe_obj, drv_bfe_obj, drvs_bfe_obj, crvs_bfe_obj, \
    hfactors_bfe_obj, dfactors_bfe_obj, calc_cond_mixture_weights, drv_belief_map, crv_belief_map, marginal_map
import utils

utils.set_path()


class OneShot:
    """
    A wrapper class that defines and optimizes the BFE over mixture beliefs, given the factor graph of a (hybrid) MRF;
    also maintains all the local variables/results created along the way.
    """

    def __init__(self, g, K, T, seed=None, Var_bds=None):
        """
        Define symbolic BFE and auxiliary objective expression to be optimized by tensorflow, given a factor graph.
        We'll use the one default tensorflow computation graph; to make sure we don't redefine it, everytime it'll
        be cleared/reset whenever a new instance of OneShot is created.
        :param g: a grounded graph corresponding to a plain old MLN; its factors must have .log_potential_fun callable
        on tf tensors
        :param K: num mixture comps
        :param T: num quad points
        :param seed:
        :param Var_bds: [lb, ub] on the variance param of Gaussian rvs
        """
        # convert potentials to log_potential_funs (b/c typically caller only sets potentials instead of log pot)
        # utils.set_log_potential_funs(g.factors_list)
        assert all([callable(f.log_potential_fun) for f in g.factors]), 'factors must have valid log_potential_fun'

        # group factors together whose log potential functions have the same call signatures
        factors_with_unique_nb_domain_types, unique_nb_domain_types = \
            utils.get_unique_subsets(g.factors_list, lambda f: f.nb_domain_types)
        print('number of unique factor domain types =', len(unique_nb_domain_types))
        print(unique_nb_domain_types)

        g.init_rv_indices()  # will create attributes like Vc, Vc_idx, etc.
        # g.init_nb()  # caller should have always run this (or done sth similar) to ensure g is well defined!

        tf.reset_default_graph()  # clear existing
        if seed is not None:  # note that seed that has been set prior to tf.reset_default_graph will be invalidated
            tf.set_random_seed(seed)  # thus we have to reseed after reset_default_graph
        zeros_K = tf.zeros(K, dtype=dtype)
        tau = tf.Variable(zeros_K, trainable=True, name='tau')  # mixture weights logits
        # tau = tf.Variable(tf.random_normal([K], dtype=dtype), trainable=True, name='tau')  # mixture weights logits
        w = tf.nn.softmax(tau, name='w')  # mixture weights
        fix_mix_op = tf.assign(tau, zeros_K)  # op that resets mixing weights to uniform

        bfe = aux_obj = 0
        if g.Nd > 0:
            common_dstates = set(rv.dstates for rv in g.Vd)
            if len(common_dstates) == 1:
                common_dstates = common_dstates.pop()
            else:
                common_dstates = -1

            if common_dstates > 0:  # all discrete rvs have the same number of states
                # Rho = tf.Variable(tf.zeros([g.Nd, K, common_dstates], dtype=dtype), trainable=True,
                #                   name='Rho')  # dnode categorical prob logits
                Rho = tf.Variable(tf.random_normal([g.Nd, K, common_dstates], dtype=dtype), trainable=True,
                                  name='Rho')  # dnode categorical prob logits
                Pi = tf.nn.softmax(Rho, name='Pi')
            else:  # general case when each dnode can have different num states
                # Rho = [tf.Variable(tf.zeros([K, rv.dstates], dtype=dtype), trainable=True, name='Rho_%d' % i) for
                #        (i, rv) in enumerate(g.Vd)]  # dnode categorical prob logits
                Rho = [tf.Variable(tf.random_normal([K, rv.dstates], dtype=dtype), trainable=True, name='Rho_%d' % i)
                       for (i, rv) in enumerate(g.Vd)]  # dnode categorical prob logits
                Pi = [tf.nn.softmax(rho, name='Pi_%d' % i) for (i, rho) in enumerate(Rho)]  # convert to probs

            # assign symbolic belief vars to rvs
            for rv in g.Vd:
                i = g.Vd_idx[rv]  # ith disc node
                rv.belief_params_ = {'pi': Pi[i]}  # K x dstates[i] matrix

            # get discrete nodes' contributions to the objective
            if common_dstates > 0:  # all discrete rvs have the same number of states
                sharing_counts = [rv.sharing_count for rv in g.Vd]  # for lifting/param sharing; 1s if no lifting
                delta_bfe, delta_aux_obj = drvs_bfe_obj(rvs=g.Vd, w=w, Pi=Pi, rvs_counts=sharing_counts)
                bfe += delta_bfe
                aux_obj += delta_aux_obj
            else:
                for rv in g.Vd:
                    delta_bfe, delta_aux_obj = drv_bfe_obj(rv, w)
                    sharing_count = rv.sharing_count
                    bfe += sharing_count * delta_bfe
                    aux_obj += sharing_count * delta_aux_obj

        clip_op = tf.no_op()  # will be replaced with real clip op if Nc > 0
        if g.Nc > 0:  # assuming Gaussian
            if Var_bds is None:
                Var_bds = [5e-3, 10]  # currently shared by all cnodes

            Mu_bds = np.empty([2, g.Nc], dtype='float')
            for n, rv in enumerate(g.Vc):
                Mu_bds[:, n] = rv.values[0], rv.values[1]  # lb, ub
            Mu_bds = Mu_bds[:, :, None] + \
                     np.zeros([2, g.Nc, K], dtype='float')  # Mu_bds[0], Mu_bds[1] give lb, ub for Mu; same for all K
            Mu = np.random.uniform(low=Mu_bds[0], high=Mu_bds[1], size=[g.Nc, K])  # init numerical value
            init_grid = False
            if init_grid:  # try spreading initial means evenly on a grid within the Mu_bds box set
                I = int(K ** (1 / g.Nc))  # number of points per dimension; need to have I^{Nc} <= K
                slices = []
                for n, rv in enumerate(g.Vc):
                    lb, ub = rv.values[0], rv.values[1]
                    step = (ub - lb) / (I + 1)
                    slices.append(slice(lb + step, ub, step))  # no boundary points included
                grid = np.mgrid[slices]  # Nc x I x I x .. x I (Nc many Is)
                num_grid_points = int(I ** g.Nc)
                Mu[:, :num_grid_points] = np.reshape(grid, [g.Nc, num_grid_points])  # the other points r random uniform

            Mu = tf.Variable(Mu, dtype=dtype, trainable=True, name='Mu')

            # optimize the log of Var (sigma squared), for numeric stability
            lVar_bds = np.log(Var_bds)
            # lVar = tf.Variable(np.log(np.random.uniform(low=Var_bds[0], high=Var_bds[1], size=[g.Nc, K])),
            #                    dtype=dtype, trainable=True, name='lVar')
            lVar = tf.Variable(np.random.uniform(low=lVar_bds[0], high=lVar_bds[1], size=[g.Nc, K]),
                               dtype=dtype, trainable=True, name='lVar')
            Var = tf.exp(lVar)

            clip_op = tf.group(tf.assign(Mu, tf.clip_by_value(Mu, *Mu_bds)),
                               tf.assign(lVar, tf.clip_by_value(lVar, *lVar_bds)))

            for rv in g.Vc:
                i = g.Vc_idx[rv]  # ith cont node
                rv.belief_params_ = {
                    'mu': Mu[i], 'var': Var[i], 'var_inv': 1 / Var[i], 'mu_K1': tf.reshape(Mu[i], [K, 1]),
                    'var_K1': tf.reshape(Var[i], [K, 1]), 'var_inv_K1': tf.reshape(1 / Var[i], [K, 1])
                }

            # get continuous nodes' contribution to the objectives (assuming all Gaussian for now)
            sharing_counts = [rv.sharing_count for rv in g.Vc]  # for lifting/param sharing; 1s if no lifting
            delta_bfe, delta_aux_obj = crvs_bfe_obj(rvs=g.Vc, T=T, w=w, Mu=Mu, Var=Var, rvs_counts=sharing_counts)
            bfe += delta_bfe
            aux_obj += delta_aux_obj

        for factors in factors_with_unique_nb_domain_types:
            factor = factors[0]
            if factor.domain_type == 'd':
                delta_bfe, delta_aux_obj = dfactors_bfe_obj(factors, w)
            else:
                assert factor.domain_type in ('c', 'h')
                delta_bfe, delta_aux_obj = hfactors_bfe_obj(factors, T, w, dtype=dtype)
            bfe += delta_bfe
            aux_obj += delta_aux_obj

        self.__dict__.update(**locals())

    def run_setup(self, lr=5e-2, optimizer=None, trainable_params=None):
        """
        Set up gradient update ops, to facilitate caching (for repeated runs)
        :param lr:
        :param optimizer:
        :param trainable_params:
        :return:
        """
        if not (hasattr(self, 'grads_and_vars') and hasattr(self, 'grads_update')):
            if not optimizer:
                optimizer = tf.train.AdamOptimizer(lr)
            if trainable_params is None:  # means all params are trainable
                trainable_params = tf.trainable_variables()

            aux_obj = self.aux_obj
            grads_and_vars = optimizer.compute_gradients(aux_obj, var_list=trainable_params)
            grads_update = optimizer.apply_gradients(grads_and_vars)

            self.__dict__.update(**locals())
            return grads_and_vars, grads_update

    def run(self, its=100, lr=5e-2, tf_session=None, optimizer=None, trainable_params=None, grad_check=False,
            logging_itv=10, fix_mix_its=0):
        """
        Launch tf training session and optimize the BFE, to get optimal mixture belief params.
        :param its:
        :param lr:
        :param tf_session:
        :param optimizer:
        :param trainable_params:
        :param grad_check: if True, will only run one iteration, print gradients, then return
        :param logging_itv: log to console every this many its
        :param fix_mix_its: fix mixing weights to uniform for this many initial iterations; defaul = 0, i.e., mixing
        weights are never fixed; set to 'all' (or its) to keep the mixing weights always at uniform
        :return:
        """
        g = self.g
        bfe, aux_obj, clip_op = self.bfe, self.aux_obj, self.clip_op
        w = self.w
        if not optimizer:
            optimizer = tf.train.AdamOptimizer(lr)
        if trainable_params is None:  # means all params are trainable
            trainable_params = tf.trainable_variables()
        if fix_mix_its == 'all':
            fix_mix_its = its

        # grads_and_vars = optimizer.compute_gradients(aux_obj, var_list=trainable_params)
        # # clip gradients if needed
        # grads_update = optimizer.apply_gradients(grads_and_vars)

        # recompute gradient every time like above can be slow (e.g. for many repeated runs with random starts)
        self.run_setup(lr=lr, optimizer=optimizer, trainable_params=trainable_params)
        grads_and_vars, grads_update = self.grads_and_vars, self.grads_update

        if hasattr(self, 'sess'):  # will reuse most recent session to avoid session init overhead
            sess = self.sess
        elif tf_session is not None:
            sess = tf_session
        else:
            from config import tfconfig
            sess = tf.Session(config=tfconfig)  # config=None by default
        self.sess = sess  # will cache session from last run

        gvnames = ['g' + v.name.split(':')[0] for v in trainable_params]
        record = {n: [] for n in gvnames + ['bfe']}

        sess.run(tf.global_variables_initializer())  # always reinit
        for it in range(its):
            if not grad_check:
                # bfe_, grads_and_vars_ = sess.run([bfe, grads_and_vars])
                # sess.run(grads_update)
                # note that grads_and_vars_ below will contain updated vars because grads_update is also run
                bfe_, grads_and_vars_, _ = sess.run([bfe, grads_and_vars, grads_update])

            else:  # gradient check on the 0th iteration
                if it == 1:
                    break
                bfe_, grads_and_vars_ = sess.run([bfe, grads_and_vars])  # no need to run grads_update
                print('var vals')
                print(dict(zip([v.name.split(':')[0] for v in trainable_params], [gv[1] for gv in grads_and_vars_])))

                print(gvnames)

                print('numerical grads')
                num_grads = utils.calc_numerical_grad(trainable_params, bfe, sess, delta=1e-4)
                print(num_grads)

                print('quad symbolic grads')
                quad_grads = [gv[0] for gv in grads_and_vars_]
                print(quad_grads)

                print('relative errs')
                rel_errs = [np.abs(g1 - g2) / np.abs(g1) for (g1, g2) in zip(num_grads, quad_grads)]
                print(rel_errs)

                num_elems = np.sum([g.size for g in num_grads])
                print('avg grads absolute err:', np.sum([np.sum(err) for err in rel_errs]) / num_elems)

            sess.run(clip_op)  # does nothing if no cont nodes
            if it < fix_mix_its:
                sess.run(self.fix_mix_op)

            avg_grads = []
            for i in range(len(grads_and_vars_)):
                grad = grads_and_vars_[i][0]
                if not isinstance(grad, np.ndarray):
                    grad = grad.values  # somehow gMu is a IndexedSlicesValue
                avg_grads.append(np.mean(np.abs(grad)))
            it_record = dict(zip(gvnames, avg_grads))
            it_record['bfe'] = bfe_
            it_record['t'] = it
            if it % logging_itv == 0 or it + 1 == its:
                for key in sorted(it_record.keys()):
                    print('%s: %g, ' % (key, it_record[key]), end='')
                print(sess.run(w))
                if hasattr(self, 'lifting_reg'):
                    print(sess.run(self.lifting_reg))
                print()
            for key in record:
                record[key].append(it_record[key])

        params = {}
        params['w'] = sess.run(w)
        if g.Nd > 0:
            Rho, Pi = self.Rho, self.Pi

            common_dstates = set(rv.dstates for rv in g.Vd)
            if len(common_dstates) == 1:
                common_dstates = common_dstates.pop()
            else:
                common_dstates = -1

            if common_dstates > 0:  # all discrete rvs have the same number of states
                params['Rho'] = sess.run(Rho)
                params['Pi'] = sess.run(Pi)
            else:
                params['Rho'] = [sess.run(p) for p in Rho]
                params['Pi'] = [sess.run(p) for p in Pi]

            for rv in g.Vd:
                i = g.Vd_idx[rv]  # ith disc node
                rv.belief_params = {'pi': params['Pi'][i]}  # K x dstates[i] matrix

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

    def map(self, obs_rvs, query_rv):
        """
        Convenience method testing marginal MAP. Existing API assumes that obs_rvs carry .value attributes that indicate
        observed values. Can be inefficient b/c cond_w is recomputed every time
        :param obs_rvs: list
        :param query_rv:
        :return:
        """
        if query_rv.value is None:
            w = self.params['w']
            X = np.array([v.value for v in obs_rvs])
            out = marginal_map(X=X, obs_rvs=obs_rvs, query_rv=query_rv, w=w)
        else:
            out = query_rv.value
        return out


class LiftedOneShot(OneShot):
    def __init__(self, g, K, T, seed=None, Var_bds=None):
        """

        :param g: cluster graph, containing cluster nodes and factors; should be of type CompressedGraphSorted
        :param K:
        :param T:
        :param seed:
        :param Var_bds:
        """
        super().__init__(g, K, T, seed, Var_bds)
        self.unlifted_g = g.g  # original (uncompressed) graph

    def run(self, its=100, lr=5e-2, tf_session=None, optimizer=None, trainable_params=None, grad_check=False,
            logging_itv=10, fix_mix_its=0):
        res = super().run(its=its, lr=lr, tf_session=tf_session, optimizer=optimizer, trainable_params=trainable_params,
                          grad_check=grad_check, logging_itv=logging_itv, fix_mix_its=fix_mix_its)

        # post-processing to add resulting params to original graph; prediction tasks (e.g. marginal MAP) can then be
        # done on the original graph
        for rv in self.g.rvs_list:  # loop over cluster rvs (of type /CompressedGraphSorted.SuperRV) in the cluster graph
            for orig_rv in rv.rvs:
                orig_rv.belief_params = rv.belief_params

        return res


class LiftedOneShot2(OneShot):
    def __init__(self, g, cg, K, T, seed=None, Var_bds=None, lifting_reg_coef=0):
        """

        :param g
        :param cg: cluster graph, containing cluster nodes and factors; should be of type CompressedGraphSorted
        :param K:
        :param T:
        :param seed:
        :param Var_bds:
        """
        super().__init__(g, K, T, seed, Var_bds)
        self.cg = cg

        lifting_reg = 0
        for crv in cg.rvs:
            if crv.domain_type[0] == 'c':
                rvs_ind = [g.Vc_idx[rv] for rv in crv.rvs]
                rvs_comp_means = tf.gather(self.Mu, rvs_ind)
            elif crv.domain_type[0] == 'd':
                # rvs_ind = [g.Vd_idx[rv] for rv in crv.rvs]
                # rvs_comp_means = tf.gather(self.Pi, rvs_ind)
                rvs_comp_means = tf.reduce_sum(
                    tf.stack([rv.belief_params_['pi'] for rv in crv.rvs], axis=0) * crv.values, axis=-1)
            else:
                raise NotImplementedError
            rvs_means = tf.reduce_sum(rvs_comp_means * self.w, axis=1)
            lifting_reg += tf.nn.l2_loss(rvs_means - tf.reduce_mean(rvs_means))
        lifting_reg *= lifting_reg_coef

        self.unreg_bfe = self.bfe
        self.unreg_aux_obj = self.aux_obj
        self.bfe += lifting_reg
        self.aux_obj += lifting_reg
        self.lifting_reg = lifting_reg
