import utils

utils.set_path(('..',))
from RelationalGraph import *
from MLNPotential import *
from Potential import QuadraticPotential, TablePotential, HybridQuadraticPotential
from EPBPLogVersion import EPBP
from GaBP import GaBP
from OneShot import OneShot, LiftedOneShot
from NPVI import NPVI, LiftedNPVI
from CompressedGraphSorted import CompressedGraphSorted
import numpy as np
import time
from copy import copy

seed = 0
utils.set_seed(seed)

# KF stuff
from KalmanFilter import KalmanFilter
from Graph import *
import scipy.io

cluster_mat = scipy.io.loadmat('Data/RKF/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/RKF/well_t.mat')['well_t']
ans = scipy.io.loadmat('Data/RKF/LRKF_tree.mat')['res']
param = scipy.io.loadmat('Data/RKF/LRKF_tree.mat')['param']
print(well_t.shape)

well_t = well_t[:, 199:]
well_t[well_t[:, 0] == 5000, 0] = 0
well_t[well_t == 5000] = 1
t = 20

cluster_id = [1]

rvs_id = []
for i in cluster_id:
    rvs_id.append(np.where(cluster_mat[:, i] == 1)[0])

rvs_id = np.concatenate(rvs_id, axis=None)
n_sum = len(rvs_id)
data = well_t[rvs_id, :t]

domain = Domain((-4, 4), continuous=True, integral_points=np.linspace(-4, 4, 30))

num_tests = param.shape[1]

from KLDivergence import kl_continuous_logpdf

record_fields = ['cpu_time',
                 'wall_time',
                 'obj',  # this is BFE/-ELBO for variational methods, -logZ for exact baseline
                 'mmap_err',  # |argmax p(xi) - argmax q(xi)|, avg over all nodes i
                 'kl_err',  # kl(p(xi)||q(xi)), avg over all nodes i
                 ]
# algo_names = ['baseline', 'EPBP', 'OSI', 'LOSI']
algo_names = ['baseline', 'GaBP', 'NPVI', 'LNPVI', 'OSI', 'LOSI']
# algo_names = ['baseline', 'GaBP', 'NPVI', 'LNPVI', 'OSI', 'LOSI']
# algo_names = ['baseline', 'EPBP']
# algo_names = ['EPBP']
# assert algo_names[0] == 'baseline'
# for each algorithm, we keep a record, which is a dict mapping a record_field to a list (which will eventually be
# averaged over)
records = {algo_name: {record_field: [] for record_field in record_fields} for algo_name in algo_names}

plot = True
print(f'########total {num_tests} tests########')
for test_num in range(num_tests):
    test_seed = seed + test_num

    # regenerate/reload evidence
    kmf = KalmanFilter(domain,
                       np.eye(n_sum) * param[2, test_num],
                       param[0, test_num],
                       np.eye(n_sum),
                       param[1, test_num])

    g, rvs_table = kmf.grounded_graph(t, data)
    print('num rvs in g', len(g.rvs))
    print('num factors in g', len(g.factors))

    # query nodes
    query_rvs = rvs_table[t - 1]

    g_rv_nbs = [copy(rv.nb) for rv in g.rvs_list]  # keep a copy of rv neighbors in the original graph

    print('number of rvs', len(g.rvs))
    print('num drvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'd']))
    print('num crvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'c']))
    print('number of factors', len(g.factors))
    print('number of evidence', len(data))

    obs_rvs = [v for v in g.rvs if v.value is not None]
    evidence = {rv: rv.value for rv in obs_rvs}
    cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs, evidence)  # this will also condition log_potential_funs

    print('cond number of rvs', len(cond_g.rvs))
    print('cond num drvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'd']))
    print('cond num crvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'c']))

    all_margs = {algo_name: [None] * len(query_rvs) for algo_name in algo_names}  # for plotting convenience

    baseline = 'exact'
    # baseline = 'gibbs'
    for a, algo_name in enumerate(algo_names):
        print('####')
        print('test_num', test_num)
        print('running', algo_name)
        np.random.seed(test_seed + a)

        # temp storage
        mmap = np.zeros(len(query_rvs)) - 123
        margs = [None] * len(query_rvs)
        marg_kls = np.zeros(len(query_rvs)) - 123
        obj = -1
        cpu_time = wall_time = -1  # don't care

        if algo_name == 'baseline':
            # guaranteed exact baseline by solving linear equations (marginal means = marginal modes in Gaussians)
            quadratic_params, rvs_idx = utils.get_quadratic_params_from_factor_graph(cond_g.factors, cond_g.rvs_list)
            mu, Sig = utils.get_gaussian_mean_params_from_quadratic_params(A=quadratic_params[0], b=quadratic_params[1],
                                                                           mu_only=False)
            Nc = len(mu)
            A, b, c = quadratic_params
            (sign, logdet) = np.linalg.slogdet(Sig)
            assert sign == 1, 'cov mat must be PD'
            log_joint_quadratic_integral = Nc / 2 * np.log(2 * np.pi) + 0.5 * logdet + 0.5 * np.dot(mu, b)
            log_joint_quadratic_integral += c  # log integral of all cont & hybrid factors (with disc nodes substituted in)
            obj = -log_joint_quadratic_integral  # -logZ
            print('true obj', obj)

            for i, rv in enumerate(query_rvs):
                rv_idx = rvs_idx[rv]
                mmap[i] = mu[rv_idx]
                margs[i] = utils.curry_normal_logpdf(mu[rv_idx], Sig[rv_idx, rv_idx])

            # save baseline
            baseline_mmap = mmap
            baseline_margs = margs

        elif algo_name == 'GaBP':
            bp = GaBP(g)
            start_time = time.process_time()
            start_wall_time = time.time()
            bp.run(15, log_enable=False)
            cpu_time = time.process_time() - start_time
            wall_time = time.time() - start_wall_time
            for i, rv in enumerate(query_rvs):
                mmap[i] = bp.map(rv)
                belief_params = bp.get_belief_params(rv)
                margs[i] = utils.curry_normal_logpdf(*belief_params)

        elif algo_name == 'EPBP':
            bp = EPBP(g, n=20, proposal_approximation='simple')
            start_time = time.process_time()
            start_wall_time = time.time()
            bp.run(10, log_enable=False)
            cpu_time = time.process_time() - start_time
            wall_time = time.time() - start_wall_time

            for i, rv in enumerate(query_rvs):
                mmap[i] = bp.map(rv)
                # marg_logpdf = lambda x: bp.belief(x, rv, log_belief=True)  # probly slightly faster if not plotting
                marg_logpdf = utils.curry_epbp_belief(bp, rv, log_belief=True)
                margs[i] = marg_logpdf

        elif algo_name in ('OSI', 'LOSI', 'NPVI', 'LNPVI'):
            cond = True
            if cond:
                cond_g.init_nb()  # this will make cond_g rvs' .nb attributes consistent (baseline didn't care so it was OK)
            K = 1
            T = 3
            lr = 0.5
            its = 300
            fix_mix_its = int(its * 0.5)
            logging_itv = 50
            utils.set_log_potential_funs(g.factors_list, skip_existing=True)  # g factors' lpot_fun should still be None
            # above will also set the lpot_fun in all the (completely unobserved) factors in cond_g
            if algo_name in ('OSI', 'NPVI'):
                if cond:  # TODO: ugly; fix
                    _g = cond_g
                else:
                    _g = g
                if algo_name == 'OSI':
                    vi = OneShot(g=_g, K=K, T=T, seed=seed)
                else:
                    vi = NPVI(g=_g, K=K, T=T, isotropic_cov=False, seed=seed)
            else:
                if cond:
                    cg = CompressedGraphSorted(cond_g)
                else:
                    # technically incorrect; currently we should run LOSI on the conditional MRF
                    cg = CompressedGraphSorted(g)
                cg.run()
                print('number of rvs in cg', len(cg.rvs))
                print('number of factors in cg', len(cg.factors))
                if algo_name == 'LOSI':
                    vi = LiftedOneShot(g=cg, K=K, T=T, seed=seed)
                else:
                    vi = LiftedNPVI(g=cg, K=K, T=T, seed=seed)
            if cond:  # clean up; only needed cond_g.init_nb() for defining symbolic objective
                for i, rv in enumerate(g.rvs_list):
                    rv.nb = g_rv_nbs[i]  # restore; undo possible mutation from cond_g.init_nb()

            start_time = time.process_time()
            start_wall_time = time.time()
            res = vi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
            cpu_time = time.process_time() - start_time
            wall_time = time.time() - start_wall_time
            obj = res['record']['obj'][-1]

            for i, rv in enumerate(query_rvs):
                if cond:
                    m = vi.map(obs_rvs=[], query_rv=rv)
                else:
                    m = vi.map(obs_rvs=obs_rvs, query_rv=rv)
                mmap[i] = m
                assert rv.domain_type[0] == 'c', 'only looking at kl for cnode queries for now'
                crv_marg_params = vi.params['w'], rv.belief_params['mu'], rv.belief_params['var']
                margs[i] = utils.get_scalar_gm_log_prob(None, *crv_marg_params, get_fun=True)

        else:
            raise NotImplementedError

        # same for all algos
        for i, rv in enumerate(query_rvs):
            lb, ub = -np.inf, np.inf
            marg_kl = kl_continuous_logpdf(log_p=baseline_margs[i], log_q=margs[i], a=lb, b=ub)
            marg_kls[i] = marg_kl

        # same for all algos
        # print('pred mmap', mmap)
        # print('true mmap', baseline_mmap)
        mmap_err = np.mean(np.abs(mmap - baseline_mmap))
        kl_err = np.mean(marg_kls)
        print('mmap_err', mmap_err, 'kl_err', kl_err)
        algo_record = dict(cpu_time=cpu_time, wall_time=wall_time, obj=obj, mmap_err=mmap_err, kl_err=kl_err)
        for key, value in algo_record.items():
            records[algo_name][key].append(value)
        all_margs[algo_name] = margs  # for plotting convenience

if plot:
    print('plotting example marginal from last run')
    import matplotlib.pyplot as plt

    plt.figure()
    domain_real = g.rvs_list[0].domain
    xs = np.linspace(domain_real.values[0], domain_real.values[1], 100)

    crv_idxs_to_plot = list(range(len([rv for rv in query_rvs if rv.domain_type[0] == 'c'])))
    num_to_plot = 1
    crv_idxs_to_plot = crv_idxs_to_plot[:num_to_plot]
    for test_crv_idx in crv_idxs_to_plot:
        # for test_crv_idx in range(len(query_rvs)):
        for algo_name in algo_names:
            marg_logpdf = all_margs[algo_name][test_crv_idx]
            # plt.plot(xs, np.exp(marg_logpdf(xs)), label=f'{algo_name} for {test_crv_idx}')
            plt.plot(xs, np.exp([marg_logpdf(x) for x in xs]), label=f'{algo_name} for crv{test_crv_idx}')

    plt.legend(loc='best')
    plt.title('crv marginals')
    # plt.show()
    save_name = __file__.split('.py')[0]
    plt.savefig('%s.png' % save_name)

print('######################')
from collections import OrderedDict

avg_records = OrderedDict()
for algo_name in algo_names:
    record = records[algo_name]
    avg_record = OrderedDict()
    for record_field in record_fields:
        avg_record[record_field] = (np.mean(record[record_field]), np.std(record[record_field]))
    avg_records[algo_name] = avg_record

from pprint import pprint

for key, value in avg_records.items():
    print(key + ':')
    pprint(dict(value))
# import json
# output = json.dumps(avg_records, indent=0, sort_keys=True)
