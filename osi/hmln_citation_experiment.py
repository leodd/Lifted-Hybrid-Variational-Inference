import utils

utils.set_path(('..', '../gibbs'))
from RelationalGraph import *
from MLNPotential import *
from Potential import QuadraticPotential, TablePotential, HybridQuadraticPotential
from EPBPLogVersion import EPBP
from OneShot import OneShot, LiftedOneShot
from NPVI import NPVI, LiftedNPVI
from CompressedGraphSorted import CompressedGraphSorted
import numpy as np
import time
from copy import copy

seed = 0
utils.set_seed(seed)

from hybrid_gaussian_mrf import HybridGaussianSampler
from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg, \
    get_rv_marg_map_from_bn_params
import sampling_utils

from KLDivergence import kl_continuous_logpdf

num_paper = 2
num_topic = 2

Paper = []
for i in range(num_paper):
    Paper.append(f'paper{i}')
Topic = []
for i in range(num_topic):
    Topic.append(f'topic{i}')

domain_bool = Domain((0, 1))
domain_real = Domain((-15, 15), continuous=True, integral_points=linspace(-15, 15, 20))

lv_paper = LV(Paper)
lv_paper_2 = LV(Paper)
lv_topic = LV(Topic)

atom_paper_popularity = Atom(domain_real, logical_variables=(lv_paper,), name='popularity')
atom_topic_popularity = Atom(domain_real, logical_variables=(lv_topic,), name='popularity')
atom_paperIn = Atom(domain_bool, logical_variables=(lv_paper, lv_topic), name='paperIn')
atom_paperIn_2 = Atom(domain_bool, logical_variables=(lv_paper_2, lv_topic), name='paperIn')
atom_cites = Atom(domain_bool, logical_variables=(lv_paper, lv_paper_2), name='cites')

w_d = 0.1
f1 = ParamF(  # disc
    MLNPotential(lambda x: imp_op(x[0] * x[1], x[2]), w=w_d),
    nb=(atom_cites, atom_paperIn, atom_paperIn_2),
    constrain=lambda sub: sub[lv_paper] != sub[lv_paper_2]
)

w_h = 0.4  # the stronger the more multi-modal things tend to be
c1, c2 = (7., 1)  # mu1 = c2 - c1 is shared mean for comp1
d1, d2 = (-10.1, 3.0)  # mu2 = d2 - d1 is shared mean for comp2
f2 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1] - c1, x[2] - c2) + (1 - x[0]) * eq_op(x[1] - d1, x[2] - d2), w=w_h),
    nb=(atom_paperIn, atom_paper_popularity, atom_topic_popularity)
)
equiv_hybrid_pot = HybridQuadraticPotential(
    A=w_h * np.array([np.array([[-1., 1.], [1., -1.]]), np.array([[-1., 1.], [1., -1.]])]),
    b=w_h * 2 * np.array([[d1 - d2, d2 - d1], [c1 - c2, c2 - c1]]),
    c=w_h * np.array([-(d1 - d2) ** 2, -(c1 - c2) ** 2])
)  # equals 0 if x[0]==0, equals -(x[1]-x[1])^2 if x[0]==1

prior_strength = 0.02
locs = [-3., 1]
f3 = ParamF(  # cont
    QuadraticPotential(A=prior_strength * -(np.eye(2)), b=prior_strength * 2 * np.array(locs),
                       c=prior_strength * -(np.square(locs).sum())),
    nb=[atom_paper_popularity, atom_topic_popularity]
)  # needed to ensure normalizability; model will be indefinite when all discrete nodes are 0


def generate_data(evidence_ratios=(.2, .5)):
    data = dict()
    X_ = np.random.choice(num_paper, int(num_paper * evidence_ratios[0]), replace=False)
    for x_ in X_:
        data[str(('popularity', f'paper{x_}'))] = np.clip(np.random.normal(0, 3), -10, 10)

    X_ = np.random.choice(num_topic, int(num_topic * evidence_ratios[1]), replace=False)
    for x_ in X_:
        data[str(('popularity', f'topic{x_}'))] = np.clip(np.random.normal(0, 3), -10, 10)

    X_ = np.random.choice(num_paper, int(num_paper * evidence_ratios[0]), replace=False)
    for x_ in X_:
        Y_ = np.random.choice(num_topic, np.random.randint(3), replace=False)
        for y_ in Y_:
            data[str(('paperIn', f'paper{x_}', f'topic{y_}'))] = int(np.random.choice([0, 1]))

    X_ = np.random.choice(num_paper, int(num_paper * 1), replace=False)
    for x_ in X_:
        Y_ = np.random.choice(num_paper, int(num_paper * 1), replace=False)
        for y_ in Y_:
            data[str(('cites', f'paper{x_}', f'paper{y_}'))] = int(np.random.choice([0, 0, 0, 1]))

    return data


rel_g = RelationalGraphSorted()
rel_g.atoms = (atom_cites, atom_paperIn, atom_paper_popularity, atom_topic_popularity)
rel_g.param_factors = (f1, f2, f3)
rel_g.init_nb()

num_tests = 5  # number of times evidence will vary; each time all methods are run to perform conditional inference
record_fields = [
    'obj',  # this is BFE/-ELBO for variational methods, -logZ for exact baseline
    'mmap_err',  # |argmax p(xi) - argmax q(xi)|, avg over all nodes i
    'kl_err',  # kl(p(xi)||q(xi)), avg over all nodes i
]
# algo_names = ['baseline', 'EPBP', 'NPVI', 'OSI']
# algo_names = ['baseline', 'NPVI', 'OSI', ]
algo_names = ['baseline', 'EPBP', 'NPVI', 'LNPVI', 'OSI', 'LOSI']
# algo_names = ['baseline', 'EPBP']
# algo_names = ['EPBP']
# assert algo_names[0] == 'baseline'
# for each algorithm, we keep a record, which is a dict mapping a record_field to a list (which will eventually be
# averaged over)
records = {algo_name: {record_field: [] for record_field in record_fields} for algo_name in algo_names}

data = dict()
for test_num in range(num_tests):
    test_seed = seed + test_num
    np.random.seed(test_seed)

    # regenerate/reload evidence
    evidence_ratios = (0.2, 0.5)
    data = generate_data(evidence_ratios)

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    g_rv_nbs = [copy(rv.nb) for rv in g.rvs_list]  # keep a copy of rv neighbors in the original graph
    print(rvs_table)

    # labels of query nodes
    query_rvs = [rv for rv in g.rvs_list if rv.domain_type[0] == 'c']

    print('number of rvs', len(g.rvs))
    print('num drvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'd']))
    print('num crvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'c']))
    print('number of factors', len(g.factors))
    print('number of evidence', len(data))

    obs_rvs = [v for v in g.rvs if v.value is not None]
    evidence = {rv: rv.value for rv in obs_rvs}
    cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs,
                                       evidence)  # this will also condition log_potential_funs

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
        mmap = np.zeros(len(query_rvs))
        margs = [None] * len(query_rvs)
        marg_kls = np.zeros(len(query_rvs))
        obj = -1

        if algo_name == 'baseline':
            # preprocessing
            # convert factors to the form accepted by HybridGaussianSampler
            # Currently there's no lifting for sampling, so we don't need to ensure the same potentials share reference
            converted_factors = []  # identical to cond_g.factors, except the potentials are converted to equivalent ones
            for factor in cond_g.factors:
                factor = copy(factor)
                if factor.domain_type == 'd' and not isinstance(factor.potential, TablePotential):
                    assert isinstance(factor.potential, MLNPotential), 'currently can only handle MLN'
                    factor.potential = utils.convert_disc_MLNPotential_to_TablePotential(factor.potential, factor.nb)
                if factor.domain_type == 'h':
                    assert isinstance(factor.potential, MLNPotential), 'currently can only handle MLN'
                    nb = factor.nb
                    num_dnb = len([v for v in nb if v.domain_type[0] == 'd'])
                    num_cnb = len([v for v in nb if v.domain_type[0] == 'c'])
                    assert num_dnb == 1 and nb[0].dstates == 2, 'must have 1st nb boolean for the hack to work'
                    if num_cnb == 2:  # of the form MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=w_h)
                        factor.potential = equiv_hybrid_pot
                    elif num_cnb == 1:  # one cont observed in the uncond factor
                        uncond_factor = factor.uncond_factor  # from conditioning
                        obs = [v.value for v in uncond_factor.nb if v in obs_rvs]
                        assert len(obs) == 1
                        obs = obs[0]
                        factor.potential = HybridQuadraticPotential(A=w_h * np.array([np.zeros([1, 1]), -np.eye(1)]),
                                                                    b=w_h * np.array([[0.], [2 * obs]]),
                                                                    c=w_h * np.array([0., -obs ** 2]))
                    else:
                        raise NotImplementedError
                if factor.domain_type == 'c':
                    if isinstance(factor.potential, MLNPotential):
                        if len(factor.nb) == 2:
                            factor.potential = equiv_hybrid_pot
                        else:
                            assert len(factor.nb) == 1
                            uncond_factor = factor.uncond_factor  # from conditioning
                            obs = [v.value for v in uncond_factor.nb if v in obs_rvs]
                            assert len(obs) == 2
                            dobs, cobs = obs[0], obs[1]
                            if dobs == 0:
                                factor.potential = QuadraticPotential(A=np.zeros([1, 1]), b=np.zeros([1]), c=0)
                            else:
                                factor.potential = QuadraticPotential(A=w_h * -np.eye(1),
                                                                      b=w_h * np.array([2 * cobs]), c=w_h * -cobs ** 2)

                assert isinstance(factor.potential, (TablePotential, QuadraticPotential, HybridQuadraticPotential))
                converted_factors.append(factor)

            utils.set_log_potential_funs(converted_factors,
                                         skip_existing=False)  # create lpot_funs to be used by baseline
            cond_g.init_rv_indices()  # create indices in the conditional mrf (for baseline and osi)
            utils.set_nbrs_idx_in_factors(converted_factors, cond_g.Vd_idx, cond_g.Vc_idx)  # preprocessing for baseline

            if baseline == 'exact':
                bn_res = convert_to_bn(converted_factors, cond_g.Vd, cond_g.Vc, return_logZ=True)
                bn = bn_res[:-1]
                logZ = bn_res[-1]
                print('true -logZ', -logZ)
                obj = -logZ
                # print('BN params', bn)

                num_dstates = np.prod([rv.dstates for rv in cond_g.Vd])
                if num_dstates > 1000:
                    print('too many dstates, exact mode finding might take a while, consider parallelizing...')

                for i, rv in enumerate(query_rvs):
                    m = get_rv_marg_map_from_bn_params(*bn, cond_g.Vd_idx, cond_g.Vc_idx, rv)
                    mmap[i] = m
                    assert rv.domain_type[0] == 'c', 'only looking at kl for cnode queries for now'
                    crv_idx = cond_g.Vc_idx[rv]
                    crv_marg_params = get_crv_marg(*bn, crv_idx)
                    marg_logpdf = utils.get_scalar_gm_log_prob(None, *crv_marg_params, get_fun=True)
                    margs[i] = marg_logpdf

            if baseline == 'gibbs':
                num_burnin = 200
                num_samples = 2000
                num_gm_components_for_crv = 3
                disc_block_its = 40
                hgsampler = HybridGaussianSampler(converted_factors, cond_g.Vd, cond_g.Vc, cond_g.Vd_idx, cond_g.Vc_idx)
                hgsampler.block_gibbs_sample(num_burnin=num_burnin, num_samples=num_samples,
                                             disc_block_its=disc_block_its)
                # np.save('cont_samples', hgsampler.cont_samples)
                # np.save('disc_samples', hgsampler.disc_samples)
                # TODO: estimate obj = -logZ from samples

                for i, rv in enumerate(query_rvs):
                    m = hgsampler.map(rv, num_gm_components_for_crv=num_gm_components_for_crv)
                    mmap[i] = m
                    assert rv.domain_type[0] == 'c', 'only looking at kl for cnode queries for now'
                    # TODO: fitting gm twice, wasteful
                    cont_samples = hgsampler.cont_samples
                    crv_idx = cond_g.Vc_idx[rv]
                    crv_marg_params = sampling_utils.fit_scalar_gm_from_samples(cont_samples[:, crv_idx],
                                                                                K=num_gm_components_for_crv)
                    marg_logpdf = utils.get_scalar_gm_log_prob(None, *crv_marg_params, get_fun=True)
                    margs[i] = marg_logpdf
            # save baseline
            baseline_mmap = mmap
            baseline_margs = margs
            marg_kls = np.zeros_like(mmap)

        elif algo_name == 'EPBP':
            bp = EPBP(g, n=20, proposal_approximation='simple')
            bp.run(10, log_enable=False)

            for i, rv in enumerate(query_rvs):
                mmap[i] = bp.map(rv)
                # marg_logpdf = lambda x: bp.belief(x, rv, log_belief=True)  # probly slightly faster if not plotting
                marg_logpdf = utils.curry_epbp_belief(bp, rv, log_belief=True)
                margs[i] = marg_logpdf
                assert rv.domain_type[0] == 'c', 'only looking at kl for cnode queries for now'
                # lb, ub = -np.inf, np.inf
                lb, ub = rv.domain.values[0], rv.domain.values[1]
                marg_kl = max(0, kl_continuous_logpdf(log_p=baseline_margs[i], log_q=margs[i], a=lb, b=ub))
                marg_kls[i] = marg_kl

        elif algo_name in ('OSI', 'LOSI', 'NPVI', 'LNPVI'):
            cond = True
            if cond:
                cond_g.init_nb()  # this will make cond_g rvs' .nb attributes consistent (baseline didn't care so it was OK)
            K = 4
            T = 16
            lr = 0.5
            its = 1500
            fix_mix_its = int(its * 0.5)
            logging_itv = 200
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

            res = vi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
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
                # lb, ub = -np.inf, np.inf
                lb, ub = rv.domain.values[0], rv.domain.values[1]
                marg_kl = max(0, kl_continuous_logpdf(log_p=baseline_margs[i], log_q=margs[i], a=lb, b=ub))
                marg_kls[i] = marg_kl

        # same for all algos
        print('pred mmap', mmap)
        print('true mmap', baseline_mmap)
        mmap_err = np.mean(np.abs(mmap - baseline_mmap))
        kl_err = np.mean(marg_kls)
        print('mmap_err', mmap_err, 'kl_err', kl_err)
        algo_record = dict(obj=obj, mmap_err=mmap_err, kl_err=kl_err)
        for key, value in algo_record.items():
            records[algo_name][key].append(value)
        all_margs[algo_name] = margs  # for plotting convenience

print('plotting example marginal from last run')

import matplotlib.pyplot as plt

plt.figure()
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

print('all records', records)

avg_records = OrderedDict()
for algo_name in algo_names:
    record = records[algo_name]
    avg_record = OrderedDict()
    for record_field in record_fields:
        # avg_record[record_field] = (np.mean(record[record_field]), np.std(record[record_field]))
        avg_record[record_field] = (np.min(record[record_field]), np.max(record[record_field]))
    avg_records[algo_name] = avg_record

from pprint import pprint

for key, value in avg_records.items():
    print(key + ':')
    pprint(dict(value))


# import json
# output = json.dumps(avg_records, indent=0, sort_keys=True)
