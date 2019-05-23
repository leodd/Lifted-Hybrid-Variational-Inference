import utils

utils.set_path(('..', '../gibbs'))
from RelationalGraph import *
from MLNPotential import *
from Potential import QuadraticPotential, TablePotential, HybridQuadraticPotential
from EPBPLogVersion import EPBP
# from OneShot import OneShot, LiftedOneShot
# from CompressedGraphSorted import CompressedGraphSorted
import numpy as np
import time

from copy import copy

seed = 3
utils.set_seed(seed)

from hybrid_gaussian_mrf import HybridGaussianSampler
from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg, \
    get_rv_marg_map_from_bn_params
import sampling_utils

from KLDivergence import kl_continuous_logpdf

num_x = 4
num_y = 2

X = []
for x in range(num_x):
    X.append(f'x{x}')
Y = []
for y in range(num_y):
    Y.append(f'y{y}')
S = ['T1', 'T2', 'T3']

domain_bool = Domain((0, 1))
domain_real = Domain((-15, 15), continuous=True, integral_points=linspace(-15, 15, 20))

lv_x = LV(X)
lv_y = LV(X)
lv_s = LV(S)
lv_s2 = LV(S)

atom_A = Atom(domain_real, logical_variables=(lv_x,), name='A')
atom_B = Atom(domain_real, logical_variables=(lv_s,), name='B')
atom_C = Atom(domain_bool, logical_variables=(lv_x, lv_s), name='C')
atom_C2 = Atom(domain_bool, logical_variables=(lv_y, lv_s2), name='C')
atom_D = Atom(domain_bool, logical_variables=(lv_x, lv_y), name='D')

f1 = ParamF(  # disc
    MLNPotential(lambda x: imp_op(x[0] * x[1], x[2]), w=0.1),
    nb=(atom_D, atom_C, atom_C2),
    constrain=lambda sub: (sub[lv_s] == 'T1' and sub[lv_s2] == 'T1') or (sub[lv_s] == 'T1' and sub[lv_s2] == 'T2')
)

w_h = 1  # the stronger the more multi-modal things tend to be
f2 = ParamF(  # hybrid
    MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=w_h),
    nb=(atom_C, atom_A, atom_B)
)
equiv_hybrid_pot = HybridQuadraticPotential(
    A=w_h * np.array([np.array([[0., 0], [0, 0]]), np.array([[-1., 1.], [1., -1.]])]),
    b=w_h * np.array([[0., 0.], [0., 0.]]),
    c=w_h * np.array([0., 0.])
)  # equals 0 if x[0]==0, equals -(x[1]-x[1])^2 if x[0]==1

prior_strength = 0.01
f3 = ParamF(  # cont
    QuadraticPotential(A=-prior_strength * (np.eye(2)), b=np.array([0., 0.]), c=0.),
    nb=[atom_A, atom_B]
)  # needed to ensure normalizability; model will be indefinite when all discrete nodes are 0

rel_g = RelationalGraphSorted()
rel_g.atoms = (atom_A, atom_B, atom_C, atom_D)
rel_g.param_factors = (f1, f2, f3)
rel_g.init_nb()

data = dict()
num_tests = 1

import matplotlib.pyplot as plt

plt.figure()
xs = np.linspace(domain_real.values[0], domain_real.values[1], 100)

for test_num in range(num_tests):
    test_seed = seed + test_num

    # regenerate/reload evidence
    data.clear()
    B_vals = np.random.normal(loc=0, scale=5, size=len(S))  # special treatment for the story
    # B_vals = np.random.uniform(low=domain_real.values[0], high=domain_real.values[1], size=len(S))
    # B_vals = [-14, 2, 20]
    for i, s in enumerate(S):
        data[('B', s)] = B_vals[i]

    for x in X:
        for y in X:
            if x != y:
                data[('D', x, y)] = np.random.randint(2)

    evidence_ratio = 0.5
    x_idx = np.random.choice(len(X), int(len(X) * evidence_ratio), replace=False)
    for i in x_idx:
        data['C', X[i], 'T1'] = np.random.randint(2)

    x_idx = np.random.choice(len(X), int(len(X) * evidence_ratio), replace=False)
    for i in x_idx:
        data['C', X[i], 'T2'] = np.random.randint(2)

    x_idx = np.random.choice(len(X), int(len(X) * evidence_ratio), replace=False)
    for i in x_idx:
        data['C', X[i], 'T3'] = np.random.randint(2)

    print(data)

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()

    print('factors with duplicate nbrs:', )
    print([f.nb for f in g.factors_list if len(f.nb) != len(set(f.nb))])

    # labels of query nodes
    key_list = list()
    for x_ in X:
        if ('A', x_) not in data:
            key_list.append(('A', x_))
    query_rvs = [rvs_table[key] for key in key_list]

    print('number of rvs', len(g.rvs))
    print('num drvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'd']))
    print('num crvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'c']))
    print('number of factors', len(g.factors))
    print('number of evidence', len(data))
    #
    obs_rvs = [v for v in g.rvs if v.value is not None]
    evidence = {rv: rv.value for rv in obs_rvs}
    cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs,
                                       evidence)  # this will also condition log_potential_funs
    #
    print('cond number of rvs', len(cond_g.rvs))
    print('cond num drvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'd']))
    print('cond num crvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'c']))

    algo_name = 'baseline'
    mmap = np.zeros(len(query_rvs))
    margs = [None] * len(query_rvs)
    marg_kls = np.zeros(len(query_rvs))
    obj = -1

    if algo_name == 'baseline':
        baseline = 'exact'
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
            num_samples = 500
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
        cpu_time = wall_time = 0  # don't care

        print('mmap baseline', mmap)

        for i, rv in enumerate(query_rvs):
            marg_logpdf = margs[i]
            plt.plot(xs, np.exp([marg_logpdf(x) for x in xs]), label=f'{algo_name} for crv{i}')

plt.legend(loc='best')
plt.title('crv marginals')
# plt.show()
save_name = __file__.split('.py')[0]
plt.savefig('%s.png' % save_name)

print('######################')
from collections import OrderedDict
#
# avg_records = OrderedDict()
# for algo_name in algo_names:
#     record = records[algo_name]
#     avg_record = OrderedDict()
#     for record_field in record_fields:
#         avg_record[record_field] = (np.mean(record[record_field]), np.std(record[record_field]))
#     avg_records[algo_name] = avg_record
#
# from pprint import pprint
#
# for key, value in avg_records.items():
#     print(key + ':')
#     pprint(dict(value))
# # import json
# # output = json.dumps(avg_records, indent=0, sort_keys=True)
