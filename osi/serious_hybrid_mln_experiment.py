import utils

utils.set_path(('..', '../gibbs'))
from RelationalGraph import *
from MLNPotential import *
from Potential import QuadraticPotential, TablePotential, HybridQuadraticPotential
from EPBPLogVersion import EPBP
from OneShot import OneShot, LiftedOneShot
from CompressedGraphSorted import CompressedGraphSorted
import numpy as np
import time
from copy import copy

seed = 1
utils.set_seed(seed)

from hybrid_gaussian_mrf import HybridGaussianSampler
from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg, \
    get_rv_marg_map_from_bn_params

num_x = 2
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

w_h = 0.4  # the stronger the more multi-modal things tend to be
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

rel_g = RelationalGraph()
rel_g.atoms = (atom_A, atom_B, atom_C, atom_D)
rel_g.param_factors = (f1, f2, f3)
rel_g.init_nb()

num_tests = 1  # num rounds with different queries
num_runs = 1

avg_diff = dict()
err_var = dict()
time_cost = dict()

data = dict()

for test_num in range(num_tests):
    test_seed = seed + test_num
    data.clear()
    #
    # X_ = np.random.choice(num_x, int(num_x * 0.2), replace=False)
    # for x_ in X_:
    #     data[('B', f'x{x_}')] = np.clip(np.random.normal(0, 5), -10, 10)
    #
    # X_ = np.random.choice(num_x, int(num_x * 1), replace=False)
    # for x_ in X_:
    #     # S_ = np.random.choice(num_s, 2, replace=False)
    #     S_ = np.random.choice(num_s, int(num_s * 1), replace=False)
    #     for s_ in S_:
    #         data[('D', f'x{x_}', f's{s_}')] = np.random.choice([0, 1])
    #
    # for y_ in Y:
    #     # S_ = np.random.choice(num_s, 5, replace=False)
    #     S_ = np.random.choice(num_s, int(num_s * 1), replace=False)
    #     for s_ in S_:
    #         data[('E', y_, f's{s_}')] = np.random.choice([0, 1])
    #

    B_vals = np.random.normal(loc=0, scale=10, size=len(S))  # special treatment for the story
    for i, s in enumerate(S):
        data[('B', s)] = B_vals[i]

    # data[('A', 'x0')] = 1.3  # and whatever other evidence
    print(data)

    rel_g.data = data
    g, rvs_table = rel_g.grounded_graph()
    print(rvs_table)

    # labels of query nodes
    key_list = list()
    for x_ in X:
        if ('A', x_) not in data:
            key_list.append(('A', x_))

    ans = dict()

    print('number of rvs', len(g.rvs))
    print('num drvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'd']))
    print('num crvs', len([rv for rv in g.rvs if rv.domain_type[0] == 'c']))
    print('number of factors', len(g.factors))
    print('number of evidence', len(data))

    obs_rvs = [v for v in g.rvs if v.value is not None]
    evidence = {rv: rv.value for rv in obs_rvs}
    cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs,
                                       evidence)  # this will also condition log_potential_funs

    baseline = 'exact'
    # baseline = 'gibbs'

    # preprocessing
    # convert factors to the form accepted by HybridGaussianSampler
    # Currently there's no lifting for sampling, so we don't need to ensure the same potentials share reference
    g2 = copy(cond_g)  # shallow copy
    g2.factors = []
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
                print("shouldn't happen in this example...")
                assert len(factor.nb) == 2
                factor.potential = equiv_hybrid_pot

        assert isinstance(factor.potential, (TablePotential, QuadraticPotential, HybridQuadraticPotential))
        g2.factors.append(factor)

    utils.set_log_potential_funs(g2.factors, skip_existing=False)  # create lpot_funs to be used by baseline
    g2.init_rv_indices()
    Vd, Vc, Vd_idx, Vc_idx = g2.Vd, g2.Vc, g2.Vd_idx, g2.Vc_idx

    # currently using the same ans
    if baseline == 'exact':
        bn_res = convert_to_bn(g2.factors, Vd, Vc, return_Z=True)
        bn = bn_res[:-1]
        Z = bn_res[-1]
        print('true -logZ', -np.log(Z))
        # print('BN params', bn)

        num_dstates = np.prod(g2.dstates)
        if num_dstates > 1000:
            print('num modes too large, exact mode finding might take a while, consider parallelizing...')
        for i, key in enumerate(key_list):
            rv = rvs_table[key]
            ans[key] = get_rv_marg_map_from_bn_params(*bn, Vd_idx, Vc_idx, rv)

    if baseline == 'gibbs':
        num_burnin = 200
        num_samples = 500
        num_gm_components_for_crv = 3
        disc_block_its = 40
        hgsampler = HybridGaussianSampler(g2)
        hgsampler.block_gibbs_sample(num_burnin=num_burnin, num_samples=num_samples, disc_block_its=disc_block_its)
        # np.save('cont_samples', hgsampler.cont_samples)
        # np.save('disc_samples', hgsampler.disc_samples)
        for i, key in enumerate(key_list):
            rv = rvs_table[key]
            ans[key] = hgsampler.map(rv, num_gm_components_for_crv=num_gm_components_for_crv)

    print('baseline', [ans[key] for i, key in enumerate(key_list)])

    name = 'EPBP'
    res = np.zeros((len(key_list), num_runs))
    for j in range(num_runs):
        # np.random.seed(test_seed + j)
        bp = EPBP(g, n=20, proposal_approximation='simple')
        start_time = time.process_time()
        bp.run(10, log_enable=False)
        time_cost[name] = (time.process_time() - start_time) / num_runs / num_tests + time_cost.get(name, 0)
        print(name, f'time {time.process_time() - start_time}')
        for i, key in enumerate(key_list):
            res[i, j] = bp.map(rvs_table[key])
        print(res[:, j])
    # for i, key in enumerate(key_list):
    #     ans[key] = np.average(res[i, :])
    for i, key in enumerate(key_list):
        res[i, :] -= ans[key]
    avg_diff[name] = np.average(np.average(abs(res), axis=1)) / num_tests + avg_diff.get(name, 0)
    err_var[name] = np.average(np.average(res ** 2, axis=1)) / num_tests + err_var.get(name, 0)
    print(name, 'diff', np.average(np.average(abs(res), axis=1)))
    print(name, 'var', np.average(np.average(res ** 2, axis=1)) - np.average(np.average(abs(res), axis=1)) ** 2)

    name = 'OSI'
    cond = True
    K = 3
    T = 16
    lr = 0.5
    its = 1500
    fix_mix_its = int(its * 0.5)
    logging_itv = 50
    res = np.zeros((len(key_list), num_runs))
    utils.set_log_potential_funs(g.factors_list)
    if cond:
        osi = OneShot(g=cond_g, K=K, T=T, seed=seed)
    else:
        osi = OneShot(g=g, K=K, T=T, seed=seed)
    for j in range(num_runs):
        # utils.set_seed(test_seed + j)
        start_time = time.process_time()
        osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
        time_cost[name] = (time.process_time() - start_time) / num_runs / num_tests + time_cost.get(name, 0)
        print(name, f'time {time.process_time() - start_time}')
        for i, key in enumerate(key_list):
            rv = rvs_table[key]
            if cond:
                mmap = osi.map(obs_rvs=[], query_rv=rv)
            else:
                mmap = osi.map(obs_rvs=obs_rvs, query_rv=rv)
            res[i, j] = mmap
        print(res[:, j])
        # print(osi.params)
        print('Mu =\n', osi.params['Mu'], '\nVar =\n', osi.params['Var'])
    for i, key in enumerate(key_list):
        res[i, :] -= ans[key]
    avg_diff[name] = np.average(np.average(abs(res), axis=1)) / num_tests + avg_diff.get(name, 0)
    err_var[name] = np.average(np.average(res ** 2, axis=1)) / num_tests + err_var.get(name, 0)
    print(name, 'diff', np.average(np.average(abs(res), axis=1)))
    print(name, 'var', np.average(np.average(res ** 2, axis=1)) - np.average(np.average(abs(res), axis=1)) ** 2)

    name = 'LOSI'
    if cond:
        cond_g.init_nb()  # this will make cond_g rvs' .nb attributes consistent (baseline/OSI didn't care so it was OK)
        cg = CompressedGraphSorted(cond_g)
    else:
        cg = CompressedGraphSorted(g)  # technically incorrect; currently we should run LOSI on the conditional MRF
    cg.run()
    print('number of rvs in cg', len(cg.rvs))
    print('number of factors in cg', len(cg.factors))
    losi = LiftedOneShot(g=cg, K=K, T=T,
                         seed=seed)  # can be moved outside of all loops if the ground MRF doesn't change
    for j in range(num_runs):
        # utils.set_seed(test_seed + j)
        start_time = time.process_time()
        losi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
        time_cost[name] = (time.process_time() - start_time) / num_runs / num_tests + time_cost.get(name, 0)
        print(name, f'time {time.process_time() - start_time}')
        for i, key in enumerate(key_list):
            rv = rvs_table[key]
            if cond:
                mmap = losi.map(obs_rvs=[], query_rv=rv)
            else:
                mmap = losi.map(obs_rvs=obs_rvs, query_rv=rv)
            res[i, j] = mmap
        print(res[:, j])
        # print(osi.params)
        print('Mu =\n', losi.params['Mu'], '\nVar =\n', losi.params['Var'])
    for i, key in enumerate(key_list):
        res[i, :] -= ans[key]
    avg_diff[name] = np.average(np.average(abs(res), axis=1)) / num_tests + avg_diff.get(name, 0)
    err_var[name] = np.average(np.average(res ** 2, axis=1)) / num_tests + err_var.get(name, 0)
    print(name, 'diff', np.average(np.average(abs(res), axis=1)))
    print(name, 'var', np.average(np.average(res ** 2, axis=1)) - np.average(np.average(abs(res), axis=1)) ** 2)

print('plotting example marginal from last run')

import matplotlib.pyplot as plt

plt.figure()
xs = np.linspace(domain_real.values[0], domain_real.values[1], 100)

for test_crv_idx in range(len(Vc)):
    if baseline == 'exact':
        test_crv_marg_params = get_crv_marg(*bn, test_crv_idx)
        plt.plot(xs, np.exp(utils.get_scalar_gm_log_prob(xs, w=test_crv_marg_params[0], mu=test_crv_marg_params[1],
                                                         var=test_crv_marg_params[2])),
                 label=f'true marg pdf {test_crv_idx}')

    if baseline == 'gibbs':
        plt.hist(hgsampler.cont_samples[:, test_crv_idx], normed=True, label='samples')

    plot_losi = True
    if plot_losi:
        osi = losi
    osi_test_crv_marg_params = osi.params['w'], osi.params['Mu'][test_crv_idx], osi.params['Var'][test_crv_idx]
    plt.plot(xs, np.exp(utils.get_scalar_gm_log_prob(xs, w=osi_test_crv_marg_params[0], mu=osi_test_crv_marg_params[1],
                                                     var=osi_test_crv_marg_params[2])),
             label=f'OSI marg pdf {test_crv_idx}')
plt.legend(loc='best')
plt.title('crv marginals')
# plt.show()
save_name = __file__.split('.py')[0]
plt.savefig('%s.png' % save_name)

print('######################')
for name, v in time_cost.items():
    print(name, f'avg time {v}')
for name, v in avg_diff.items():
    print(name, f'diff {v}')
    print(name, f'std {np.sqrt(abs(err_var[name] - v ** 2))}')  # sqrt arg can sometimes be a tiny bit negative
