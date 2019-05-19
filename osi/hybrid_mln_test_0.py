import utils

utils.set_path(('..', '../gibbs'))

seed = 0
utils.set_seed(seed)

from Graph import F, RV, Domain, Graph
from Potential import LogTable, LogQuadratic, LogHybridQuadratic
from Potential import TablePotential, HybridQuadraticPotential, QuadraticPotential
from MLNPotential import MLNPotential, and_op, eq_op
import numpy as np

from EPBPLogVersion import EPBP
from OneShot import OneShot, LiftedOneShot

from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg, get_rv_marg_map

rvs = [RV(domain=Domain(values=(0, 1), continuous=False)),
       RV(domain=Domain(values=(0, 1), continuous=False)),
       RV(domain=Domain(values=(-5, 5), continuous=True)),
       RV(domain=Domain(values=(-5, 5), continuous=True))]
Nc = 2
covs = np.array([np.eye(Nc)] * len(rvs[0].values))
# means = np.array([[0., 0.], [0., 1.], [1., 0.]])
means = np.array([[-2., -2.], [0., 1.], [3., 0.]])
# factors = [F(nb=(rvs[0], rvs[2], rvs[3]),
#              log_potential_fun=LogHybridQuadratic(A=-0.5 * covs,
#                                                   b=means,
#                                                   c=-0.5 * np.array([np.dot(m, m) for m in means]))),
#            F(nb=(rvs[0],), log_potential_fun=LogTable(np.array([-0.1, 0, 2.]))),
#            F(nb=(rvs[0], rvs[1]), log_potential_fun=LogTable(np.array([[2., 0], [-0.1, 1]]))),
#            F(nb=(rvs[2],), log_potential_fun=LogQuadratic(A=-0.5 * np.ones([1, 1]), b=np.zeros([1]), c=0.))
#            ]

factors = [F(nb=(rvs[0], rvs[2], rvs[3]),
             potential=MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=0.01)),
           # F(nb=(rvs[0], rvs[1]), potential=MLNPotential(lambda x: imp_op(x[0] * x[1], x[2]), w=1)),
           F(nb=(rvs[0], rvs[1]), potential=MLNPotential(lambda x: and_op(x[0], x[1]), w=1)),
           # F(nb=(rvs[2],), potential=QuadraticPotential(A=-0.5 * np.ones([1, 1]), b=np.zeros([1]), c=0.))
           F(nb=(rvs[2], rvs[3]), potential=QuadraticPotential(A=-0.5 * np.eye(2), b=np.array([-0.1, 0.2]), c=0.))
           # ensure normalizability
           ]

g = Graph()  # not really needed here; just to conform to existing API
g.rvs = rvs
g.factors = factors
g.init_nb()
g.init_rv_indices()
Vd, Vc, Vd_idx, Vc_idx = g.Vd, g.Vc, g.Vd_idx, g.Vc_idx

names = ('EPBP', 'OSI')
num_algos = len(names)
mmap_res = np.empty([num_algos, len(rvs)])

algo = 0
name = names[algo]
bp = EPBP(g, n=20, proposal_approximation='simple')
# start_time = time.process_time()
bp.run(10, log_enable=False)
# time_cost[name] = (time.process_time() - start_time) / num_runs / num_tests + time_cost.get(name, 0)
# print(name, f'time {time.process_time() - start_time}')
for i, rv in enumerate(rvs):
    mmap_res[algo, i] = bp.map(rv)

algo += 1
name = names[algo]
utils.set_log_potential_funs(g.factors_list)  # OSI assumes factors have callable .log_potential_fun
K = 2
T = 12
lr = 0.4
its = 200
fix_mix_its = int(its * 0.8)
osi = OneShot(g=g, K=K, T=T, seed=seed)  # can be moved outside of all loops if the ground MRF doesn't change
osi.run(lr=lr, its=its, fix_mix_its=fix_mix_its)
for i, rv in enumerate(rvs):
    mmap_res[algo, i] = osi.map(obs_rvs=[], query_rv=rv)

# ground truth
# first, need to convert factor.log_potential_funs to one of (LogTable, LogQuadratic, LogHybridQuadratic);
# this can be done by first converting the corresponding potentials to one of (TablePotential, QuadraticPotential,
# HybridQuadraticPotential), then calling the .to_log_potential method on the potential objects
# manual conversion here:
factors[0].potential = HybridQuadraticPotential(
    A=np.array([np.zeros([2, 2]), -0.5 * factors[0].potential.w * np.array([[1., -1.], [-1., 1.]])]),
    b=np.zeros([rvs[0].dstates, 2]),  # 2 is b/c there's 2 cont nodes
    c=np.zeros(rvs[0].dstates)
)
factors[1].potential = utils.convert_disc_MLNPotential_to_TablePotential(factors[1].potential, factors[1].nb)
utils.set_log_potential_funs(factors, skip_existing=False)  # reset lpot_funs

bn_res = convert_to_bn(factors, Vd, Vc, return_Z=True)
bn = bn_res[:-1]
Z = bn_res[-1]
print('BN params', bn)
print('true -logZ', -np.log(Z))
true_mmap = np.empty(len(rvs))
for i, rv in enumerate(rvs):
    true_mmap[i] = get_rv_marg_map(*bn, Vd_idx, Vc_idx, rv)

for a, name in enumerate(names):
    print(f'{name} mmap diff', mmap_res[a] - true_mmap)

#
# num_burnin = 200
# num_samples = 500
# disc_samples, cont_samples = block_gibbs_sample(factors, Vd=Vd, Vc=Vc, num_burnin=num_burnin,
#                                                 num_samples=num_samples, disc_block_its=20)
#
# test_drv_idx = 0
# print('true test drv marg', get_drv_marg(bn[0], Vd_idx, Vd[test_drv_idx]))
# print('sampled test drv marg', np.bincount(disc_samples[:, test_drv_idx]) / num_samples)
#
# # test_crv_idx = 0
# test_crv_idx = 1
# test_crv_marg_params = get_crv_marg(*bn, Vc_idx, Vc[test_crv_idx])
# print('true test crv marg params', test_crv_marg_params)
#
# import matplotlib.pyplot as plt
#
# plt.figure()
# xs = np.linspace(-5, 5, 50)
# plt.hist(cont_samples[:, test_crv_idx], normed=True, label='samples')
# plt.plot(xs,
#          np.exp(utils.get_scalar_gm_log_prob(xs, w=test_crv_marg_params[0], mu=test_crv_marg_params[1],
#                                              var=test_crv_marg_params[2])),
#          label='ground truth marg pdf')
# plt.legend(loc='best')
# # plt.show()
# save_name = __file__.split('.py')[0]
# plt.savefig('%s.png' % save_name)