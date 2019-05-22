import utils

utils.set_path()
import sampling_utils

from Graph import F, RV, Domain, Graph
from Potential import LogTable, LogQuadratic, LogHybridQuadratic
import numpy as np

seed = 0
np.random.seed(0)
from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg

rvs = [RV(domain=Domain(values=(0, 1, 2), continuous=False)), RV(domain=Domain(values=(-5, 5), continuous=True)),
       RV(domain=Domain(values=(-5, 5), continuous=True))]
Nc = 2
covs = np.array([np.eye(Nc)] * rvs[0].dstates)
# means = np.array([[0., 0.], [0., 1.], [1., 0.]])
means = np.array([[-2., -2.], [0., 1.], [3., 0.]])
factors = [F(nb=(rvs[0], rvs[1], rvs[2]),
             log_potential_fun=LogHybridQuadratic(A=-0.5 * covs,
                                                  b=means,
                                                  c=-0.5 * np.array([np.dot(m, m) for m in means]))),
           F(nb=(rvs[0],), log_potential_fun=LogTable(np.array([-0.1, 0, 0.2]))),
           F(nb=(rvs[2],), log_potential_fun=LogQuadratic(A=-0.5 * np.ones([1, 1]), b=np.zeros([1]), c=0.))
           ]

Vd = [rv for rv in rvs if rv.domain_type[0] == 'd']  # list of of discrete rvs
Vc = [rv for rv in rvs if rv.domain_type[0] == 'c']  # list of cont rvs
Vd_idx = {n: i for (i, n) in enumerate(Vd)}
Vc_idx = {n: i for (i, n) in enumerate(Vc)}
utils.set_nbrs_idx_in_factors(factors, Vd_idx=Vd_idx, Vc_idx=Vc_idx)  # pre-processing for hybrid mln stuff
dstates = [rv.dstates for rv in Vd]

bn = convert_to_bn(factors, Vd, Vc)
print('BN params', bn)

num_burnin = 200
num_samples = 1000
disc_samples, cont_samples = block_gibbs_sample(factors, Vd=Vd, Vc=Vc, num_burnin=num_burnin,
                                                num_samples=num_samples, disc_block_its=2)
sampled_disc_marginal_table = sampling_utils.get_disc_marg_table_from_samples(disc_samples, dstates)

test_drv_idx = 0
print('true test drv marg', get_drv_marg(bn[0], test_drv_idx))
print('sampled test drv marg', get_drv_marg(sampled_disc_marginal_table, test_drv_idx))

test_crv_idx = 0
# test_crv_idx = 1
test_crv_marg_params = get_crv_marg(*bn, test_crv_idx)
print('true test crv marg params', test_crv_marg_params)
sampled_test_crv_marg_params = sampling_utils.fit_scalar_gm_from_samples(cont_samples[:, test_crv_idx], K=3)
print('sampled test crv marg params (using gm fit)', sampled_test_crv_marg_params)

import matplotlib.pyplot as plt

plt.figure()
xs = np.linspace(-5, 5, 50)
plt.hist(cont_samples[:, test_crv_idx], normed=True, label='samples')
plt.plot(xs,
         np.exp(utils.get_scalar_gm_log_prob(xs, w=test_crv_marg_params[0], mu=test_crv_marg_params[1],
                                             var=test_crv_marg_params[2])),
         label='ground truth marg pdf')
plt.plot(xs,
         np.exp(utils.get_scalar_gm_log_prob(xs, w=sampled_test_crv_marg_params[0], mu=sampled_test_crv_marg_params[1],
                                             var=sampled_test_crv_marg_params[2])),
         label='marg pdf fit from samples (using gm)')
plt.legend(loc='best')
# plt.show()
save_name = __file__.split('.py')[0]
plt.savefig('%s.png' % save_name)
