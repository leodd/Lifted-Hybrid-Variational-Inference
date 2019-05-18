import utils

utils.set_path()

from Graph import F, RV, Domain
from Potential import LogTable, LogQuadratic, LogHybridQuadratic
import numpy as np
from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg

rvs = [RV(domain=Domain(values=(0, 1, 2), continuous=False)),
       RV(domain=Domain(values=(0, 1), continuous=False)),
       RV(domain=Domain(values=(-5, 5), continuous=True)),
       RV(domain=Domain(values=(-5, 5), continuous=True))]
N = len(rvs)
Vd = [rv for rv in rvs if rv.domain_type[0] == 'd']  # list of of discrete rvs
Vc = [rv for rv in rvs if rv.domain_type[0] == 'c']  # list of cont rvs
Nd = len(Vd)
Nc = len(Vc)
Vd_idx = {n: i for (i, n) in enumerate(Vd)}
Vc_idx = {n: i for (i, n) in enumerate(Vc)}

covs = np.array([np.eye(Nc)] * 3)
# means = np.array([[0., 0.], [0., 1.], [1., 0.]])
means = np.array([[-2., -2.], [0., 1.], [3., 0.]])
factors = [F(nb=(rvs[0], rvs[2], rvs[3]),
             log_potential_fun=LogHybridQuadratic(A=-0.5 * covs,
                                                  b=means,
                                                  c=-0.5 * np.array([np.dot(m, m) for m in means]))),
           F(nb=(rvs[0],), log_potential_fun=LogTable(np.array([-0.1, 0, 2.]))),
           F(nb=(rvs[0], rvs[1]), log_potential_fun=LogTable(np.array([[2., 0], [-0.1, 1], [0, 0.2]]))),
           F(nb=(rvs[2],), log_potential_fun=LogQuadratic(A=-0.5 * np.ones([1, 1]), b=np.zeros([1]), c=0.))
           ]

for factor in factors:  # pre-processing for efficiency; may be better done in Graph.py
    disc_nb_idx = ()
    cont_nb_idx = ()
    for rv in factor.nb:
        if rv.domain_type[0] == 'd':
            disc_nb_idx += (Vd_idx[rv],)
        else:
            assert rv.domain_type[0] == 'c'
            cont_nb_idx += (Vc_idx[rv],)
    factor.disc_nb_idx = disc_nb_idx
    factor.cont_nb_idx = cont_nb_idx

bn = convert_to_bn(factors, Vd, Vc)
print('BN params', bn)

num_burnin = 200
num_samples = 500
disc_samples, cont_samples = block_gibbs_sample(factors, rvs=None, Vd=Vd, Vc=Vc, num_burnin=num_burnin,
                                                num_samples=num_samples,
                                                disc_block_its=20)

test_drv_idx = 0
print('true test drv marg', get_drv_marg(bn[0], Vd_idx, Vd[test_drv_idx]))
print('sampled test drv marg', np.bincount(disc_samples[:, test_drv_idx]) / num_samples)

# test_crv_idx = 0
test_crv_idx = 1
test_crv_marg_params = get_crv_marg(*bn, Vc_idx, Vc[test_crv_idx])
print('true test crv marg params', test_crv_marg_params)

import matplotlib.pyplot as plt

plt.figure()
xs = np.linspace(-5, 5, 50)
plt.hist(cont_samples[:, test_crv_idx], normed=True, label='samples')
plt.plot(xs,
         np.exp(utils.scalar_gmm_log_prob(xs, w=test_crv_marg_params[0], mu=test_crv_marg_params[1],
                                          var=test_crv_marg_params[2])),
         label='ground truth marg pdf')
plt.legend(loc='best')
# plt.show()
save_name = __file__.split('.py')[0]
plt.savefig('%s.png' % save_name)
