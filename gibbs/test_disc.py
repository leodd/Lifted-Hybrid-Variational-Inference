import numpy as np
from disc_mrf import gibbs_sample

N = 2
dstates = [2, 2]
lpot_tables = [np.array([[5., 2.], [1., 0.]]), np.array([-1., 1])]
scopes = [(0, 1), (1,)]
nbr_factor_ids = [[] for _ in range(N)]
for j, scope in enumerate(scopes):
    for i in scope:
        nbr_factor_ids[i].append(j)

num_burnin = 500
num_samples = 2000
x = np.array([1, 0])
samples = gibbs_sample(lpot_tables, scopes, nbr_factor_ids, dstates, x, num_burnin, num_samples)
print('sample marginals', np.mean(samples, axis=0))

# ground truth
from itertools import product

margs = np.zeros(dstates)
for config in product(*[range(d) for d in dstates]):
    val = 0
    for j, scope in enumerate(scopes):
        table = lpot_tables[j]
        val += table[tuple(config[i] for i in scope)]
    val = np.e ** val
    margs[config] = val
Z = np.sum(margs)
margs /= Z
print('true marginals', np.sum(margs[1]), np.sum(margs[:, 1]))
