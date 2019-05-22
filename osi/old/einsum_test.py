# fun with tensors
# evaluating beliefs on M ndgrids of shape V1 x V2 x V3

import numpy as np

# np.random.seed(0)
dims = [4, 5, 6]
N = len(dims)
K = 2
M = 3
w = np.random.rand(K)

Vs = [np.random.rand(M, d) for d in dims]  # M grids
Mus = [np.random.rand(K, 1) for d in dims]  # K centers x N dimension

# loop version
w_broadcast = w.reshape([K] + [1] * N)
res = []
for m in range(M):
    comp_vals = []
    for n, v in enumerate(Vs):
        axis = v[m]  # Vn
        comp_val = axis - Mus[n]  # K x Vn
        comp_vals.append(comp_val)
    prod = np.einsum('ai,aj,ak->aijk', *comp_vals)  # K x V1 x V2 x V3
    re = np.sum(w_broadcast * prod, axis=0, keepdims=True)  # 1 x V1 x V2 x V3
    res.append(re)
out1 = np.concatenate(res)  # M x V1 x V2 x V3

# less-loop version
comp_vals = []
for n, v in enumerate(Vs):
    mu = Mus[n]  # K x 1
    mu_deep = mu[:, np.newaxis, ...]  # K x 1 x 1
    comp_val = v - mu_deep  # K x M x Vn
    comp_vals.append(comp_val)
prod = np.einsum('abi,abj,abk->abijk', *comp_vals)  # K x M x V1 x V2 x V3
w_broadcast = w.reshape([K] + [1] * (N + 1))
out2 = np.sum(w_broadcast * prod, axis=0)

print(np.all(np.isclose(out1, out2)))
