# fun with tensors
# evaluating beliefs on P ndgrids of shape V1 x V2 x V3

import numpy as np

# np.random.seed(0)
dims = [4, 5, 6]
N = len(dims)
K = 2
P = 3
w = np.random.rand(K)

Vs = [np.random.rand(P, d) for d in dims]  # P grids
Mus = [np.random.rand(K, d) for d in dims]  # K centers

# loop version
w_broadcast = w.reshape([K] + [1] * N)
res = []
for p in range(P):
    comp_vals = []
    for n, v in enumerate(Vs):
        axis = v[p]  # Vn
        comp_val = axis - Mus[n]  # K x Vn
        comp_vals.append(comp_val)
    prod = np.einsum('ai,aj,ak->aijk', *comp_vals)  # K x V1 x V2 x V3
    re = np.sum(w_broadcast * prod, axis=0, keepdims=True)  # 1 x V1 x V2 x V3
    res.append(re)
out1 = np.concatenate(res)  # P x V1 x V2 x V3

# less-loop version
comp_vals = []
for n, v in enumerate(Vs):
    mu = Mus[n]  # K x Vn
    mu_deep = mu[:, np.newaxis, ...]  # K x 1 x Vn
    comp_val = v - mu_deep  # K x P x Vn
    comp_vals.append(comp_val)
prod = np.einsum('abi,abj,abk->abijk', *comp_vals)  # K x P x V1 x V2 x V3
w_broadcast = w.reshape([K] + [1] * (N + 1))
out2 = np.sum(w_broadcast * prod, axis=0)

print(np.all(np.isclose(out1, out2)))
