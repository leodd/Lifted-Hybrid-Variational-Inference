# Gibbs sampling in discrete MRF represented by factors
# For now assuming there's no deterministic interaction (i.e., the distribution is positive) for simplicity
# May need to do cythonize for speed

import numpy as np
from utils import softmax


def gibbs_sample_one(lpot_tables, scopes, nbr_factor_ids, dstates, x, its=100):
    """

    :param lpot_tables:
    :param scopes:
    :param nbr_factor_ids:
    :param dstates:
    :param x:
    :param its:
    :return:
    """
    N = len(dstates)  # num nodes
    for it in range(its):
        for n in range(N):
            d = dstates[n]
            lprobs = np.zeros(d)  # floats
            for f in nbr_factor_ids[n]:  # ids of neighboring factors
                scope = scopes[f]  # list of node ids
                lpot_table = lpot_tables[f]
                idx = tuple(slice(None) if i == n else x[i] for i in scope)
                lprobs += lpot_table[idx]
            probs = softmax(lprobs)
            x[n] = np.random.choice(d, p=probs)
    return x


def gibbs_sample(lpot_tables, scopes, nbr_factor_ids, dstates, x, num_burnin, num_samples):
    """

    :param lpot_tables:
    :param scopes:
    :param nbr_factor_ids:
    :param dstates:
    :param x:
    :param its:
    :return:
    """
    N = len(dstates)  # num nodes
    samples = np.empty([num_samples, N], dtype=int)
    for it in range(num_burnin + num_samples):
        for n in range(N):
            d = dstates[n]
            lprobs = np.zeros(d)  # floats
            for f in nbr_factor_ids[n]:  # ids of neighboring factors
                scope = scopes[f]  # list of node ids
                lpot_table = lpot_tables[f]
                idx = tuple(slice(None) if i == n else x[i] for i in scope)
                lprobs += lpot_table[idx]
            probs = softmax(lprobs)
            x[n] = np.random.choice(d, p=probs)

        sample_it = it - num_burnin
        if sample_it >= 0:
            samples[sample_it] = x
    return samples
