import numpy as np


def get_disc_marg_table_from_samples(samples, dstates):
    """

    :param samples: S x N array, nth column contains S samples for node n
    :param dstates: list [v1, v2, ..., vN], num discrete states for the N nodes
    :return: arr of shape [v1, v2, ..., vN]
    """
    S, N = samples.shape
    joint_counts = np.zeros(dstates, dtype=int)
    unique_joint_configs, unique_counts = np.unique(samples, axis=0, return_counts=True)
    joint_counts[tuple(unique_joint_configs[:, n] for n in range(N))] = unique_counts  # N columns of axis-wise indices
    # resulting in np.bincounts(samples[:, n], minlength=dstates[n]) = np.sum(joint_counts, axis=(0,..,n-1,n+1,..)

    # assert joint_counts.sum() == S
    return joint_counts / S


def fit_scalar_gm_from_samples(samples, K):
    """
    Given a vector of float samples, fit a Gaussian mixture
    :param samples:
    :param K:
    :return:
    """
    from sklearn import mixture
    clf = mixture.GaussianMixture(n_components=K, covariance_type='diag')
    samples = samples.reshape(-1, 1)  # required by API, when there's one feature
    clf.fit(samples)

    w = clf.weights_
    means = np.ravel(clf.means_)
    vars = np.ravel(clf.covariances_)
    return w, means, vars
