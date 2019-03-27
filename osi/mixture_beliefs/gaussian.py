# common procedures related to mixture of independent Gaussian mixture beliefs
import tensorflow as tf
import numpy as np


def node_belief(x, w, m, v):
    """

    :param x: shape N x ...
    :param w: shape K
    :param m: shape N x K, mixture means
    :param v: shape N x K, mixture variances
    :param backend: tf for tensorflow symbolic computation, np for eager evaluation with numpy
    :return:
    """
    assert len(m.shape) > 1, '1D case to be added later'
    [N, K] = m.shape
    # xmat = tf.reshape(x, [N, -1])    # N x d
    # xmat = tf.reshape(tf.transpose(xmat), [-1, N, 1])  # d x N x 1

    xmat = tf.reshape(tf.transpose(tf.reshape(x, [N, -1])), [-1, N, 1])  # d x N x 1
    v_inv = 1 / v
    comp_probs = (2 * np.pi) ** (-0.5) * tf.sqrt(v_inv) * tf.exp(-0.5 * (xmat - m) ** 2 * v_inv)  # d x N x K
    out = tf.reshape(tf.transpose(tf.reduce_sum(w * comp_probs, 2)), x.shape)
    return out


def edge_belief(x1, x2, w, m1, m2, v1, v2):
    if len(m1.shape) == 1:
        pass
    else:
        assert len(m1.shape) == 2
        [N, K] = m1.shape
        x1mat = tf.reshape(tf.transpose(tf.reshape(x1, [N, -1])), [-1, N, 1])  # d x N x 1
        x2mat = tf.reshape(tf.transpose(tf.reshape(x2, [N, -1])), [-1, N, 1])  # d x N x 1
        v1_inv = 1 / v1
        v2_inv = 1 / v2
        comp_probs = 1 / (2 * np.pi) * tf.sqrt(v1_inv * v2_inv) * \
                     tf.exp(-0.5 * (x1mat - m1) ** 2 * v1_inv - 0.5 * (x2mat - m2) ** 2 * v2_inv)
        out = tf.reshape(tf.transpose(tf.reduce_sum(w * comp_probs, 2)), x1.shape)
        return out
        # assert len(m1.shape) == 2
        # [N, K] = m1.shape
        # m1_NK1, m2_NK1 = tf.reshape(m1, [N, K, 1]),tf.reshape(m2, [N, K, 1])
        # v1_inv,v2_inv = 1/tf.reshape(v1, [N, K, 1]), 1/tf.reshape(v2, [N, K, 1])
        # comp_probs = 1 / (2 * np.pi) * tf.sqrt(v1_inv * v2_inv) * \
        #              tf.exp(-0.5 * (x1 - m1_NK1) ** 2 * v1_inv - 0.5 * (x2 - m2_NK1) ** 2 * v2_inv)
        # out = tf.reshape(tf.reduce_sum(tf.reshape(w, [1, K, 1]) * comp_probs, 1), x1.shape)
        # return out


def get_quad_bfe(g, w, Mu, Sigs, T, node_lpot, edge_lpot):
    """
    Get the symbolic tensorflow objective for optimizing BFE with quadrature approximation for the integrals.
    :param g: graph; for now assuming all its continuous nodes are modeled by diag gaussian mixtures
    # :param nodes: length N list/set of node ids that are modeled by diag gaussian mixtures
    :param Mu: N x K tensor of diagonal gaussian mixture nodes means
    :param Sigs: N x K tensor of diagonal gaussian mixture nodes variances
    :param T: num quad points
    :return: bfe, aux_obj; aux_obj is the actual (minimization) objective that tensorflow does auto-diff w.r.t.
    (tf can't directly differentiate thru expectations in the bfe)
    """
    from scipy.special import roots_hermite
    [N, K] = Mu.shape
    assert g.Nc == N  # TODO: no longer assume all continuous nodes are gm; allow specifying a subset of nodes
    num_cedges = len(g.Ec)
    dtype = Mu.dtype

    bfe = 0
    aux_obj = 0

    w_col = tf.reshape(w, [K, 1])

    qx_np, qw_np = roots_hermite(T)
    qx = tf.constant(qx_np, dtype=dtype)
    qw = tf.constant(qw_np, dtype=dtype)
    qw_outer = tf.constant(np.outer(qw_np, qw_np))  # TxT

    integral_coef = (np.pi) ** (-0.5)

    QY = qx * (2 * tf.reshape(Sigs, [N, K, 1])) ** 0.5 + tf.reshape(Mu, [N, K, 1])  # N x K x T
    QY = tf.stop_gradient(QY)  # don't want to differentiate w.r.t. quadrature points

    # all nodes
    num_nbrs = np.array([len(g.adj[n]) for n in g.Vc])
    num_nbrs = num_nbrs.reshape([N, 1, 1])
    node_log_belief = tf.log(node_belief(QY, w, Mu, Sigs))  # N x K x T
    F = node_lpot('c', QY) - (1 - num_nbrs) * node_log_belief  # N x K x T

    grals = integral_coef * tf.reduce_sum(qw * F, 2)  # Nc x K
    bfe += tf.reduce_sum(grals @ w_col)

    grals = integral_coef * tf.reduce_sum(qw * tf.stop_gradient(F) * node_log_belief, 2)  # treating F as const
    aux_obj += tf.reduce_sum(grals @ w_col)

    # all edges
    cedge_i = np.array([g.Vc_idx[n] for n in g.Ec[:, 0]])  # 1 x num_cedges; from
    cedge_j = np.array([g.Vc_idx[n] for n in g.Ec[:, 1]])  # 1 x num_cedges; to
    QYi = tf.gather(QY, cedge_i)  # num_cedges x K x T
    QYYi = tf.zeros([num_cedges, K, T, T], dtype=Mu.dtype) \
           + tf.reshape(QYi, [num_cedges, K, T, 1])  # hack, since there's no tf.repeat
    QYj = tf.gather(QY, cedge_j)  # num_cedges x K x T
    QYYj = tf.zeros([num_cedges, K, T, T], dtype=dtype) \
           + tf.reshape(QYj, [num_cedges, K, 1, T])  # hack, since there's no tf.repeat
    cedge_log_belief = tf.log(edge_belief(QYYi, QYYj, w, tf.gather(Mu, cedge_i), tf.gather(Mu, cedge_j),
                                          tf.gather(Sigs, cedge_i), tf.gather(Sigs, cedge_j)))
    F = edge_lpot('c', 'c', QYYi, QYYj) - cedge_log_belief

    inner_prods = tf.reduce_sum(qw_outer * F, axis=[2, 3])  # num_cedges x K
    bfe += integral_coef ** 2 * tf.reduce_sum(inner_prods @ w_col)

    inner_prods = tf.reduce_sum(qw_outer * tf.stop_gradient(F) * cedge_log_belief, axis=[2, 3])  # treating F as const
    aux_obj += integral_coef ** 2 * tf.reduce_sum(inner_prods @ w_col)

    return bfe, aux_obj


def get_quad_elbo(g, w, Mu, Sigs, T, node_lpot, edge_lpot):
    """
    Get the symbolic tensorflow objective for optimizing ELBO, with quadrature approximation for the energy terms and
    Jensen's inequality lower-bound for the mixture entropy term (using the formulae from Gershman 2012 NPV paper)
    :param g: graph; for now assuming all its continuous nodes are modeled by diag gaussian mixtures
    :param w: K tensor of mixture weights
    # :param nodes: length N list/set of node ids that are modeled by diag gaussian mixtures
    :param Mu: N x K tensor of diagonal gaussian mixture nodes means
    :param Sigs: N x K tensor of diagonal gaussian mixture nodes variances
    :param T: num quad points
    :return: elbo, aux_obj; aux_obj is the actual (minimization) objective that tensorflow does auto-diff w.r.t.
    (tf can't directly differentiate thru expectations in the bfe)
    """
    from scipy.special import roots_hermite
    [N, K] = Mu.shape
    assert g.Nc == N  # TODO: no longer assume all continuous nodes are gm; allow specifying a subset of nodes
    num_cedges = len(g.Ec)
    dtype = Mu.dtype

    elbo = 0
    aux_obj = 0

    w_col = tf.reshape(w, [K, 1])

    qx_np, qw_np = roots_hermite(T)
    qx = tf.constant(qx_np, dtype=dtype)
    qw = tf.constant(qw_np, dtype=dtype)
    qw_outer = tf.constant(np.outer(qw_np, qw_np))  # TxT

    integral_coef = (np.pi) ** (-0.5)

    QY = qx * (2 * tf.reshape(Sigs, [N, K, 1])) ** 0.5 + tf.reshape(Mu, [N, K, 1])  # N x K x T
    QY = tf.stop_gradient(QY)  # don't want to differentiate w.r.t. quadrature points

    # energy terms (same as in BFE)

    # all nodes
    node_log_belief = tf.log(node_belief(QY, w, Mu, Sigs))  # N x K x T
    F = node_lpot('c', QY)  # N x K x T

    grals = integral_coef * tf.reduce_sum(qw * F, 2)  # Nc x K
    elbo += tf.reduce_sum(grals @ w_col)

    grals = integral_coef * tf.reduce_sum(qw * tf.stop_gradient(F) * node_log_belief, 2)  # treating F as const
    aux_obj += tf.reduce_sum(grals @ w_col)

    # all edges
    cedge_i = np.array([g.Vc_idx[n] for n in g.Ec[:, 0]])  # 1 x num_cedges; from
    cedge_j = np.array([g.Vc_idx[n] for n in g.Ec[:, 1]])  # 1 x num_cedges; to
    QYi = tf.gather(QY, cedge_i)  # num_cedges x K x T
    QYYi = tf.zeros([num_cedges, K, T, T], dtype=Mu.dtype) \
           + tf.reshape(QYi, [num_cedges, K, T, 1])  # hack, since there's no tf.repeat
    QYj = tf.gather(QY, cedge_j)  # num_cedges x K x T
    QYYj = tf.zeros([num_cedges, K, T, T], dtype=dtype) \
           + tf.reshape(QYj, [num_cedges, K, 1, T])  # hack, since there's no tf.repeat
    cedge_log_belief = tf.log(edge_belief(QYYi, QYYj, w, tf.gather(Mu, cedge_i), tf.gather(Mu, cedge_j),
                                          tf.gather(Sigs, cedge_i), tf.gather(Sigs, cedge_j)))
    F = edge_lpot('c', 'c', QYYi, QYYj)

    inner_prods = tf.reduce_sum(qw_outer * F, axis=[2, 3])  # num_cedges x K
    elbo += integral_coef ** 2 * tf.reduce_sum(inner_prods @ w_col)

    inner_prods = tf.reduce_sum(qw_outer * tf.stop_gradient(F) * cedge_log_belief, axis=[2, 3])  # treating F as const
    aux_obj += integral_coef ** 2 * tf.reduce_sum(inner_prods @ w_col)

    # entropy approximation of the mixture using Jensen's inequality
    ent_lb = get_entropy_lb(w, Mu, Sigs)
    elbo += ent_lb
    aux_obj += ent_lb

    return elbo, aux_obj


def get_entropy_lb(w, Mu, Sigs):
    """
    Get symbolic tensor representing Jensen's inequality lower-bound for the mixture entropy term (using the formulae
    from Gershman 2012 NPV paper)
    :param w: K tensor of mixture weights
    :param Mu: N x K tensor of diagonal gaussian mixture nodes means
    :param Sigs:
    :return:
    """
    [N, K] = map(int, Mu.shape)

    # need to compute an K x K (symmetric) matrix of convolutions of the ith component with the jth (which turn out to
    # be simply pdf evaluations of N-dimensional Gaussians)
    conv_Mu_diffs = tf.reshape(Mu, [N, K, 1]) - tf.reshape(Mu, [N, 1, K])  # N x K x K
    conv_Sigs = tf.reshape(Sigs, [N, K, 1]) + tf.reshape(Sigs, [N, 1, K])  # N x K x K
    conv_Sigs_inv = 1 / conv_Sigs

    conv_mahalanobis_dists = tf.reduce_sum(conv_Mu_diffs ** 2 * conv_Sigs_inv, axis=0)  # K x K
    log_conv_probs = -0.5 * (N * np.log(2 * np.pi) + tf.reduce_sum(tf.log(conv_Sigs), axis=0) + conv_mahalanobis_dists)

    # w_col = tf.reshape(w, [K, 1])
    # inner_integrals = tf.log(tf.exp(log_conv_probs) @ w_col)
    inner_integrals = tf.reduce_logsumexp(log_conv_probs + tf.log(w), axis=1)
    out = - tf.reduce_sum(w * inner_integrals)
    return out
