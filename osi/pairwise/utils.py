import networkx as nx
import numpy as np


# class Graph(nx.Graph):
#     """A thin wrapper around networkx.Graph that caches a few useful attributes for our purposes"""
#     def __init__(self, adjmat, dnodes, dstates):

def update_nodes_info(g, dnodes, dstates):
    """
    Add/update graph metadata associated with nodes; mainly to avoid re-computing this information
    on every access; should be called again whenever relevant info changes.
    :param g:
    :param dnodes:
    :param dstates:
    :return: None. Done in place.
    """

    for n in g.nodes:  # cont by default
        g.nodes[n]['type'] = 'c'

    for n, s in zip(dnodes, dstates):
        g.nodes[n]['type'] = 'd'
        g.nodes[n]['nstates'] = s

    # g.dnodes = np.array(dnodes)
    # g.cnodes = np.array(list(set(g.nodes) - set(dnodes)))
    # g.Vd = g.dnodes
    # g.Vc = g.cnodes

    Vd = np.array(dnodes)
    Vc = np.setdiff1d(np.array(g.nodes()), Vd)

    Vd_idx = {n: i for (i, n) in enumerate(Vd)}
    Vc_idx = {n: i for (i, n) in enumerate(Vc)}

    Ec = []
    Ed = []
    Em = []
    for e, edge in enumerate(g.edges):
        i, j = edge
        if i in Vc and j in Vc:
            Ec.append(edge)
            edge_type = 'cc'
        elif i in Vd and j in Vd:
            Ed.append(edge)
            edge_type = 'dd'
        else:
            Em.append(edge)
            if i in Vd:
                edge_type = 'dc'
            else:
                edge_type = 'cd'
        g.edges[edge]['type'] = edge_type

    Ec, Ed, Em = map(np.array, (Ec, Ed, Em))

    g.Vd = Vd
    g.Vc = Vc
    g.Nc = len(Vc)
    g.Nd = len(Vd)
    g.Vc_idx = Vc_idx
    g.Vd_idx = Vd_idx
    g.Ec = Ec
    g.Ed = Ed
    g.Em = Em


def build_grid_graph(nrows, ncols):
    g = nx.grid_graph(dim=[nrows, ncols])  # networkx names nodes as coordinate tuples, but I want ints
    coords_to_ids = {n: i for i, n in enumerate(sorted(g.nodes()))}  # nodes from 0 to num_nodes - 1
    edges = []
    for edge in g.edges():
        i, j = coords_to_ids[edge[0]], coords_to_ids[edge[1]]
        edges.append((i, j))
    g = nx.Graph()
    g.add_edges_from(edges)

    return g, coords_to_ids


def build_grid_graph_manually(nrows, ncols):
    num_nodes = nrows * ncols
    A = np.zeros([num_nodes, num_nodes])  # adj mat
    coords_to_ids = {}
    for n in range(num_nodes):
        row, col = n // ncols, n % ncols
        coords_to_ids[(row, col)] = n

        # above
        if row > 0:
            A[n, n - ncols] = 1
        # below
        if row < nrows - 1:
            A[n, n + ncols] = 1
        # left
        if col > 0:
            A[n, n - 1] = 1
        # right
        if col < ncols - 1:
            A[n, n + 1] = 1
    g = nx.from_numpy_matrix(A)
    return g, coords_to_ids


# tensorflow stuff
import tensorflow as tf


def poly1d(x, theta):
    """
    Compute order-R polynomial of x with coef param vector theta
    :param x: tensor
    :param theta: if shape = (R+1,), result will be the same shape as x; if shape=(N, R+1), result will have shape
      (N, *x.shape), so result[n] = poly1d(x, theta[n])
    :return:
    """
    if len(theta.shape) == 1:
        theta = tf.reshape(theta, [1, -1])  # tmp ref, doesn't alter the original arg
    else:
        assert len(theta.shape) == 2
    N, R_p1 = theta.shape

    x_row = tf.reshape(x, [1, -1])  # 1 x ?
    pows = tf.pow(x_row, np.arange(R_p1)[:, np.newaxis])  # (R+1) x ?
    out = theta @ pows  # N x ?
    out_shape = x.shape if N == 1 else [N, *x.shape]
    out = tf.reshape(out, out_shape)
    return out


def poly1d_aligned(x, theta):
    """
    The same as poly1d, but using a different theta vector for each corresponding slice of x
    :param x:
    :param theta:
    :return:
    """
    N = theta.shape[0]
    if N == 1:
        out = poly1d(x, theta)
    else:
        R_p1 = theta.shape[1]
        # x = tf.expand_dims(x, 1)  # N x ... => N x 1 x ...
        # tmp_shape = [1] * len(x.shape)
        # tmp_shape[1] = R_p1  # 1 x (R+1) x 1 x ...
        # r = np.arange(R_p1).reshape(tmp_shape)
        # pows = tf.pow(x, r)  # broadcasting to N x (R+1) x ...
        # tmp_shape[0] = Nc  # N x (R+1) x 1 x ...
        # out = tf.reduce_sum(pows * tf.reshape(Theta.cnode, tmp_shape), 1)
        # out = tf.reshape(out, x.shape)
        # return out
        ## more concisely,
        x_mat = tf.reshape(x, [N, -1, 1])  # N x d x 1
        pows = tf.pow(x_mat, np.arange(R_p1))  # N x d x (R+1)
        out = tf.matmul(pows, tf.reshape(theta, [N, R_p1, 1]))  # N x d x (R+1) @ N x (R+1) x 1 = N x d x 1
        out = tf.reshape(out, x.shape)
    return out


def poly2d(x1, x2, theta):
    """
    x1, x2 should be tensors of the same sizes
    :param x1:
    :param x2:
    :param theta: (R+1)x(R+1), whose (r1,r2)th entry gives coef for x1^(r1)*x2^(r2)
    :return:
    """
    assert len(theta.shape) == 2
    R_p1 = theta.shape[0]
    x1_pows = tf.pow(tf.reshape(x1, [1, -1]), np.arange(R_p1)[:, np.newaxis])  # (R+1) x ?
    x2_pows = tf.pow(tf.reshape(x2, [1, -1]), np.arange(R_p1)[:, np.newaxis])  # (R+1) x ?

    out = tf.reshape(tf.reduce_sum(x1_pows * (theta @ x2_pows), 0), x1.shape)
    return out


def poly2d_aligned(x1, x2, theta):
    """
    The same as poly2d, but using a different theta slice for each corresponding slice of x1, x2
    :param x1:
    :param x2:
    :param theta:
    :return:
    """
    assert x1.shape[0] == x2.shape[0] == theta.shape[0]

    N, R_p1, R_p1 = theta.shape

    x1_mat = tf.reshape(x1, [N, -1, 1])  # N x d x 1
    x2_mat = tf.reshape(x2, [N, -1, 1])  # N x d x 1
    x1_pows = tf.pow(x1_mat, np.arange(R_p1))  # N x d x (R+1)
    x2_pows = tf.pow(x2_mat, np.arange(R_p1))  # N x d x (R+1)

    out = tf.reduce_sum(tf.matmul(x1_pows, theta) * x2_pows, 2)
    out = tf.reshape(out, x1.shape)
    return out


# miscellaneous

# I/O with matlab; see https://docs.scipy.org/doc/scipy/reference/tutorial/io.html for how to
def loadmat_mrf(mat):
    from scipy.io import loadmat
    m = loadmat(mat, struct_as_record=False, squeeze_me=True)
    assert 'gr' in m and 'Theta' in m
    # graph obj from matlab
    g = m['gr']
    # restore important fields to make them readily usable in Python;
    # recall matlab indexing starts at 1
    if g.Vc_idx.size > 0:
        g.Vc_idx = g.Vc_idx.toarray().squeeze().astype(int)  # originally a scipy sparse matrix
        g.Vc_idx -= 1
    if g.Vd_idx.size > 0:
        g.Vd_idx = g.Vd_idx.toarray().squeeze().astype(int)  # originally a scipy sparse matrix
        g.Vd_idx -= 1
    if g.Vc.size > 0:
        g.Vc -= 1
    if g.Vd.size > 0:
        g.Vd -= 1
    g.E = g.edges  # alias to avoid name-clashing with networkx
    g.E -= 1
    g.Ec -= 1
    g.Ed -= 1
    g.Em -= 1
    g.Ec_idx = g.Ed_idx = g.Em_idx = g.edge_idx = None  # TODO: fix these
    g.hidden_nodes -= 1
    g.label_node -= 1
    g.neighbors -= 1
    cell_arrays = [g.neighbors, g.cneighbors, g.dneighbors]
    for cell_array in cell_arrays:
        for a in cell_array:
            if a.size > 0:
                a -= 1

    # Theta struct
    Theta = m['Theta']
    Theta.cnode = Theta.cnode.T  # (R+1) x num_nodes -> num_nodes x (R+1)
    Theta.cedge = np.swapaxes(Theta.cedge, 0, 2)  # (R+1) x (R+1) x num_cedges -> num_cedges (R+1) x (R+1)
    return g, Theta
