import numpy as np


class Graph:
    """Undirected graph"""

    def __init__(self, adjmat, Vd, Vstates):
        """

        :param adjmat: 2d array; must have a valid upper trianglular part
        :param Vd: arr of discrete node ids
        :param Vstates: Vstates[i] = num states the ith disc node (i.e. Vd_idx[i]) can take
        """
        N = np.shape(adjmat)[0]
        adjmat[np.arange(N), np.arange(N)] = 0  # zero out diag
        Vc = np.setdiff1d(np.arange(N), Vd)
        Vd_idx = {n: i for (i, n) in enumerate(Vd)}
        Vc_idx = {n: i for (i, n) in enumerate(Vc)}

        adjmat = np.triu(adjmat)  # avoid double counting
        rows, cols = np.where(adjmat)
        edges = list(zip(rows, cols))  # list of tups
        Ec = []
        Ed = []
        Em = []  # i disc, j cont
        for edge in edges:
            i, j = edge
            if i in Vc and j in Vc:
                Ec.append(edge)
            elif i in Vd and j in Vd:
                Ed.append(edge)
            elif i in Vd:
                Em.append(edge)
            else:
                Em.append((j, i))

        adjmat = adjmat + adjmat.T  # make symmetric
        neighbors = []
        num_neighbors = []
        for n in range(N):
            neighbors.append(np.where(adjmat[n])[0])
            num_neighbors.append(len(neighbors[n]))
        self.neighbors = neighbors
        self.num_neighbors = num_neighbors

        self.adjmat = adjmat
        self.N = N
        self.Vstates = Vstates
        self.Vd = Vd
        self.Vc = Vc
        self.Nc = len(Vc)
        self.Nd = len(Vd)
        self.Vc_idx = Vc_idx
        self.Vd_idx = Vd_idx
        self.E = edges
        self.Ec = Ec
        self.Ed = Ed
        self.Em = Em
