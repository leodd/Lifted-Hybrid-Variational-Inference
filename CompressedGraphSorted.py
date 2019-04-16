from Graph import *
from collections import Counter
from statistics import mean
from random import uniform
import numpy as np

import itertools


class SuperRV:
    id_counter = itertools.count()  # will assign unique numeric ids to instances of the class, auto-incrementing from 0

    def __init__(self, rvs, domain=None):
        self.rvs = rvs
        self.domain = next(iter(rvs)).domain if domain is None else domain
        self.nb = None
        for rv in rvs:
            rv.cluster = self
        self.id = next(self.id_counter)

    def __lt__(self, other):
        return self.id < other.id

    @property
    def domain_type(self):
        rv0 = next(iter(self.rvs))
        return rv0.domain_type

    @property
    def dstates(self):
        rv0 = next(iter(self.rvs))
        return rv0.dstates

    @property
    def values(self):
        rv0 = next(iter(self.rvs))
        return rv0.values


    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def update_nb(self):
        rv = next(iter(self.rvs))
        self.nb = tuple(map(self.get_cluster, rv.nb))

    def split_by_structure(self):
        clusters = dict()
        for rv in sorted(self.rvs):
            signature = tuple(sorted(map(self.get_cluster, rv.nb)))
            if signature in clusters:
                clusters[signature].add(rv)
            else:
                clusters[signature] = {rv}

        res = set()
        i = iter(clusters)

        # reuse THIS instance
        self.rvs = clusters[next(i)]
        res.add(self)

        for _ in range(1, len(clusters)):
            res.add(SuperRV(clusters[next(i)], self.domain))

        return res


class SuperF:
    id_counter = itertools.count()  # will assign unique numeric ids to instances of the class, auto-incrementing from 0

    def __init__(self, factors):
        self.factors = factors
        self.potential = next(iter(factors)).potential
        self.nb = None
        for f in factors:
            f.cluster = self
        self.id = next(self.id_counter)

    def __lt__(self, other):
        return self.id < other.id

    @property
    def log_potential_fun(self):
        factor0 = next(iter(self.factors))
        return factor0.log_potential_fun

    @property
    def domain_type(self):
        factor0 = next(iter(self.factors))
        return factor0.domain_type

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def update_nb(self):
        f = next(iter(self.factors))
        self.nb = tuple(map(self.get_cluster, f.nb))

    def split_by_structure(self):
        clusters = dict()
        for f in sorted(self.factors):
            signature = tuple(sorted(map(self.get_cluster, f.nb))) \
                if f.potential.symmetric else tuple(map(self.get_cluster, f.nb))
            if signature in clusters:
                clusters[signature].add(f)
            else:
                clusters[signature] = {f}

        res = set()
        i = iter(clusters)

        # reuse THIS instance
        self.factors = clusters[next(i)]
        res.add(self)

        for _ in range(1, len(clusters)):
            res.add(SuperF(clusters[next(i)]))

        return res


class CompressedGraph:
    # color passing algorithm for compressing graph

    def __init__(self, graph):
        self.g = graph
        self.rvs = set()
        self.factors = set()

    def init_cluster(self):
        self.rvs.clear()
        self.factors.clear()

        # group rvs according to domain
        color_table = dict()
        for rv in self.g.rvs:
            if rv.domain in color_table:
                color_table[rv.domain].add(rv)
            else:
                color_table[rv.domain] = {rv}
        for _, cluster in color_table.items():
            self.rvs.add(SuperRV(cluster))

        # group factors according to potential
        color_table.clear()
        for f in self.g.factors:
            if f.potential in color_table:
                color_table[f.potential].add(f)
            else:
                color_table[f.potential] = {f}
        for _, cluster in color_table.items():
            self.factors.add(SuperF(cluster))

    def split_rvs(self):
        temp = set()
        for rv in self.rvs:
            temp |= rv.split_by_structure()
        self.rvs = temp

    def split_factors(self):
        temp = set()
        for f in self.factors:
            temp |= f.split_by_structure()
        self.factors = temp

    def run(self):
        self.init_cluster()

        prev_rvs_num = -1
        while prev_rvs_num != len(self.rvs):
            prev_rvs_num = len(self.rvs)
            self.split_factors()
            self.split_rvs()

        for rv in self.rvs:
            rv.update_nb()
        for f in self.factors:
            f.update_nb()

    def init_rv_indices(self):
        """
        Get lists of disc/cont rvs, and build indices. These (along with the .rb attributes of rv/f) should be the only
        pieces of information used by OSI.
        :return:
        """
        self.rvs_list = sorted(self.rvs)
        self.factors_list = sorted(self.factors)
        Vd = [rv for rv in self.rvs_list if rv.domain_type == 'd']  # list of of discrete rvs
        Vc = [rv for rv in self.rvs_list if rv.domain_type == 'c']  # list of cont rvs
        Vd_idx = {n: i for (i, n) in enumerate(Vd)}
        Vc_idx = {n: i for (i, n) in enumerate(Vc)}

        self.Vd = Vd
        self.Vc = Vc
        self.Nc = len(Vc)
        self.Nd = len(Vd)
        self.Vc_idx = Vc_idx
        self.Vd_idx = Vd_idx
