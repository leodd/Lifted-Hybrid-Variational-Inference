# A version of CompressedGraph that (hopefully) produces the same graph (with identical labeling) every time when run,
# which makes a bit easier to run/test algorithms with it.
# from collections import Counter
# from statistics import mean
# from random import uniform
# import numpy as np

import itertools
import functools


class SuperRV:
    id_counter = itertools.count()  # will assign unique numeric ids to instances of the class, auto-incrementing from 0

    def __init__(self, rvs, domain=None):
        self.rvs = rvs
        rv0 = next(iter(self.rvs))
        self.domain = rv0.domain if domain is None else domain
        self.domain_type = rv0.domain_type
        self.dstates = rv0.dstates
        self.values = rv0.values
        self.nb = None
        for rv in rvs:
            rv.cluster = self
        self.id = next(self.id_counter)

    def __lt__(self, other):
        return self.id < other.id

    @property
    def sharing_count(self):
        return len(self.rvs)

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
        factor0 = next(iter(factors))
        self.potential = factor0.potential
        self.log_potential_fun = factor0.log_potential_fun
        self.domain_type = factor0.domain_type
        self.nb_domain_types = factor0.nb_domain_types
        self.nb = None
        for f in factors:
            f.cluster = self
        self.id = next(self.id_counter)

    def __lt__(self, other):
        return self.id < other.id

    @property
    def sharing_count(self):
        return len(self.factors)

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


class CompressedGraphSorted:
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

    @property
    @functools.lru_cache()
    def rvs_list(self):
        """
        A sorted list of self.rvs
        :return:
        """
        return list(sorted(self.rvs))

    @property
    @functools.lru_cache()
    def factors_list(self):
        """
        A sorted list of self.factors
        :return:
        """
        return list(sorted(self.factors))

    def init_rv_indices(self):
        """
        Get lists of disc/cont rvs, and build indices. These (along with the .rb attributes of rv/f) should be the only
        pieces of information used by OSI.
        :return:
        """
        Vd = [rv for rv in self.rvs_list if rv.domain_type[0] == 'd']  # list of of discrete rvs
        Vc = [rv for rv in self.rvs_list if rv.domain_type[0] == 'c']  # list of cont rvs
        Vd_idx = {n: i for (i, n) in enumerate(Vd)}
        Vc_idx = {n: i for (i, n) in enumerate(Vc)}

        self.Vd = Vd
        self.Vc = Vc
        self.Nc = len(Vc)
        self.Nd = len(Vd)
        self.Vc_idx = Vc_idx
        self.Vd_idx = Vd_idx
