from abc import ABC, abstractmethod
from numpy import linspace
import numpy as np


class Domain:
    def __init__(self, values, continuous=False, integral_points=None):
        self.values = tuple(values)
        self.continuous = continuous
        if continuous:
            if integral_points is None:
                self.integral_points = linspace(values[0], values[1], 30)
            else:
                self.integral_points = integral_points

    # def __hash__(self):
    #     return hash((self.values, self.continuous))
    #
    # def __eq__(self, other):
    #     return (
    #         self.__class__ == other.__class__ and
    #         self.values == other.values and
    #         self.continuous == other.continuous
    #     )


class Potential(ABC):
    def __init__(self, symmetric=False):
        self.symmetric = symmetric

    @abstractmethod
    def get(self, parameters):
        pass


class RV:
    def __init__(self, domain, value=None, id=None):
        self.domain = domain
        self.value = value
        self.id = id
        self.nb = []
        self.belief_params_ = {}  # symbolic, used by tf
        self.belief_params = {}  # np arrs

    @property
    def domain_type(self):
        return 'c' if self.domain.continuous else 'd'

    @property
    def dstates(self):
        if self.domain_type == 'd':
            return len(self.domain.values)

    @property
    def values(self):  # numpy arrays for a node's all possible values, not tuple
        return np.array(self.domain.values)

    def __str__(self):
        return '{}rv #{}'.format(self.domain_type, self.id)


class F:
    def __init__(self, potential=None, log_potential=None, nb=None, id=None):
        self.potential = potential
        self.log_potential = log_potential
        if nb is None:
            self.nb = []
        else:
            self.nb = nb
        self.id = id

    @property
    def domain_type(self):
        rv_domain_types = [rv.domain_type for rv in self.nb]
        if all([t == 'd' for t in rv_domain_types]):
            type = 'd'  # disc
        elif all([t == 'c' for t in rv_domain_types]):
            type = 'c'  # cont
        elif self.nb:
            type = 'h'  # hybrid
        else:  # empty nb
            type = None
        return type

    def __str__(self):
        return '{}factor #{}'.format(self.domain_type, self.id)


class Graph:
    def __init__(self):
        self.rvs = set()
        self.factors = set()

    def init_nb(self):
        """
        Should be called after specifying self.factors, to update the neighbors of self.rvs
        :return:
        """
        for rv in self.rvs:
            rv.nb = []
        for f in self.factors:
            for rv in f.nb:
                rv.nb.append(f)

        # for convenience
        self.rv_neighbors = {rv: rv.nb for rv in self.rvs}
        self.factor_neighbors = {f: f.nb for f in self.factors}

    def init_rv_indices(self):
        """

        :return:
        """
        # Vd = [rv.id for rv in self.rvs if rv.domain_type == 'd']  # id of discrete rvs
        # Vc = [rv.id for rv in self.rvs if rv.domain_type == 'c']
        Vd = [rv for rv in self.rvs if rv.domain_type == 'd']
        Vc = [rv for rv in self.rvs if rv.domain_type == 'c']
        Vd_idx = {n: i for (i, n) in enumerate(Vd)}
        Vc_idx = {n: i for (i, n) in enumerate(Vc)}

        self.Vd = Vd
        self.Vc = Vc
        self.Nc = len(Vc)
        self.Nd = len(Vd)
        self.Vc_idx = Vc_idx
        self.Vd_idx = Vd_idx

        self.rvs_dict = {rv.id: rv for rv in self.rvs}  # in case rv.id isn't the same as its order in self.rvs
