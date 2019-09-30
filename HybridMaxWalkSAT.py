import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from math import sqrt, pi, e, log
from itertools import product
from MLNPotential import MLNPotential, MLNHardPotential
import time


class HybridMaxWalkSAT:
    def __init__(self, g):
        self.g = g
        self.best_assignment = None
        self.best_score = None

    def random_assignment(self):
        assignment = dict()
        for rv in self.g.rvs:
            if rv.domain.continuous:
                assignment[rv] = np.random.uniform(rv.domain.values[0], rv.domain.values[1])
            else:
                assignment[rv] = np.random.choice(2)

        return assignment

    def score(self, assignment):
        score = 1
        for f in self.g.factors:
            parameters = [assignment[rv] for rv in f.nb]
            value = f.potential.get(parameters)
            if value == 0:
                return -np.Inf
            score += log(value)

    def discrete_and_numeric_factors(self):
        numeric_factors = set()

        for rv in self.g.rvs:
            if rv.domain.continuous:
                numeric_factors |= set(rv.nb)

        discrete_factors = set(self.g.factors) - numeric_factors

        return numeric_factors, discrete_factors

    @staticmethod
    def unsatisfied_factors(assignment, discrete_factors):
        res = list()
        for f in discrete_factors:
            parameters = [assignment[rv] for rv in f.nb]
            value = f.potential.get(parameters)
            if type(f.potential) == MLNHardPotential:
                if value == 0:
                    res.append(f)
            elif type(f.potential) == MLNPotential:
                if value == 1:
                    res.append(f)

        return res

    @staticmethod
    def random_factor(unsatisfied_factors, numeric_factors):
        p = len(unsatisfied_factors) / (len(unsatisfied_factors) + len(numeric_factors))
        if np.random.rand() < p:
            return np.random.choice(list(unsatisfied_factors))
        else:
            return np.random.choice(list(numeric_factors))

    @staticmethod
    def argmax_rv_wrt_factor(f, rv):
        rv_idx =

    def run(self, max_tries, max_flips, epsilon=0.5):
        numeric_factors, discrete_factors = self.discrete_and_numeric_factors()

        self.best_assignment = None
        self.best_score = -np.Inf

        for i_try in range(max_tries):
            assignment = self.random_assignment()
            score = self.score(assignment)
            unsatisfied_factors = self.unsatisfied_factors(assignment, discrete_factors)

            for i_flip in range(max_flips):
                if score > self.best_score:
                    self.best_assignment = assignment.copy()
                    self.best_score = score

                c = self.random_factor(unsatisfied_factors, numeric_factors)

                if np.random.rand() < epsilon:
                    rv = np.random.choice(c.nb)
                    if rv.domain.continuous:
                        assignment[rv] =
                    else:
                        assignment[rv] = 1 - assignment[rv]
