import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from math import sqrt, pi, e, log
from itertools import product
import time


class HybridMaxWalkSAT:
    def __init__(self, g):
        self.g = g
        self.best_assignment = None
        self.best_score = None

    def random_assignment(self):
        assignment = dict()
        for rv in self.g:
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

    def list_unsatisfied_factor(self, assignment):
        res = list()
        for f in self.g.factors:

            parameters = [assignment[rv] for rv in f.nb]
            value = f.potential.get(parameters)


    def run(self, max_tries, max_flips):
        self.best_assignment = None
        self.best_score = -np.Inf

        for i_try in range(max_tries):
            assignment = self.random_assignment()
            score = self.score(assignment)
            for i_flip in range(max_flips):
                if score > self.best_score:
                    self.best_assignment = assignment.copy()
                    self.best_score = score

                c =
