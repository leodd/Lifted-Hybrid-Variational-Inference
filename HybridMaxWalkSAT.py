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
        score = 0
        for

    def run(self, max_tries, max_flips):
        self.best_assignment = None
        self.best_score = -np.Inf

        for i_try in range(max_tries):
            assignment = self.random_assignment()
