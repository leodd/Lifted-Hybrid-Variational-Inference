from Graph import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import sqrt, pi, e, log
from itertools import product


class OneShot:
    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = g
        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quad_x, self.quad_w = hermgauss(self.T)
        self.quad_w /= sqrt(pi)

        self.tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta = dict()  # continuous eta = (mu, var), discrete eta = [domain vector, value vector]

    def expectation(self, f, *args):  # arg = (is_continuous, eta)
        xs, ws = list(), list()

        for is_continuous, eta in args:
            if is_continuous:
                xs.append(sqrt(2 * eta[1]) * self.quad_x + eta[0])
                ws.append(self.quad_w)
            else:
                xs.append(eta[:, 0])
                ws.append(eta[:, 1])

        res = 0
        for x, w in zip(product(*xs), product(*ws)):
            res += np.prod(w) * f(x)

        return res

    def gradient_tau(self):
        gs = np.zeros(self.K)

        for k in range(self.K):
            g = 0
            for f in self.g.factors:
                rvs = f.nb
                if len(rvs) == 1:
                    eta = self.eta[rv]
                    g += self.expectation(
                        lambda x:
                    )


    def gradient_mu(self, rv):
        pass

    def gradient_var(self, rv):
        pass

    def gradient_category(self, rv):
        pass

    def softmax(self, X):
        res = e ** X
        return res / np.sum(res)

    def update_eta(self):
        pass

    def belief(self, rv, ):
        pass

    def factor_belief(self):
        pass

    def run(self, iteration=100):
        pass
