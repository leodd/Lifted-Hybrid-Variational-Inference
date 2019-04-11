from Graph import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import sqrt, pi, e, log
from itertools import product


class OneShot:
    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = g
        self.init_sharing_count()

        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quad_x, self.quad_w = hermgauss(self.T)
        self.quad_w /= sqrt(pi)

        self.tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta = dict()  # key=rv, value=list of eta
                           # continuous eta = (mu, var), discrete eta = array of values

    def init_sharing_count(self):
        for rv in self.g.rvs:
            rv.N = 0
        for f in self.g.factors:
            if len(f.nb) > 1:
                for rv in f.nb:
                    rv.N += 1

    @staticmethod
    def norm_pdf(x, eta):
        u = (x - eta[0]) / eta[1]
        y = e ** (-u * u * 0.5) / (2.506628274631 * eta[1])
        return y

    @staticmethod
    def softmax(X):
        res = e ** X
        return res / np.sum(res)

    def expectation(self, f, *args):  # arg = (is_continuous, eta), discrete eta = (domains, values)
        xs, ws = list(), list()

        for is_continuous, eta in args:
            if is_continuous:
                xs.append(sqrt(2 * eta[1]) * self.quad_x + eta[0])
                ws.append(self.quad_w)
            else:
                xs.append(eta[0])
                ws.append(eta[1])

        res = 0
        for x, w in zip(product(*xs), product(*ws)):
            res += np.prod(w) * f(x)

        return res

    def gradient_tau(self):
        g_w = np.zeros(self.K)

        for k in range(self.K):
            g = 0

            for f in self.g.factors:
                if len(f.nb) == 1:
                    rv = f.nb[0]
                    g += self.expectation()
                else:
                    g += self.expectation(
                        lambda x:
                    )

    def gradient_mu(self, rv):
        pass

    def gradient_var(self, rv):
        pass

    def gradient_category(self, rv):
        pass

    def update_eta(self):
        pass

    def belief(self, x, rv):
        b = 0
        eta = self.eta[rv]

        if rv.domain.continuous:
            for k in range(self.K):
                b += self.w[k] * self.norm_pdf(x, eta[k])
        else:
            idx = rv.domain.values.index(x)
            for k in range(self.K):
                b += self.w[k] * eta[k][idx]

        return b

    def factor_belief(self, x, f):
        b = np.copy(self.w)

        for i, rv in enumerate(f.nb):
            eta = self.eta[rv]

            if rv.domain.continuous:
                for k in range(self.K):
                    b[k] *= self.norm_pdf(x[i], eta[k])
            else:
                idx = rv.domain.values.index(x[i])
                for k in range(self.K):
                    b[k] *= eta[k][idx]

        return np.sum(b)

    def run(self, iteration=100):
        pass
