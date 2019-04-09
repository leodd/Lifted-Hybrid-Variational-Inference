from Graph import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import sqrt, pi, e, log


class OneShot:
    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = g
        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quadrature_points = hermgauss(self.T)

        self.tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta = dict()

    @staticmethod
    def gaussian_product(*gaussian):
        # input a list of gaussian's mean and variance
        # output the product distribution's mean and variance
        mu, var = 0, 0
        for eta in gaussian:
            mu_, var_ = eta
            var += var_ ** -1
            mu += var_ ** -1 * mu_
        var = var ** -1
        mu = var * mu
        return mu, var

    def gaussian_expectation(self, f, *eta):
        mu, var = eta
        c = sqrt(2 * var)

        res = 0

        x = self.quadrature_points[0]
        w = self.quadrature_points[1]
        for t in range(self.T):
            res = w[t] * f(c * x[t] + mu)

        return res / sqrt(pi)

    def category_expectation(self, f, *p):
        pass

    def gradient_tau(self):
        g_w = np.zeros(self.K)

        for k in range(self.K):
            g = 0
            for rv in self.g.rvs:
                if rv.domain.continuous:
                    eta = self.eta[rv]



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

    def edge_belief(self):
        pass

    def run(self, iteration=100):
        pass
