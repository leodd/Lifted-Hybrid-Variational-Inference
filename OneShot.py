from Graph import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import sqrt, pi, e, log
from itertools import product


class OneShot:
    var_threshold = 0.1

    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = g
        self.init_sharing_count()

        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quad_x, self.quad_w = hermgauss(self.T)
        self.quad_w /= sqrt(pi)

        self.w_tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta_tau = dict()
        self.eta = dict()  # key=rv, value={continuous eta: [, [mu, var]], discrete eta: [k, d]}

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
    def softmax(x, axis=0):
        res = e ** x
        if axis == 0:
            return res / np.sum(res, 0)
        else:
            return res / np.sum(res, 1)[:, np.newaxis]

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

    def gradient_w_tau(self):
        g_w = np.zeros(self.K)

        for k in range(self.K):
            for f in self.g.factors:
                if len(f.nb) == 1:
                    def f_w(x): return log(f.potential.get(x)) - (1 - f.nb[0].N) * log(self.rvs_belief(x, f.nb))
                else:
                    def f_w(x): return log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))

                args = list()
                for rv in f.nb:
                    if rv.domain.continuous:
                        args.append((True, self.eta[rv][k]))
                    else:
                        args.append((False, (rv.domain.values, self.eta[rv][k])))

                g_w[k] -= self.expectation(f_w, *args)

        return self.w * (g_w - np.sum(g_w * self.w))

    def gradient_mu_var(self, rv):
        g_mu_var = np.zeros((self.K, 2))
        eta = self.eta[rv]

        for k in range(self.K):
            for f in rv.nb:
                idx = f.nb.index(rv)
                if len(f.nb) == 1:
                    def f_mu(x): return (log(f.potential.get(x)) - (1 - f.nb[0].N) * log(self.rvs_belief(x, f.nb))) * \
                                        (x[idx] - eta[k][0]) ** 2

                    def f_var(x): return (log(f.potential.get(x)) - (1 - f.nb[0].N) * log(self.rvs_belief(x, f.nb))) * \
                                         ((x[idx] - eta[k][0]) ** 2 / eta[k][1] - 1)
                else:
                    def f_mu(x): return (log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))) * \
                                        (x[idx] - eta[k][0]) ** 2

                    def f_var(x): return (log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))) * \
                                         ((x[idx] - eta[k][0]) ** 2 / eta[k][1] - 1)

                args = list()
                for rv_ in f.nb:
                    if rv.domain.continuous:
                        args.append((True, self.eta[rv_][k]))
                    else:
                        args.append((False, (rv.domain.values, self.eta[rv_][k])))

                g_mu_var[k, 0] -= self.expectation(f_mu, *args) / eta[k][1]
                g_mu_var[k, 1] -= self.expectation(f_var, *args) * 0.5 / eta[k][1]

        return g_mu_var * self.w[:, np.newaxis]

    def gradient_category_tau(self, rv):
        g_c = np.zeros((self.K, len(rv.domain.values)))
        eta = self.eta[rv]

        for k in range(self.K):
            for f in rv.nb:
                args = list()
                for rv_ in f.nb:
                    if rv_ is not rv:
                        if rv.domain.continuous:
                            args.append((True, self.eta[rv_][k]))
                        else:
                            args.append((False, (rv.domain.values, self.eta[rv_][k])))

                idx = f.nb.index(rv)
                for d, (xi, v) in enumerate(zip(rv.domain.values, eta[k])):
                    if len(f.nb) == 1:
                        g_c[k, d] -= log(f.potential.get((xi,))) - (1 - f.nb[0].N) * log(self.rvs_belief((xi,), f.nb))
                    else:
                        def f_c(x):
                            new_x = x[:idx] + (xi,) + x[idx:]
                            return log(f.potential.get(new_x)) - log(self.rvs_belief(new_x, f.nb))

                        g_c[k, d] -= self.expectation(f_c, *args)

        g_c = g_c * self.w[:, np.newaxis]

        return eta * (g_c - np.sum(g_c * eta, 1)[:, np.newaxis])

    def rvs_belief(self, x, rvs):
        b = np.copy(self.w)

        for i, rv in enumerate(rvs):
            eta = self.eta[rv]

            if rv.domain.continuous:
                for k in range(self.K):
                    b[k] *= self.norm_pdf(x[i], eta[k])
            else:
                d = rv.domain.values.index(x[i])
                for k in range(self.K):
                    b[k] *= eta[k, d]

        return np.sum(b)

    def belief(self, x, rv):
        return self.rvs_belief((x,), (rv,))

    def run(self, iteration=100, lr=0.1):
        # initiate parameters
        self.w_tau = np.zeros(self.K)
        self.eta, self.eta_tau = dict(), dict()
        for rv in self.g.rvs:
            if rv.domain.continuous:
                self.eta[rv] = np.ones((self.K, 2))
            else:
                self.eta_tau[rv] = np.zeros((self.K, len(rv.domain.values)))

        # update w and categorical distribution
        self.w = self.softmax(self.w_tau)
        for rv, table in self.eta_tau.items():
            self.eta[rv] = self.softmax(table, 1)

        # Bethe iteration
        for itr in range(iteration):
            # compute gradient
            w_tau_g = self.gradient_w_tau() * lr
            eta_g = dict()
            eta_tau_g = dict()
            for rv in self.g.rvs:
                if rv.domain.continuous:
                    eta_g[rv] = self.gradient_mu_var(rv) * lr
                else:
                    eta_tau_g[rv] = self.gradient_category_tau(rv) * lr

            # update parameters
            self.w_tau = self.w_tau - w_tau_g
            self.w = self.softmax(self.w_tau)
            for rv in self.g.rvs:
                if rv.domain.continuous:
                    table = self.eta[rv] - eta_g[rv]
                    table[:, 1] = np.clip(table[:, 1], a_min=self.var_threshold, a_max=np.inf)
                    self.eta[rv] = table
                else:
                    table = self.eta_tau[rv] - eta_tau_g[rv]
                    self.eta_tau[rv] = table
                    self.eta[rv] = self.softmax(table, 1)
