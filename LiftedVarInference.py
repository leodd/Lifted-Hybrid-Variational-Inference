from CompressedGraphWithObs import CompressedGraph
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import fminbound
from math import sqrt, pi, e, log
from itertools import product


class VarInference:
    var_threshold = 1

    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = CompressedGraph(g)
        self.g.run()
        self.init_rv()

        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quad_x, self.quad_w = hermgauss(self.T)
        self.quad_w /= sqrt(pi)

        self.w_tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta_tau = dict()
        self.eta = dict()  # key=rv, value={continuous eta: [, [mu, var]], discrete eta: [k, d]}

    def init_rv(self):
        for rv in self.g.rvs:
            rv.N = 0
            rv.node_factor = None
        for f in self.g.factors:
            if len(f.nb) > 1:
                for rv in f.nb:
                    rv.N += rv.count[f]
            else:
                f.nb[0].node_factor = f

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

        for rv in self.g.rvs:
            phi = rv.node_factor
            if phi is None:
                def f_w(x):
                    return (rv.N - 1) * log(self.rvs_belief(x, [rv]))
            else:
                def f_w(x):
                    return log(phi.potential.get(x)) - (1 - rv.N) * log(self.rvs_belief(x, [rv]))

            for k in range(self.K):
                if rv.value is not None:
                    arg = (False, ((rv.value,), (1,)))
                elif rv.domain.continuous:
                    arg = (True, self.eta[rv][k])
                else:
                    arg = (False, (rv.domain.values, self.eta[rv][k]))

                g_w[k] -= len(rv.rvs) * self.expectation(f_w, arg)

        for f in self.g.factors:
            if len(f.nb) > 1:
                def f_w(x):
                    return log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))

                for k in range(self.K):
                    args = list()
                    for rv in f.nb:
                        if rv.value is not None:
                            args.append((False, ((rv.value,), (1,))))
                        elif rv.domain.continuous:
                            args.append((True, self.eta[rv][k]))
                        else:
                            args.append((False, (rv.domain.values, self.eta[rv][k])))

                    g_w[k] -= len(f.factors) * self.expectation(f_w, *args)

        return self.w * (g_w - np.sum(g_w * self.w))

    def gradient_mu_var(self, rv):
        g_mu_var = np.zeros((self.K, 2))
        eta = self.eta[rv]

        phi = rv.node_factor

        for k in range(self.K):
            if phi is None:
                def f_mu(x):
                    return ((rv.N - 1) * log(self.rvs_belief(x, [rv]))) * (x[0] - eta[k][0])

                def f_var(x):
                    return ((rv.N - 1) * log(self.rvs_belief(x, [rv]))) * ((x[0] - eta[k][0]) ** 2 - eta[k][1])
            else:
                def f_mu(x):
                    return (log(f.potential.get(x)) - (1 - rv.N) * log(self.rvs_belief(x, phi.nb))) * \
                           (x[0] - eta[k][0]) ** 2

                def f_var(x):
                    return (log(f.potential.get(x)) - (1 - rv.N) * log(self.rvs_belief(x, phi.nb))) * \
                           ((x[0] - eta[k][0]) ** 2 - eta[k][1])

            arg = (True, self.eta[rv][k])

            g_mu_var[k, 0] -= self.expectation(f_mu, arg) / eta[k][1]
            g_mu_var[k, 1] -= self.expectation(f_var, arg) / (2 * eta[k][1] ** 2)

        for f in rv.nb:
            if len(f.nb) > 1:
                count = rv.count[f]
                idx = f.nb.index(rv)
                for k in range(self.K):
                    def f_mu(x):
                        return (log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))) * \
                               (x[idx] - eta[k][0])

                    def f_var(x):
                        return (log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))) * \
                               ((x[idx] - eta[k][0]) ** 2 - eta[k][1])

                    args = list()
                    for rv_ in f.nb:
                        if rv_.value is not None:
                            args.append((False, ((rv_.value,), (1,))))
                        elif rv_.domain.continuous:
                            args.append((True, self.eta[rv_][k]))
                        else:
                            args.append((False, (rv_.domain.values, self.eta[rv_][k])))

                    g_mu_var[k, 0] -= count * self.expectation(f_mu, *args) / eta[k][1]
                    g_mu_var[k, 1] -= count * self.expectation(f_var, *args) / (2 * eta[k][1] ** 2)

        return g_mu_var

    def gradient_category_tau(self, rv):
        g_c = np.zeros((self.K, len(rv.domain.values)))
        eta = self.eta[rv]

        for k in range(self.K):
            for d, (xi, v) in enumerate(zip(rv.domain.values, eta[k])):
                phi = rv.node_factor
                if phi is None:
                    g_c[k, d] -= (rv.N - 1) * log(self.rvs_belief([xi], [rv]))
                else:
                    g_c[k, d] -= log(phi.potential.get((xi,))) - (1 - rv.N) * log(self.rvs_belief([xi], [rv]))

            for f in rv.nb:
                count = rv.count[f]
                idx = f.nb.index(rv)
                args = list()
                for i, rv_ in enumerate(f.nb):
                    if i is not idx:
                        if rv_.value is not None:
                            args.append((False, ((rv_.value,), (1,))))
                        elif rv.domain.continuous:
                            args.append((True, self.eta[rv_][k]))
                        else:
                            args.append((False, (rv.domain.values, self.eta[rv_][k])))

                for d, xi in enumerate(rv.domain.values):
                    if len(f.nb) > 1:
                        def f_c(x):
                            new_x = x[:idx] + (xi,) + x[idx:]
                            return log(f.potential.get(new_x)) - log(self.rvs_belief(new_x, f.nb))

                        g_c[k, d] -= count * self.expectation(f_c, *args)

        return eta * (g_c - np.sum(g_c * eta, 1)[:, np.newaxis])

    def free_energy(self):
        energy = 0

        for rv in self.g.rvs:
            phi = rv.node_factor
            if phi is None:
                def f_bfe(x):
                    return (rv.N - 1) * log(self.rvs_belief(x, [rv]))
            else:
                def f_bfe(x):
                    return log(phi.potential.get(x)) - (1 - rv.N) * log(self.rvs_belief(x, [rv]))

            for k in range(self.K):
                if rv.value is not None:
                    arg = (False, ((rv.value,), (1,)))
                elif rv.domain.continuous:
                    arg = (True, self.eta[rv][k])
                else:
                    arg = (False, (rv.domain.values, self.eta[rv][k]))

                energy -= len(rv.rvs) * self.w[k] * self.expectation(f_bfe, arg)

        for f in self.g.factors:
            if len(f.nb) > 1:
                def f_bfe(x):
                    return log(f.potential.get(x)) - log(self.rvs_belief(x, f.nb))

                for k in range(self.K):
                    args = list()
                    for rv in f.nb:
                        if rv.value is not None:
                            args.append((False, ((rv.value,), (1,))))
                        elif rv.domain.continuous:
                            args.append((True, self.eta[rv][k]))
                        else:
                            args.append((False, (rv.domain.values, self.eta[rv][k])))

                    energy -= len(f.factors) * self.w[k] * self.expectation(f_bfe, *args)

        return energy

    def rvs_belief(self, x, rvs):
        b = np.copy(self.w)

        for i, rv in enumerate(rvs):
            if rv.value is not None:
                if x[i] != rv.value:
                    return 0
            elif rv.domain.continuous:
                eta = self.eta[rv]
                for k in range(self.K):
                    b[k] *= self.norm_pdf(x[i], eta[k])
            else:
                eta = self.eta[rv]
                d = rv.domain.values.index(x[i])
                for k in range(self.K):
                    b[k] *= eta[k, d]

        return np.sum(b)

    def init_param(self):
        self.w_tau = np.zeros(self.K)
        self.eta, self.eta_tau = dict(), dict()
        for rv in self.g.rvs:
            if rv.value is not None:
                continue
            elif rv.domain.continuous:
                temp = np.ones((self.K, 2))
                temp[:, 0] = np.random.rand(self.K) * 3 - 1.5
                self.eta[rv] = temp
            else:
                self.eta_tau[rv] = np.random.rand(self.K, len(rv.domain.values)) * 10

        # update w and categorical distribution
        self.w = self.softmax(self.w_tau)
        for rv, table in self.eta_tau.items():
            self.eta[rv] = self.softmax(table, 1)

    def run(self, iteration=100, lr=0.1):
        # initiate parameters
        self.init_param()

        # Bethe iteration
        for itr in range(iteration):
            # compute gradient
            w_tau_g = self.gradient_w_tau() * lr
            eta_g = dict()
            eta_tau_g = dict()
            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    eta_g[rv] = self.gradient_mu_var(rv) * lr
                else:
                    eta_tau_g[rv] = self.gradient_category_tau(rv) * lr

            # update parameters
            self.w_tau = self.w_tau - w_tau_g
            self.w = self.softmax(self.w_tau)
            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    table = self.eta[rv] - eta_g[rv]
                    table[:, 1] = np.clip(table[:, 1], a_min=self.var_threshold, a_max=np.inf)
                    self.eta[rv] = table
                else:
                    table = self.eta_tau[rv] - eta_tau_g[rv]
                    self.eta_tau[rv] = table
                    self.eta[rv] = self.softmax(table, 1)

            # print('iteration:', itr)
            print(self.free_energy())

    def belief(self, x, rv):
        return self.rvs_belief((x,), (rv.cluster,))

    def map(self, rv):
        if rv.value is None:
            if rv.domain.continuous:
                # p = dict()
                # for x in self.eta[rv.cluster][:, 0]:
                #     p[x] = self.belief(x, rv)
                # res = max(p.keys(), key=(lambda k: p[k]))

                res = fminbound(
                    lambda val: -self.belief(val, rv),
                    rv.domain.values[0], rv.domain.values[1],
                    disp=False
                )
            else:
                p = dict()
                for x in rv.domain.values:
                    p[x] = self.belief(x, rv)
                res = max(p.keys(), key=(lambda k: p[k]))

            return res
        else:
            return rv.value
