import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from math import sqrt, pi, e, log
from itertools import product
from MLNPotential import MLNPotential, MLNHardPotential
import time
from utils import log_likelihood


class HybridMaxWalkSAT:
    def __init__(self, g):
        self.g = g
        self.best_assignment = None
        self.best_score = None

    def random_assignment(self):
        assignment = dict()
        for rv in self.g.rvs:
            if rv.value is not None:
                assignment[rv] = rv.value
            else:
                if rv.domain.continuous:
                    assignment[rv] = np.random.uniform(rv.domain.values[0], rv.domain.values[1])
                else:
                    assignment[rv] = np.random.choice(rv.domain.values)

        return assignment

    def score(self, assignment):
        score = 0
        for f in self.g.factors:
            parameters = [assignment[rv] for rv in f.nb]
            value = f.potential.get(parameters)
            if value == 0:
                score -= 700
            else:
                score += log(value)

        return score

    @staticmethod
    def local_score(rvs, assignment):
        fs = set()
        for rv in rvs:
            fs |= set(rv.nb)

        score = 0
        for f in fs:
            parameters = [assignment[rv] for rv in f.nb]
            value = f.potential.get(parameters)
            if value == 0:
                score -= 700
            else:
                score += log(value)

        return score

    def discrete_and_numeric_factors(self):
        numeric_factors = set()

        for rv in self.g.rvs:
            if rv.domain.continuous:
                numeric_factors |= set(rv.nb)

        discrete_factors = set(self.g.factors) - numeric_factors

        return numeric_factors, discrete_factors

    @staticmethod
    def prune_factors_without_latent_variables(factors):
        res = set()
        for f in factors:
            for rv in f.nb:
                if rv.value is None:
                    res.add(f)
                    break

        return res

    @staticmethod
    def unsatisfied_factors(assignment, discrete_factors):
        res_hard = list()
        res_soft = list()
        for f in discrete_factors:
            parameters = [assignment[rv] for rv in f.nb]
            value = f.potential.get(parameters)
            if type(f.potential) == MLNHardPotential:
                if value == 0:
                    res_hard.append(f)
            elif type(f.potential) == MLNPotential:
                if value == 1:
                    res_soft.append(f)

        return res_hard, res_soft

    @staticmethod
    def random_factor(unsatisfied_factors, numeric_factors):
        p = len(unsatisfied_factors) / (len(unsatisfied_factors) + len(numeric_factors))
        if np.random.rand() < p:
            return np.random.choice(list(unsatisfied_factors))
        else:
            return np.random.choice(list(numeric_factors))

    @staticmethod
    def argmax_rv_wrt_factor(f, rv, assignment):
        objective = lambda x: -f.potential.get([x[0] if rv_ is rv else assignment[rv_] for rv_ in f.nb])
        initial_guess = np.array([assignment[rv]])

        res = minimize(objective, initial_guess, method='L-BFGS-B', options={'disp': False})

        return res.x[0]

    @staticmethod
    def argmax_rvs_wrt_score(rvs, assignment):
        fs = set()
        x_idx = dict()
        counter = 0

        for rv in rvs:
            fs |= set(rv.nb)

            if rv not in x_idx:
                x_idx[rv] = counter
                counter += 1

        def neg_score(x):
            score = 0
            for f in fs:
                parameters = [x[x_idx[rv_]] if rv_ in rvs else assignment[rv_] for rv_ in f.nb]
                value = f.potential.get(parameters)
                if value == 0:
                    score += 700
                else:
                    score -= log(value)

            return score

        initial_guess = np.array([assignment[rv_] for rv_ in rvs])

        res = minimize(neg_score, initial_guess, method='L-BFGS-B', options={'disp': False})

        return {rv_: res.x[x_idx[rv_]] for rv_ in rvs}

    @staticmethod
    def argmax_discrete_rv_wrt_score(rv, assignment):
        def neg_score(x):
            score = 0
            for f in rv.nb:
                parameters = [x if rv_ is rv else assignment[rv_] for rv_ in f.nb]
                value = f.potential.get(parameters)
                if value == 0:
                    score += 700
                else:
                    score -= log(value)

            return score

        res = {x: neg_score(x) for x in rv.domain.values}

        return max(res.keys(), key=lambda k: res[k])

    def argmax_numeric_term_wrt_score(self, f, assignment):
        params = list()
        x_idx = dict()
        counter = 0
        numeric_rvs = set()

        for rv in f.nb:
            if rv.value is not None:
                continue
            if rv.domain.continuous:
                numeric_rvs.add(rv)
            elif rv not in x_idx:
                x_idx[rv] = counter
                counter += 1
                params.append(rv.domain.values)

        best_assignment = assignment
        best_score = self.local_score(f.nb, assignment)

        if len(params) == 0:
            if len(numeric_rvs) > 0:
                numeric_assignment = self.argmax_rvs_wrt_score(numeric_rvs, best_assignment)

                for rv in numeric_rvs:
                    best_assignment[rv] = numeric_assignment[rv]

                best_score = self.local_score(f.nb, assignment)
        else:
            for xs in product(*params):
                current_assignment = assignment.copy()
                for rv, idx in x_idx.items():
                    current_assignment[rv] = xs[idx]

                if len(numeric_rvs) > 0:
                    numeric_assignment = self.argmax_rvs_wrt_score(numeric_rvs, current_assignment)

                    for rv in numeric_rvs:
                        current_assignment[rv] = numeric_assignment[rv]

                score = self.local_score(f.nb, assignment)
                if score > best_score:
                    best_assignment = current_assignment
                    best_score = score

        return best_assignment

    def run(self, max_tries=100, max_flips=1000, epsilon=0.9, noise_std=1, is_log=True):
        if is_log:
            self.time_log = list()
            total_time = 0

        numeric_factors, discrete_factors = self.discrete_and_numeric_factors()
        numeric_factors = self.prune_factors_without_latent_variables(numeric_factors)
        discrete_factors = self.prune_factors_without_latent_variables(discrete_factors)

        self.best_assignment = None
        self.best_score = -np.Inf

        for i_try in range(max_tries):
            assignment = self.random_assignment()

            for i_flip in range(max_flips):
                start_time = time.process_time()

                score = self.score(assignment)
                if score > self.best_score:
                    self.best_assignment = assignment.copy()
                    self.best_score = score
                    # print(self.best_score, i_try, i_flip)

                unsatisfied_hard, unsatisfied_soft = self.unsatisfied_factors(assignment, discrete_factors)
                if len(unsatisfied_hard) > 0:
                    c = np.random.choice(unsatisfied_hard)
                else:
                    c = self.random_factor(unsatisfied_soft, numeric_factors)

                if np.random.rand() < epsilon:
                    rv = np.random.choice(list(filter(lambda rv_: rv_.value is None, c.nb)))

                    if rv.value is not None:
                        continue
                    if rv.domain.continuous:
                        assignment[rv] = self.argmax_rv_wrt_factor(c, rv, assignment) + \
                                         np.random.normal(scale=noise_std)
                    else:
                        assignment[rv] = 1 - assignment[rv]
                else:
                    rv_score_dict = dict()
                    rv_assignment_dict = dict()
                    for rv in c.nb:
                        if rv.value is not None:
                            continue

                        temp_assignment = assignment.copy()

                        if rv.domain.continuous:
                            rv_assignment_dict[rv] = self.argmax_rvs_wrt_score([rv], assignment)[rv]
                        else:
                            rv_assignment_dict[rv] = self.argmax_discrete_rv_wrt_score(rv, assignment)
                        temp_assignment[rv] = rv_assignment_dict[rv]
                        rv_score_dict[rv] = self.local_score(c.nb, temp_assignment)

                    if len(rv_score_dict) == 0:
                        continue

                    rv = max(rv_score_dict.keys(), key=lambda k: rv_score_dict[k])

                    if rv_score_dict[rv] > self.local_score(c.nb, assignment) or c in discrete_factors:
                        assignment[rv] = rv_assignment_dict[rv]
                    else:
                        assignment = self.argmax_numeric_term_wrt_score(c, assignment)

                if is_log:
                    current_time = time.process_time()
                    total_time += current_time - start_time
                    if self.score(assignment) > self.best_score:
                        map_res = dict()
                        for rv in self.g.rvs:
                            map_res[rv] = assignment[rv]
                        ll = log_likelihood(self.g, map_res)
                        print(ll, total_time)
                        self.time_log.append([total_time, ll])

    def map(self, rv):
        return self.best_assignment[rv]
