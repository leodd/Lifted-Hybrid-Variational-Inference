from Graph import *


class OneShot:
    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = g
        self.K = num_mixtures
        self.quadrature_points = self.compute_quadrature_points(num_quadrature_points)

    def compute_quadrature_points(self, n):
        return 1

    def gaussian_expectation(self):
        pass

    def gradient_tau(self):
        pass

    def gradient_mu_sig(self):
        pass

    def gradient_category(self):
        pass

    def run(self, iteration=100):
        pass
