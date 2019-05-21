import numpy as np


def KL(q, p, domain):
    if domain.continuous:
        integral_points = domain.integral_points
        w = (domain.values[1] - domain.values[0]) / (len(integral_points) - 1)
    else:
        integral_points = domain.values
        w = 1

    res = 0
    for x in integral_points:
        qx, px = q(x) * w + 1e-8, p(x) * w + 1e-8
        res += qx * np.log(qx / px)

    return res
