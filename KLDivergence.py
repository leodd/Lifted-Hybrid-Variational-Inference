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


if __name__ == '__main__':
    from Graph import Domain

    domain = Domain([-30, 30], continuous=True)


    def norm_pdf(x, mu, sig):
        u = (x - mu) / sig
        y = np.exp(-u * u * 0.5) / (2.506628274631 * sig)
        return y


    res = KL(
        lambda x: norm_pdf(x, 20, 1),
        lambda x: norm_pdf(x, 0, 2),
        domain
    )

    print(res)
