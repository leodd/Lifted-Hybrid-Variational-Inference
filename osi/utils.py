import tensorflow as tf
import numpy as np


def set_path():  # to facilitate importing from pardir
    import sys
    sys.path.append('..')


def weighted_feature_fun(feature_fun, weight):
    # return lambda vs: weight * feature_fun(vs)
    def wf(args):
        return weight * feature_fun(args)

    return wf


def outer_prod_einsum_equation(ndim, mats=False):
    """
    Get the einsum equation for n-dimensional outer/tensor product
    :param ndim:
    :param mats: if True, will get the tensor-product of matrices with shared 0th dimension; otherwise will get the
    tensor product of vectors (default)
    :return:
    """
    # TODO: allow arbitrary number of shared first p dimensions (currently p=1 for mats)
    indices = 'ijklmnopqrstuvwxyz'
    assert ndim < len(indices)
    indices = indices[:ndim]
    if mats:
        lhs = ','.join(['a' + c for c in list(indices)])  # "ai,aj,ak,..."
        rhs = 'a' + indices  # "aijk..."
    else:
        lhs = ','.join(indices)  # "i,j,k,..."
        rhs = indices  # "ijk..."
    return lhs + '->' + rhs


def expand_dims_for_grid(arrs):
    """
    https://stackoverflow.com/a/22778484/4115369
    :param arrs:
    :return:
    """
    # TODO: extend to matrices of shapes K x V1, K x V2, ..., K x Vn
    return [x[(None,) * i + (slice(None),) + (None,) * (len(arrs) - i - 1)] for i, x in enumerate(arrs)]


def eval_fun_grid(fun, arrs, sep_args=False):
    """
    Evaluate a function R^n -> R on the tensor (outer) product of vectors [v1, v2, ..., vn]
    :param fun:
    :param arrs: iterable (tuple/list) of tf/np arrays
    :param sep_args: whether fun takes iterable (instead of *args) as arguments, i.e., f(xyz) vs f(x,y,z)
    :return:
    """
    expanded_arrs = expand_dims_for_grid(arrs)

    if sep_args:
        res = fun(*expanded_arrs)
    else:
        res = fun(expanded_arrs)  # should evaluate on the Cartesian product (ndgrid) of axes by broadcasting
    return res
