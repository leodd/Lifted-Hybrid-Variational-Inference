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


def get_log_potential_fun_from_MLNPotential(potential):
    """
    Extract the log potential function from a given MLNPotential object
    :param potential: an instance of MLNPotential
    :return:
    """
    f = weighted_feature_fun(potential.formula, potential.w)
    f.potential_obj = potential  # ref for convenience
    return f


def outer_prod_einsum_equation(ndim, mats=False):
    """
    Get the einsum equation for n-dimensional outer/tensor product
    :param ndim:
    :param mats: if True, will get the tensor-product of matrices with shared 0th dimension; otherwise will get the
    tensor product of vectors (default)
    :return:
    """
    # TODO: allow arbitrary number of shared first p dimensions (currently p=1 for mats), like with expand_dims_for_grid
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


def expand_dims_for_grid(arrs, first_ndims_to_keep=0):
    """
    https://stackoverflow.com/a/22778484/4115369
    Extended to tensors with arbitrary number of dimensions (but same rank); e.g., if arrs are matrices of shapes
    K x V1, K x V2, ..., K x Vn, then use first_ndims_to_keep=1, and arrs will be reshaped to be K x V1 x 1 x ... x 1,
    K x 1 x V2 x ... x 1, ..., K x 1 x 1 x ... x Vn, each shape of length (n+1)
    :param arrs:
    :return:
    """
    # return [x[(None,) * i + (slice(None),) + (None,) * (len(arrs) - i - 1)] for i, x in enumerate(arrs)]

    shared_first_ndims_slices = (slice(None),) * first_ndims_to_keep  # these dims will not be expanded
    return [x[shared_first_ndims_slices + (None,) * i + (slice(None),) + (None,) * (len(arrs) - i - 1)] for i, x in
            enumerate(arrs)]


def eval_fun_grid(fun, arrs, sep_args=False):
    """
    Evaluate a function R^n -> R on the tensor (outer) product of vectors [v1, v2, ..., vn];
    Extended to tensors (thought of containers of vectors) for efficient simultaneously evaluations, e.g., if arrs have
    shapes [K x v1, K x v2, ..., K x vn], the result will be of shape K x v1 x v2 x ... x vn, and should be equivalent
    to concatenating the results of evaluating on K ndgrids each of shape v1 x v2 ... x vn (such that the kth grid is
    formed by taking the kth rows of all the arrs).
    :param fun:
    :param arrs: iterable (tuple/list) of tf/np arrays
    :param sep_args: whether fun takes iterable (instead of *args) as arguments, i.e., f(xyz) vs f(x,y,z)
    :return:
    """
    arrs_shapes_except_last = [a.shape[:-1] for a in arrs]
    arrs_shapes_except_last = [tuple(s.as_list()) if isinstance(s, tf.TensorShape) else s
                               for s in arrs_shapes_except_last]  # convert tf tensor shapes (np shapes already tuples)
    assert len(set(arrs_shapes_except_last)) == 1, 'Shapes of input tensors can only differ in the last dimension!'
    first_ndims_to_keep = len(arrs_shapes_except_last[0])  # form grid based on the last dimension
    expanded_arrs = expand_dims_for_grid(arrs, first_ndims_to_keep)

    if sep_args:
        res = fun(*expanded_arrs)
    else:
        res = fun(expanded_arrs)  # should evaluate on the Cartesian product (ndgrid) of axes by broadcasting
    return res
