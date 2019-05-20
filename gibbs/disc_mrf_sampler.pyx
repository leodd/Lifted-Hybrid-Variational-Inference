# from libcpp cimport bool as bool_t
# ctypedef bool_t sample_type_t
# from libcpp.numeric cimport inner_product # doesn't work!
from libc.math cimport exp, log
from libc.math cimport fabs
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY as inf


# # for the type of numeric data; float32 is generally fine and memory efficient
# value_type = np.float64  # runtime type, for dynamically creating numpy arrays etc.
# ctypedef np.float64_t value_type_t  # '_t' suffix means compile-time type; np.float32 ~ C float, np.float64 ~ C double

int_type = np.int32
ctypedef np.int32_t int_type_t

# from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

# https://stackoverflow.com/questions/16138090/correct-way-to-generate-random-numbers-in-cython
# http://pubs.opengroup.org/onlinepubs/7908799/xsh/drand48.html
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)
cdef extern from "time.h":    # more flexible to seed at python level
    long int time(int)


def seed(seed=None):
    """
    Seed the rng generator for this module; will call srand48
    :param seed: if None, use current time (in seconds) as seed
    :return:
    """
    if seed is None:
        srand48(time(0))  # seed rng generator; will give current time in seconds (pseudorandomness will be broken if same code run twice in < 1s)
    else:
        srand48(int(seed))


# cdef void softmax(vector[double]& vec, size_t vec_size):  # pass by reference not working somehow; C++11 move faster anyways
cdef vector[double] softmax(vector[double] vec, size_t vec_size):
    cdef size_t i
    cdef double m, sum=0, scale
    cdef vector[double] res = vector[double](vec_size)
    # find maximum value from input array
    m = -inf
    for i in range(vec_size):
        if vec[i] > m:
            m = vec[i]
    for i in range(vec_size):
        sum += exp(vec[i] - m)

    scale =  m + log(sum)
    for i in range(vec_size):
        res[i] = exp(vec[i] - scale)

    return res


cdef unsigned sample_categorical(vector[double] prob_vec, size_t vec_size):
    # sample in range(vec_size) according to a probability vector
    cdef double v, cum_sum=0
    cdef unsigned i=0
    v = drand48() # uniform in [0, 1)
    for i in range(vec_size):
        cum_sum += prob_vec[i]
        if v <= cum_sum:
            return i


# def gibbs_sample_one(list lpot_tables, vector[vector[uint]] scopes, vector[vector[uint]] nbr_factor_ids,
#                      vector[uint] dstates, long[:]x, unsigned its, int seed):
def gibbs_sample_one(list lpot_tables, vector[vector[uint]] scopes, vector[vector[uint]] nbr_factor_ids,
                     vector[uint] dstates, long[:]x, unsigned its):
    """
    Run Gibbs sampling for given number of iterations to obtain one sample; will modify x in place
    :param lpot_tables:
    :param scopes:
    :param nbr_factor_ids:
    :param dstates:
    :param x:
    :param its:
    :param seed:
    :return:
    """
    cdef size_t it, n, i, j, d, f, N=x.shape[0]
    cdef vector[double] lprobs, probs
    cdef vector[uint] scope
    # cdef np.ndarray[np.float_t] lpot_table    # don't know dims in advance, can't type
    cdef double[:] indexed_prob_vec

    for it in range(its):
        for n in range(N):
            d = dstates[n]
            lprobs = vector[double](d, 0)
            for f in nbr_factor_ids[n]:  # ids of neighboring factors
                scope = scopes[f]  # list of node ids
                lpot_table = lpot_tables[f]
                idx = tuple(slice(None) if i == n else x[i] for i in scope)
                indexed_prob_vec = lpot_table[idx]
                # print('  lpot_table[idx]', lpot_table[idx])

                # probs += lpot_table[idx]
                for j in range(d):
                    lprobs[j] += indexed_prob_vec[j]

            probs = softmax(lprobs, d)
            # print('probs')
            # print(probs)
            # x[n] = np.random.choice(d, p=probs)
            x[n] = sample_categorical(probs, d)

    # return x
