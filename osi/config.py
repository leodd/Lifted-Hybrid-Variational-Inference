import tensorflow as tf

num_threads = 1
tfconfig = tf.ConfigProto()
tfconfig.intra_op_parallelism_threads = num_threads
tfconfig.inter_op_parallelism_threads = num_threads
# tfconfig = None  # no limit on number of cores (default)

# if True, will ignore constant terms in LogQuadratic when evaluating log pot
# in mixture_beliefs.py; ok to ignore in e.g. Gaussian mrfs, so tensorflow may
# be more numerically stable/accurate; should only shift the objective by const
ignore_const_when_group_eval_LogQuadratic = False

# initialize means on as large a grid as possible, for variational methods to
# hopefully find better solution
init_grid, init_grid_noise = True, 0.1
 #init_grid, init_grid_noise = False, -1

