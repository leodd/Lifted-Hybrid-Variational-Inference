import tensorflow as tf

num_threads = 1
tfconfig = tf.ConfigProto()
tfconfig.intra_op_parallelism_threads = num_threads
tfconfig.inter_op_parallelism_threads = num_threads
# tfconfig = None  # no limit on number of cores (default)

# if True, will ignore constant terms in LogQuadratic when evaluating log pot
# in mixture_beliefs.py; ok to ignore in e.g. Gaussian mrfs, so tensorflow may
# be more numerically stable/accurate; can give wrong result in say hybrid mrf
ignore_const_when_group_eval_LogQuadratic = False

