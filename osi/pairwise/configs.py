import tensorflow as tf

num_threads = 4
tfconfig = tf.ConfigProto()
tfconfig.intra_op_parallelism_threads = num_threads
tfconfig.inter_op_parallelism_threads = num_threads
datadir = '~/datasets'
