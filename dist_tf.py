import tensorflow as tf
import sys

t_num = int(sys.argv[1])

cl_spec = tf.train.ClusterSpec(
    {"local": ["localhost:2222","localhost:2223"]}
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if t_num == 0:
    server = tf.train.Server(cl_spec,job_name="local",task_index=0,config=config)
    server.join()
else:
    server = tf.train.Server(cl_spec,job_name="local",task_index=1,config=config)
    server.join()