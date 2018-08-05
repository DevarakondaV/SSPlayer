import tensorflow as tf
import sys
import numpy as np
from dist_cnn import * 

s_name = str(sys.argv[1])
t_num = int(sys.argv[2])

cl_spec = tf.train.ClusterSpec({
    "worker": [
        "localhost:2223",
        "localhost:2224"
    ],
    "ps": [
        "localhost:2222"
    ]
})


x = np.random.rand(1,110,84,4).astype(np.uint8)
i = np.random.rand(10,110,84,4).astype(np.uint8)
ia = np.random.rand(10,1).astype(np.uint8)
ir = np.random.rand(10,1).astype(np.float16)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


conv_k_size = [8,4]
conv_stride = [4,2]
conv = [0,16,32]
fclyr = [0,125,5]
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 1e-4
gamma = np.array([.9]).astype(np.float16)
batch_size = 10
LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log"    
    
if s_name == "ps":
    server = tf.train.Server(cl_spec,job_name="ps",task_index=0,config=config)
    server.join()
else:
    server = tf.train.Server(cl_spec,job_name="worker",task_index=t_num,config=config)
    with tf.device(tf.train.replica_device_setter(cluster=cl_spec)):
        x1,a = infer_model(learning_rate,batch_size,
                    conv_count,fc_count,
                    conv,fclyr,
                    conv_k_size,conv_stride,LOGDIR)
        
        writer,summ,train,enqueue_op,q_sl,s_img1,s_a,s_r,s_img2 = train_model(learning_rate,gamma,
                                                     batch_size,conv_count,
                                                     fc_count,conv,
                                                     fclyr,conv_k_size,
                                                     conv_stride,LOGDIR)

    
    if (t_num == 0):
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(t_num == 0),config=config) as sess:
            pp = 0
            while not sess.should_stop():
                sess.run([a],{x1: x})
                sess.run([enqueue_op],{s_img1: i,s_a: ia,s_r: ir,s_img2: i})
                print(pp)
                pp = pp+1
    else:
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(t_num == 0),config=config) as sess:
            while not sess.should_stop():
                sess.run([train,q_sl])