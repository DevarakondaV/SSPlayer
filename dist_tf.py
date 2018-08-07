import tensorflow as tf
import sys
import numpy as np
from dist_cnn import *
from trainer import *

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
LOGDIR = r"c:\Users\devar\Documents\EngProj\SSPlayer\log"    
app_dir = r"c:\Users\devar\Documents\EngProj\SSPlayer\Release.win32\ShapeScape.exe"

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
        
        writer,summ,train,enqueue_op,q_sl,s_img1,s_a,s_r,s_img2,pimg,uwb = train_model(learning_rate,gamma,
                                                     batch_size,conv_count,
                                                     fc_count,conv,
                                                     fclyr,conv_k_size,
                                                     conv_stride,LOGDIR)

    ops = {
        'action': a,'enqueue_op': enqueue_op,
        'train': train,'uwb': uwb
    }
    
    phs = {
        'x1': x1, 's_img1': s_img1,
        's_a': s_a,'s_r': s_r,
        's_img2': s_img2
    }

    if (t_num == 0):
        game = SSPlayer(app_dir,2)
        wait_for(1)
        game.click_play()
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(t_num == 0),config=config) as sess:
            writer.add_graph(sess.graph)
            #while not sess.should_stop():
            dist_run(sess,game,.9,25,batch_size,ops,phs)
            key_req = input("Enter any key to play")
            dist_play(sess,game,3,ops,phs)
    else:
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(t_num == 0),config=config) as sess:
            pp = 0
            while not sess.should_stop():
                tt,qq,s = sess.run([train,q_sl,summ],{x1: np.random.rand(1,110,84,4).astype(np.uint8)})
                writer.add_summary(s,pp)
                pp = pp+1

    
