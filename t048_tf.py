import tensorflow as tf
import sys
import numpy as np
from dist_cnn import *
from t048_trainer import *
#from gsheets import *


x = np.random.rand(1,110,84,4).astype(np.uint8)
i = np.random.rand(10,110,84,4).astype(np.uint8)
ia = np.random.rand(10,1).astype(np.uint8)
ir = np.random.rand(10,1).astype(np.float16)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

conv_k_size = [8,4]
conv_stride = [4,2]
conv = [0,16,32]
fclyr = [0,256,128,64] #5
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 0.00025
gamma = np.array([.9]).astype(np.float16)
batch_size = 5   
LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log"

writer,summ,infer_ops,train_ops,update_ops = create_model2(learning_rate,gamma,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)


with tf.Session(config=config) as Sess:
    Sess.run([tf.global_variables_initializer()])
    writer.add_graph(Sess.graph) 

