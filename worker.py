import tensorflow as tf
from GameController import *
from dist_cnn import *

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
writer,summ,train,a,x1,train_q = create_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)
wait_for(2)

x = np.random.rand(1,110,84,4)
i = np.random.rand(10,110,84,4).astype(np.uint8)
ia = np.random.rand(10,1).astype(np.uint8)
ir = np.random.rand(10,1).astype(np.float16)


sess_param = tf.ConfigProto()
sess_param.gpu_options.allow_growth = True

with tf.Session("grpc://localhost:2222",config=sess_param) as sess:
    #writer.add_graph(sess.graph)
    sess.run([tf.global_variables_initializer()])
    #tf.train.start_queue_runners(sess=sess)
    print(sess.run([a],{x1: x}))
    print("TESTEST")
    #print(sess.run([train]))
    #print(sess.run([train],{img1: i,a: ia,r: ir,img2: i}))

