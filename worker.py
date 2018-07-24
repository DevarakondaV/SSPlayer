import tensorflow as tf
from GameController import *
from cnn import *

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
writer,summ,q,qs_1,qs_2,a = create_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)
wait_for(2)

sess_param = tf.ConfigProto()
sess_param.gpu_options.allow_growth = True

with tf.Session("grpc://localhost:2222",config=sess_param) as sess:
    writer.add_graph(sess.graph)
    sess.run([tf.global_variables_initializer()])
    print(sess.run([a,qs_1,qs_2]))

