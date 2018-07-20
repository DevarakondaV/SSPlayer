
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf
"test"


# In[2]:

def conv_layer(m_input,size_in,size_out,k_size_w,k_size_h,conv_stride,pool_k_size,pool_stride_size,name,num):
    sdev = np.power(2.0/(k_size_w*k_size_h*size_in),0.5)
    print("sdev"+name+num+": ",sdev)
    with tf.name_scope(name+num):
        w = tf.Variable(tf.truncated_normal([k_size_w,k_size_h,size_in,size_out],stddev=sdev,dtype=tf.float16),dtype=tf.float16,name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),dtype=tf.float16,name="b{}".format(num))
        conv = tf.nn.conv2d(m_input,w,strides=[1,conv_stride,conv_stride,1],padding="SAME")
        act = tf.nn.leaky_relu((conv+b),alpha=0.1)
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return tf.nn.max_pool(act,ksize=[1,pool_k_size,pool_k_size,1],strides=[1,pool_stride_size,pool_stride_size,1],padding='SAME')


def fc_layer(m_input,size_in,size_out,name,num):
    sdev = np.power(2.0/(size_in*size_out),0.5)
    print("sdev"+name+num+": ",sdev)
    with tf.name_scope(name+num):
        w = tf.Variable(tf.truncated_normal([size_in, size_out],stddev=sdev,dtype=tf.float16),dtype=tf.float16,name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),dtype=tf.float16,name="b{}".format(num))
        z = tf.matmul(m_input,w)
        act = tf.nn.leaky_relu(z+b,alpha=0.1,name=("act"+num))
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return act
        
    
    
def get_place_holders():
    a = tf.get_default_graph().get_tensor_by_name("place_holder/x1:0")
    b = tf.get_default_graph().get_tensor_by_name("place_holder/y:0")
    c = tf.get_default_graph().get_tensor_by_name("place_holder/x2:0")
    d = tf.get_default_graph().get_tensor_by_name("place_holder/next_state:0")
    e = tf.get_default_graph().get_tensor_by_name("place_holder/qnext:0")
    return a,b,c,d,e


def create_model(learning_rate,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    if (len(conv_feats) != conv_count):
        return
    
    tf.reset_default_graph()
    
    with tf.name_scope("place_holder"):
        x1 = tf.placeholder(tf.float16,shape=[None,110,84,4],name="x1")
        y = tf.placeholder(tf.float16,shape=[None,4],name="y")
        next_state = tf.placeholder(tf.bool,name="next_state")
        Qnext = tf.placeholder(tf.float16,shape=[None,1],name="qnext")
    
    with tf.name_scope("FIFOQueue"):
        q = tf.FIFOQueue(50,dtypes=tf.float16,shapes=[50,110,84,4],name="Train_Queue")
        img_in = q.dequeue(name="img_in")
        num_threads = 4
        qr = tf.train.QueueRunner(q,[img_in]*num_threads)
        tf.train.add_queue_runner(qr)
       
    with tf.device('/cpu:0'):
        x_img = tf.map_fn(lambda frame:tf.image.per_image_standardization(frame),img_in,dtype=tf.float32)
        x_imgs = tf.cast(x_img,tf.float16)
    
 
    #x_img = tf.map_fn(lambda frame:tf.image.per_image_standardization(frame),x1,dtype=tf.float32)
    #x_imgs = tf.cast(x_img,tf.float16)
   

    tf.summary.image("image",x_imgs,max_outputs=4)
    conv_name="conv"
    fcs_name="FC"
    conv_feats[0] = 4
    #fc_feats[0] = conv_feats[len(conv_feats)-1]*4
    fc_feats[0] = 384
    p = 0
    with tf.name_scope("convolution_layers"):
        convs = []
        convs.append(x_imgs)
        p = 0
        for i in range(0,conv_count-1):
            convs.append(conv_layer(convs[i],conv_feats[i],conv_feats[i+1],conv_k_size[p],conv_k_size[p],conv_stride[p],2,2,conv_name,str(i+1)))
            p = p+1
        
    #dim = tf.Variable(0.0,dtype=tf.float16)
    #dim_op = tf.assign(dim,tf.cast(tf.reduce_prod(tf.shape(convs[len(convs)-1])),tf.float16))
    
    
    flatten = tf.reshape(convs[conv_count-1],[-1,fc_feats[0]])
    
    p = 0
    with tf.name_scope("dense_layers"):
        fcs = []
        fcs.append(flatten)
        for i in range(0,fc_count-1):
            fcs.append(fc_layer(fcs[i],fc_feats[i],fc_feats[i+1],fcs_name,str(i+1)))
    output_layer = fcs[len(fcs)-1]

    with tf.name_scope("train"):
        loss = tf.reduce_sum(tf.pow(Qnext-output_layer,2))
        tf.summary.scalar("loss",loss)
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,name="trainer")

    
    Qnext_val = tf.reduce_max(output_layer,name="Qnext_val")
    action = tf.argmax(output_layer,axis=1,name="action")
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    
    with tf.train.MonitoredSession() as sess:
        writer.add_graph(sess.graph)
        it = 0
        while not sess.should_stop():
            a,s = sess.run([train,summ],{x1: np.random.rand(1,110,84,4),Qnext: np.random.rand(1).reshape(1,1)})
            writer.add_summary(s,it)
            it = it+1
    return

    #sess = tf.InteractiveSession(config=config)
    #sess = tf.InteractiveSession()
    #sess.run(tf.global_variables_initializer())
    #summ = tf.summary.merge_all()
    #writer = tf.summary.FileWriter(LOGDIR)
    #return sess,writer,summ,[x1,x2,y,next_state,Qnext]


# In[4]:


#conv_k_size = [8,4]
#conv_stride = [4,2]
#conv = [0,16,32]
#fclyr = [0,125,4]
#conv_count = len(conv)
#fc_count = len(fclyr)
#learning_rate = 1e-4
#batch_size = 10
#LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log"
#sess,writer,summ,place_holders= create_model(learning_rate,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)

#writer.add_graph(sess.graph)

