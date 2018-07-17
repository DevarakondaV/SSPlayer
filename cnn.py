
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf
"test"


# In[2]:


def conv_layer(m_input,size_in,size_out,k_size_w,k_size_h,conv_stride,pool_k_size,pool_stride_size,name,num):
    with tf.name_scope(name+num):
        w = tf.get_default_graph().get_tensor_by_name("network_conv_weights/"+name+num+"/w"+num+":0")
        b = tf.get_default_graph().get_tensor_by_name("network_conv_weights/"+name+num+"/b"+num+":0")
        conv = tf.nn.conv2d(m_input,w,strides=[1,conv_stride,conv_stride,1],padding="SAME")
        act = tf.nn.leaky_relu((conv+b),alpha=0.1)
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return tf.nn.max_pool(act,ksize=[1,pool_k_size,pool_k_size,1],strides=[1,pool_stride_size,pool_stride_size,1],padding='SAME')


def fc_layer(m_input,size_in,size_out,name,num):
    with tf.name_scope(name+num):
        w = tf.get_default_graph().get_tensor_by_name("network_fc_weights/"+name+num+"/w"+num+":0")
        b = tf.get_default_graph().get_tensor_by_name("network_fc_weights/"+name+num+"/b"+num+":0")
        z = tf.matmul(m_input,w)
        act = tf.nn.leaky_relu(z+b,alpha=0.1,name=("act"+num))
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return act

def conv_weights(size_in,size_out,k_size_w,k_size_h,name,num):
    w = tf.get_default_graph().get_tensor_by_name("conv_weights_container/w"+num+"cur:0")
    b = tf.get_default_graph().get_tensor_by_name("conv_weights_container/b"+num+"cur:0")
    with tf.name_scope(name+num):
        tf.Variable(w,name="w"+num)
        tf.Variable(b,name="b"+num)

def fc_weights(size_in,size_out,name,num):
    w = tf.get_default_graph().get_tensor_by_name("fc_weights_container/w"+num+"cur:0")
    b = tf.get_default_graph().get_tensor_by_name("fc_weights_container/b"+num+"cur:0")
    with tf.name_scope(name+num):
        tf.Variable(w,name="w"+num)
        tf.Variable(b,name="b"+num)
    
def conv_weights_container(size_in, size_out, k_size_w, k_size_h,name,num):
    sdev = np.power(2.0/(k_size_w*k_size_h*size_in),0.5)
    print("sdev"+name+num+": ",sdev)
    lower,upper = -1,1
    mu = 0
    wi = tn((lower-mu)/sdev,(upper-mu)/sdev,loc = mu,scale=sdev).rvs(size=[k_size_w,k_size_h,size_in,size_out])
    bi = np.full(size_out,0)
    
    w1 = tf.Variable(tf.truncated_normal([k_size_w,k_size_h,size_in,size_out],stddev=sdev,dtype=tf.float16),dtype=tf.float16,name=("w"+num+"cur"),trainable=False)
    b1 = tf.Variable(bi,dtype=tf.float16,name=("b"+num+"cur"),trainable=False)
    w2 = tf.Variable(tf.truncated_normal([k_size_w,k_size_h,size_in,size_out],stddev=sdev,dtype=tf.float16),dtype=tf.float16,name=("w"+num+"pre"),trainable=False)
    b2 = tf.Variable(bi,dtype=tf.float16,name=("b"+num+"pre"),trainable=False)
    
        
def fc_weights_container(size_in,size_out,name,num):
    sdev = np.power(2.0/(size_in*size_out),0.5)
    print("sdev"+name+num+": ",sdev)
    lower,upper = -1,1
    mu = 0
    wi = tn((lower-mu)/sdev,(upper-mu)/sdev,loc = mu,scale=sdev).rvs(size=[size_in,size_out])
    bi = np.full(size_out,0)

    w1 = tf.Variable(tf.truncated_normal([size_in, size_out],stddev=sdev,dtype=tf.float16),dtype=tf.float16,name=("w"+num+"cur"),trainable=False)
    b1 = tf.Variable(bi,dtype=tf.float16,name=("b"+num+"cur"),trainable=False)
    w2 = tf.Variable(tf.truncated_normal([size_in, size_out],stddev=sdev,dtype=tf.float16),dtype=tf.float16,name=("w"+num+"pre"),trainable=False)
    b2 = tf.Variable(bi,dtype=tf.float16,name=("b"+num+"pre"),trainable=False)
    
    
def get_place_holders():
    a = tf.get_default_graph().get_tensor_by_name("place_holder/x1:0")
    b = tf.get_default_graph().get_tensor_by_name("place_holder/y:0")
    c = tf.get_default_graph().get_tensor_by_name("place_holder/x2:0")
    d = tf.get_default_graph().get_tensor_by_name("place_holder/next_state:0")
    e = tf.get_default_graph().get_tensor_by_name("place_holder/qnext:0")
    return a,b,c,d,e

def get_network_WB():
    CW1 = tf.get_default_graph().get_tensor_by_name("network_conv_weights/conv1/w1:0")
    CW2 = tf.get_default_graph().get_tensor_by_name("network_conv_weights/conv2/w2:0")
    FW1 = tf.get_default_graph().get_tensor_by_name("network_fc_weights/FC1/w1:0")
    FW2 = tf.get_default_graph().get_tensor_by_name("network_fc_weights/FC2/w2:0")

    CB1 = tf.get_default_graph().get_tensor_by_name("network_conv_weights/conv1/b1:0")
    CB2 = tf.get_default_graph().get_tensor_by_name("network_conv_weights/conv2/b2:0")
    FB1 = tf.get_default_graph().get_tensor_by_name("network_fc_weights/FC1/b1:0")
    FB2 = tf.get_default_graph().get_tensor_by_name("network_fc_weights/FC2/b2:0")
    
    return [CW1,CW2,FW1,FW2],[CB1,CB2,FB1,FB2]

def assign_weights_to_network(dim):
    weights,biases = get_network_WB()
    wcur,bcur,wpre,bpre = get_container_WB()
    
    if (dim == 0):
        wval = wcur
        bval = bcur
    else:
        wval = wpre
        bval = bpre
        
    ops = []
    for i in range(0,len(weights)):
        ops.append(tf.assign(weights[i],wval[i]))
        ops.append(tf.assign(biases[i],bval[i]))
    sess.run(ops)
    
    return

def update_container_matricies(sess):
    w,b = get_network_WB()
    wcur,bcur,wpre,bpre = get_container_WB()
    
    ops = []
    for i in range(0,len(wcur)):
        ops.append(tf.assign(wpre[i],wcur[i]))
        ops.append(tf.assign(bpre[i],bcur[i]))
    sess.run(ops)
    
    ops = []
    for i in range(0,len(wcur)):
        ops.append(tf.assign(wcur[i],w[i]))
        ops.append(tf.assign(bcur[i],b[i]))
    sess.run(ops)
    return

def test_update_container_matricies(sess):
    w,b = get_network_WB()
    w1,b1 = get_test_WB()
    
    ops = []
    for i in range(0,len(w1)):
        ops.append((w1[i][1]).assign(w1[i][0]))
        ops.append((b1[i][1]).assign(b1[i][0]))
    sess.run(ops)
    
    ops = []
    for i in range(0,len(w1)):
        ops.append(w1[i][0].assign(w[i]))
        ops.append(b1[i][0].assign(b[i]))
    sess.run(ops)
    return
    

def print_test_FC(sess):
    wcur,bcur,wpre,bpre = get_container_WB()
    w1,b1 = get_network_WB()
    print("current: ",sess.run([wcur[2][0][0]]))
    print("previous: ",sess.run([wpre[2][0][0]]))
    print("current NN",sess.run([w1[2][0][0]]))
    print(" ")
    
def get_container_WB():
    CW1cur = tf.get_default_graph().get_tensor_by_name("conv_weights_container/w1cur:0")
    CW2cur = tf.get_default_graph().get_tensor_by_name("conv_weights_container/w2cur:0")
    FW1cur = tf.get_default_graph().get_tensor_by_name("fc_weights_container/w1cur:0")
    FW2cur = tf.get_default_graph().get_tensor_by_name("fc_weights_container/w2cur:0")
    
    CW1pre = tf.get_default_graph().get_tensor_by_name("conv_weights_container/w1pre:0")
    CW2pre = tf.get_default_graph().get_tensor_by_name("conv_weights_container/w2pre:0")
    FW1pre = tf.get_default_graph().get_tensor_by_name("fc_weights_container/w1pre:0")
    FW2pre = tf.get_default_graph().get_tensor_by_name("fc_weights_container/w2pre:0")

    CB1cur = tf.get_default_graph().get_tensor_by_name("conv_weights_container/b1cur:0")
    CB2cur = tf.get_default_graph().get_tensor_by_name("conv_weights_container/b2cur:0")
    FB1cur = tf.get_default_graph().get_tensor_by_name("fc_weights_container/b1cur:0")
    FB2cur = tf.get_default_graph().get_tensor_by_name("fc_weights_container/b2cur:0")
    
    CB1pre = tf.get_default_graph().get_tensor_by_name("conv_weights_container/b1pre:0")
    CB2pre = tf.get_default_graph().get_tensor_by_name("conv_weights_container/b2pre:0")
    FB1pre = tf.get_default_graph().get_tensor_by_name("fc_weights_container/b1pre:0")
    FB2pre = tf.get_default_graph().get_tensor_by_name("fc_weights_container/b2pre:0")
    
    
    
    return [CW1cur,CW2cur,FW1cur,FW2cur],[CB1cur,CB2cur,FB1cur,FB2cur],[CW1pre,CW2pre,FW1pre,FW2pre],[CB1pre,CB2pre,FB1pre,FB2pre]
    
def trainer(current_state,next_state,reward,gamma):
    train = tf.get_default_graph().get_operation_by_name("train/trainer")
    x1,y,x2,next_state_bool,Qnext= get_place_holders()
    q_compute = tf.get_default_graph().get_tensor_by_name("Qnext_val:0")
    action = tf.get_default_graph().get_tensor_by_name("action/action:0")
    
    #print("Before Next State")
    #print_test_FC(sess)
    Qnext_val = sess.run([q_compute],{x1: current_state, x2: next_state,next_state_bool: True})
    Qnext_val = reward+(gamma*np.max(Qnext_val))
    Qnext_val = np.array(Qnext_val).reshape((1,1))
    #print("After Next State True")
    #print_test_FC(sess)
    s = sess.run([train],{x1: current_state,x2: next_state, next_state_bool: False, Qnext: Qnext_val})
    #print("After Train")
    #print_test_FC(sess)
    return




# In[3]:


def create_model(learning_rate,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    if (len(conv_feats) != conv_count):
        return
    
    tf.reset_default_graph()
    
    with tf.name_scope("place_holder"):
        x1 = tf.placeholder(tf.float16,shape=[None,110,84,4],name="x1")
        y = tf.placeholder(tf.float16,shape=[None,4],name="y")
        x2 = tf.placeholder(tf.float16,shape=[None,110,84,4],name="x2")
        next_state = tf.placeholder(tf.bool,name="next_state")
        Qnext = tf.placeholder(tf.float16,shape=[None,1],name="qnext")
        

    tf.summary.image("image",x1,max_outputs=4)
    conv_name="conv"
    conv_feats[0] = 4
    p = 0
    with tf.name_scope("conv_weights_container"):
        for i in range(0,conv_count-1):
            conv_weights_container(conv_feats[i],conv_feats[i+1],conv_k_size[p],conv_k_size[p],conv_name,str(i+1))
            p = p+1
    
    p = 0
    with tf.name_scope("network_conv_weights"):
        for i in range(0,conv_count-1):
            conv_weights(conv_feats[i],conv_feats[i+1],conv_k_size[p],conv_k_size[p],conv_name,str(i+1))
            p = p+1
    
    p = 0
    fcs_name="FC"
    fc_feats[0] = conv_feats[len(conv_feats)-1]*4
    with tf.name_scope("fc_weights_container"):
        for i in range(0,fc_count-1):
            fc_weights_container(fc_feats[i],fc_feats[i+1],fcs_name,str(i+1))
    
    p = 0
    with tf.name_scope("network_fc_weights"):
        for i in range(0,fc_count-1):
            fc_weights(fc_feats[i],fc_feats[i+1],fcs_name,str(i+1))
            p = p+1
        
    
    weights,biases = get_network_WB()
    wcur,bcur,wpre,bpre = get_container_WB()
    
    def f_true():
        #if next_state = true
        #Replace all weights with previous
        ops = []
        for i in range(0,len(weights)):
            ops.append(tf.assign(wcur[i],weights[i]))
            ops.append(tf.assign(bcur[i],biases[i]))
            ops.append(tf.assign(weights[i],wpre[i]))
            ops.append(tf.assign(biases[i],bpre[i]))

        return ops
        
    def f_false():
        #if next_state = false
        #Replace all weights with current
        ops = []
        for i in range(0,len(weights)):
            ops.append(tf.assign(weights[i],wcur[i]))
            ops.append(tf.assign(biases[i],bcur[i]))
            ops.append(tf.assign(wpre[i],wcur[i]))
            ops.append(tf.assign(bpre[i],bcur[i]))
        
        return ops

    #control_ops = tf.cond(next_state,f_true,f_false,name="control_op_cond")
    #in_image = tf.cond(next_state,lambda: x2,lambda: x1,name="state_condition")

    
    #with tf.control_dependencies(control_ops):
    #    with tf.name_scope("convolution_layers"):
    #        convs = []
    #        #convs.append(in_image)    
    #        convs.append(x_in)
    #        p = 0
    #        for i in range(0,conv_count-1):
    #            convs.append(conv_layer(convs[i],conv_feats[i],conv_feats[i+1],conv_k_size[p],conv_k_size[p],conv_stride[p],2,2,conv_name,str(i+1)))
    

    #   flatten = tf.reshape(convs[conv_count-1],[-1,fc_feats[0]])
    
    #    with tf.name_scope("dense_layers"):
    #        fcs = []
    #        fcs.append(flatten)
    #        for i in range(0,fc_count-1):
    #            fcs.append((fc_layer(fcs[i],fc_feats[i],fc_feats[i+1],fcs_name,str(i+1))))
    
    #    output_layer = fcs[len(fcs)-1]
     
    with tf.name_scope("convolution_layers"):
        convs = []
        convs.append(x1)
        p = 0
        for i in range(0,conv_count-1):
            convs.append(conv_layer(convs[i],conv_feats[i],conv_feats[i+1],conv_k_size[p],conv_k_size[p],conv_stride[p],2,2,conv_name,str(i+1)))
        
    flatten = tf.reshape(convs[conv_count-1],[-1,fc_feats[0]])
    
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
    
    sess = tf.InteractiveSession(config=config)
    #sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    return sess,writer,summ,[x1,x2,y,next_state,Qnext]


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

