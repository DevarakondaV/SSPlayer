
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf


# In[2]:


def conv_layer(m_input,size_in,size_out,k_size_w,k_size_h,conv_stride,pool_k_size,pool_stride_size,trainable_vars,name,num):
    sdev = np.power(2.0/(k_size_w*k_size_h*size_in),0.5)
    print("sdev"+name+num+": ",sdev)
    with tf.name_scope(name+num):
        w = tf.Variable(tf.truncated_normal([k_size_w,k_size_h,size_in,size_out],stddev=sdev,dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="b{}".format(num))
        conv = tf.nn.conv2d(m_input,w,strides=[1,conv_stride,conv_stride,1],padding="SAME")
        act = tf.nn.leaky_relu((conv+b),alpha=0.1)
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return tf.nn.max_pool(act,ksize=[1,pool_k_size,pool_k_size,1],strides=[1,pool_stride_size,pool_stride_size,1],padding='SAME')


def fc_layer(m_input,size_in,size_out,trainable_vars,name,num):
    sdev = np.power(2.0/(size_in*size_out),0.5)
    print("sdev"+name+num+": ",sdev)
    with tf.name_scope(name+num):
        w = tf.Variable(tf.truncated_normal([size_in, size_out],stddev=sdev,dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="b{}".format(num))
        z = tf.matmul(m_input,w)
        act = tf.nn.leaky_relu(z+b,alpha=0.1,name=("act"+num))
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return act
        
    
    
def get_place_holders():
    a = tf.get_default_graph().get_tensor_by_name("infer_place_holder/x1:0")
    b = tf.get_default_graph().get_tensor_by_name("infer_place_holder/train_bool:0")
    c = tf.get_default_graph().get_tensor_by_name("train_place_holder/seq:0")
    return a,b,c


def build_graph(name,net_in,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,trainable_vars):
    with tf.name_scope(name):
        conv_name="conv"
        fcs_name="FC"
        conv_feats[0] = 4
        fc_feats[0] = 384
        with tf.name_scope("Convolution_Layers"):
            convs = []
            convs.append(net_in)
            p = 0
            for i in range(0,conv_count-1):
                convs.append(conv_layer(convs[i],
                                        conv_feats[i],conv_feats[i+1],
                                        conv_k_size[p],conv_k_size[p],
                                        conv_stride[p],
                                        2,2,trainable_vars,
                                        conv_name,str(i+1)))
                p = p+1
            
            flatten = tf.reshape(convs[conv_count-1],[-1,fc_feats[0]])
            
        with tf.name_scope("Dense_Layers"):
            fcs = []
            fcs.append(flatten)
            for i in range(0,fc_count-1):
                fcs.append(fc_layer(fcs[i],
                                    fc_feats[i],fc_feats[i+1],
                                    trainable_vars,fcs_name,str(i+1)))
            output_layer = fcs[len(fcs)-1]
    return output_layer

def parse_fn(seq):
    fmt = {
        "img1": tf.FixedLenFeature([110,84,4],tf.int64,tf.zeros(shape=[110,84,4])),
        "a": tf.FixedLenFeature([1],tf.int64,-1),
        "r": tf.FixedLenFeature([1],tf.int64,-1),
        "img2": tf.FixedLenFeature([110,84,4],tf.int64,tf.zeros(shape=[110,84,4]))        
    }
    
    parsed = tf.parse_single_example(seq,fmt)
    img1 = parsed["img1"]
    img2 = parsed["img2"]
    a = parsed["a"]
    r = parsed["r"]
    img1 = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),tf.expand_dims(tf.reshape(img1,shape=[4,110,84]),axis=3))
    img2 = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),tf.expand_dims(tf.reshape(img2,shape=[4,110,84]),axis=3))
    img1 = tf.cast(tf.reshape(img1,shape=[110,84,4]),tf.float16)
    img2 = tf.cast(tf.reshape(img2,shape=[110,84,4]),tf.float16)
    return img1,a,r,img2

def build_train_data_pipeline(filenames,batchsize):
    with tf.name_scope("Train_Data_Pipeline"):
        files = tf.data.Dataset.list_files(filenames)
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.Dataset.from_tensors(filename),
            cycle_length=4,
            prefetch_input_elements=1))
        dataset = dataset.shuffle(buffer_size=25)
        dataset = dataset.map(parse_fn,num_parallel_calls=2)
        dataset = dataset.batch(batchsize).prefetch(2)
    return dataset
        
        

def build_train_queue(batch_size):
    with tf.name_scope("TrainQueue"):
        q = tf.FIFOQueue(capacity=25,
                         dtypes= (tf.uint8,tf.uint8,tf.float16,tf.uint8),
                         shapes= (tf.TensorShape([batch_size,110,84,4]),
                                  tf.TensorShape([batch_size,1]),
                                  tf.TensorShape([batch_size,1]),
                                  tf.TensorShape([batch_size,110,84,4])),
                         name="tq",shared_name="train_queue")
        
    return q

    
def conv_img_float(img_frames):
    img1 = tf.expand_dims(tf.image.convert_image_dtype(img_frames[:,:,0],tf.float16),axis=2)
    img2 = tf.expand_dims(tf.image.convert_image_dtype(img_frames[:,:,1],tf.float16),axis=2)
    img3 = tf.expand_dims(tf.image.convert_image_dtype(img_frames[:,:,2],tf.float16),axis=2)
    img4 = tf.expand_dims(tf.image.convert_image_dtype(img_frames[:,:,3],tf.float16),axis=2)
    return tf.concat([img1,img2,img3,img4],axis=2)

def standardize_frames(img_frames):
    img1 = tf.expand_dims(img_frames[:,:,0],axis=2)
    img2 = tf.expand_dims(img_frames[:,:,1],axis=2)
    img3 = tf.expand_dims(img_frames[:,:,2],axis=2)
    img4 = tf.expand_dims(img_frames[:,:,3],axis=2)

    print(img1.dtype)

    std_img1 = tf.image.per_image_standardization(img1)
    std_img2 = tf.image.per_image_standardization(img2)
    std_img3 = tf.image.per_image_standardization(img3)
    std_img4 = tf.image.per_image_standardization(img4)

    print(std_img1.dtype)
    frames = tf.concat([std_img1,std_img2,std_img3,std_img4],axis=2)
    print(frames.dtype)
    return frames

def standardize_img(img_array):
    img_std = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), 
                        img_array,dtype=tf.float32,
                        parallel_iterations=4)
    return tf.cast(img_std,tf.float16)

def build_update_infer_weights_op(conv_name,fc_name,conv_count,fc_count):
    num_conv = conv_count
    num_fc = fc_count
    
    def get_tensor(name):
        return tf.get_default_graph().get_tensor_by_name(name)
    
    infer_conv_w = [get_tensor("Inference/Convolution_Layers/{}{}/w{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    infer_conv_b = [get_tensor("Inference/Convolution_Layers/{}{}/b{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    infer_fc_w = [get_tensor("Inference/Dense_Layers/{}{}/w{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    infer_fc_b = [get_tensor("Inference/Dense_Layers/{}{}/b{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    
    
    
    train_conv_w = [get_tensor("Train/Convolution_Layers/{}{}/w{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    train_conv_b = [get_tensor("Train/Convolution_Layers/{}{}/b{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    train_fc_w = [get_tensor("Train/Dense_Layers/{}{}/w{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    train_fc_b = [get_tensor("Train/Dense_Layers/{}{}/b{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]

    assign_ops_conv_w = [tf.assign(a,b) for a,b in zip(infer_conv_w,train_conv_w)]
    assign_ops_conv_b = [tf.assign(a,b) for a,b in zip(infer_conv_b,train_conv_b)]
    assign_ops_fc_w = [tf.assign(a,b) for a,b in zip(infer_fc_w,train_fc_w)]
    assign_ops_fc_b = [tf.assign(a,b) for a,b in zip(infer_fc_b,train_fc_b)]
    return [assign_ops_conv_w,assign_ops_conv_b,assign_ops_fc_w,assign_ops_fc_b]

def start_server():
    server = tf.train.Server.create_local_server()
    print(server.target)
    return server


# In[3]:


def create_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    if (len(conv_feats) != conv_count):
        return

    # 0 is Trainer....1 is inference

    tf.reset_default_graph()
    
    with tf.name_scope("infer_place_holder"):
        x1 = tf.placeholder(tf.uint8,shape=[1,110,84,4],name="x1")
 
    
    with tf.device("/job:local/task:0"):
        train_q = build_train_queue(batch_size)

   
    with tf.device("/job:local/task:0"):
        q_s1 = tf.Print(train_q.size(),[train_q.size()],message="Q Size1: ")
    with tf.device("/job:local/task:1"):
        q_s2 = tf.Print(train_q.size(),[train_q.size()],message="Q Size2: ")
        p_op = tf.Print(x1.get_shape().as_list(),[x1.get_shape().as_list()],message="p_op Task 1")
    
    std_infer_img = standardize_img(x1)
    
    #Test Variable
    with tf.device("/job:local/task:0"):
        img1,a,r,img2 = train_q.dequeue(name="Dequeue")
    

    input_var = tf.Variable(tf.zeros([batch_size,110,84,4],dtype=tf.float16),name="input_var")
    assign_infer_op = tf.assign(input_var,standardize_img(img2))
    assign_train_op = tf.assign(input_var,standardize_img(img1))

    with tf.device("/job:local/task:1"):
        infer_output = build_graph("Inference",std_infer_img,
                                    conv_count,fc_count,
                                    conv_feats,fc_feats,conv_k_size,conv_stride,False)

    with tf.device("/job:local/task:0"):
        train_output = build_graph("Train",input_var,
                                   conv_count,fc_count,
                                   conv_feats,fc_feats,conv_k_size,conv_stride,True)

    with tf.device("/job:local/task:1"):
        Qnext_val = tf.reduce_max(infer_output,name="Qnext_val")
        action = tf.argmax(infer_output,axis=1,name="action")

    with tf.device("/job:local/task:0"):
        with tf.name_scope("Trainer"):
            with tf.control_dependencies([assign_infer_op]):
                Qnext = tf.reduce_max(train_output,name="Qnext_train")
                #gamma_seq = tf.tile(gamma,[batch_size])
                y = tf.add(r,tf.multiply(gamma,Qnext),name="y")
            with tf.control_dependencies([assign_train_op]):
                loss = tf.reduce_sum(tf.pow(y-train_output,2))
                tf.summary.scalar("loss",loss)
                train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,name="trainer")
        
    with tf.device("/job:local/task:0"):
        with tf.name_scope("weight_update_ops"):
            ops = build_update_infer_weights_op("conv","FC",conv_count,fc_count)

    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    #writer.add_summary(summ.eval(),g_step.eval())
    return writer,summ,train,action,x1,train_q,p_op,q_s1

def infer_model(learning_rate,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    if (len(conv_feats) != conv_count):
        return
    
    with tf.name_scope("infer_place_holder"):
        x1 = tf.placeholder(tf.uint8,shape=[None,110,84,4],name="x1")
        
    #tf.summary.image("image",x1)
    std_img = standardize_img(x1)
    #tf.summary.image("imagestd",std_img)

    infer_output = build_graph("Inference",std_img,
                                conv_count,fc_count,
                                conv_feats,fc_feats,conv_k_size,conv_stride,False)
    
    Qnext_val = tf.reduce_max(infer_output,name="Qnext_val")
    action = tf.argmax(infer_output,axis=1,name="action")
    return x1,action
    
    
def train_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    if (len(conv_feats) != conv_count):
        return
    
    with tf.device("/job:worker/task:1"):    
        with tf.name_scope("train_place_holder"):
            s_img1 = tf.placeholder(tf.uint8,shape=[None,110,84,4],name="s_img1")
            s_a = tf.placeholder(tf.uint8,shape=[None,1],name="s_a")
            s_r = tf.placeholder(tf.float16,shape=[None,1],name="s_r")
            s_img2 = tf.placeholder(tf.uint8,shape=[None,110,84,4],name="s_img2")
       
        with tf.name_scope("queue"):
            train_q = build_train_queue(batch_size)
        
        enqueue_op = train_q.enqueue((s_img1,s_a,s_r,s_img2))
        q_s1 = tf.Print(train_q.size(),[train_q.size()],message="Q Size1: ")
        img1,a,r,img2 = train_q.dequeue(name="dequeue")

        std_img1 = standardize_img(img1)
        std_img2 = standardize_img(img2)

        tf.summary.image("std_img1",img1)

        pimg = tf.Print(std_img1,[std_img1],"val: ")

        input_var = tf.Variable(tf.zeros([batch_size,110,84,4],dtype=tf.float16),name="input_var")
        assign_infer_op = tf.assign(input_var,std_img2)
        assign_train_op = tf.assign(input_var,std_img1)
        
        train_output = build_graph("Train",input_var,
                                    conv_count,fc_count,
                                    conv_feats,fc_feats,conv_k_size,conv_stride,True)
        
        with tf.name_scope("Trainer"):
            with tf.control_dependencies([assign_infer_op]):
                Qnext = tf.reduce_max(train_output,name="Qnext_train")
            #gamma_seq = tf.tile(gamma,[batch_size])
                y = tf.add(r,tf.multiply(gamma,Qnext),name="y")
            with tf.control_dependencies([assign_train_op]):
                loss = tf.reduce_sum(tf.pow(y-train_output,2))
                tf.summary.scalar("loss",loss)
                train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,name="trainer")
    
    with tf.name_scope("weight_update_ops"):
        ops = build_update_infer_weights_op("conv","FC",conv_count,fc_count)
        
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    #return writer,summ,train,train_q,q_s1
    return writer,summ,train,enqueue_op,q_s1,s_img1,s_a,s_r,s_img2,pimg,ops