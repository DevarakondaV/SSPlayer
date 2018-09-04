import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf


def conv_layer(m_input,size_in,size_out,k_size_w,k_size_h,conv_stride,pool_k_size,pool_stride_size,trainable_vars,name,num):
    """
    Creates convolution Layer
    
    args:
        m_input: Input into the convolution layer
        size_in: number of input kernels
        size_out: number of output kernels
        k_size_w: Kernel width
        k_size_h: Kernel Height
        conv_stride: convolution stride
        pool_k_size: pool kernel size. Currently pooling kernel shape is square
        trainable_vars: Bool. Specifies if variables created in this convolution are trainable
        name: Name of the convolution layer
        num: Convolution layer number

    Returns:
        A tensor representing the max pool of activations of the layer
    """

    #He et Al. Standardeviation
    sdev = np.power(2.0/(k_size_w*k_size_h*size_in),0.5)
    #Xavier Initialization
    sdev = np.power(2.0/(size_in+size_out),0.5)
    print("sdev"+name+num+": ",sdev)
    
    with tf.name_scope(name+num):

        #Weight and bias initializations
        w = tf.Variable(tf.truncated_normal([k_size_w,k_size_h,size_in,size_out],stddev=sdev,dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="b{}".format(num))

        #Convolution and activations
        conv = tf.nn.conv2d(m_input,w,strides=[1,conv_stride,conv_stride,1],padding="SAME")
        #act = tf.nn.relu((conv+b),name="relu")
        act = tf.nn.leaky_relu((conv+b),alpha=0.3)
        #act = tf.sigmoid((conv+b),name="sigmoid")
        #act = tf.tanh((conv+b),name="tanh")
        intermediate_summary_img(act,num,trainable_vars)
        
        #summaries
        tf.summary.histogram("weights",w)
        flatten_weights_summarize(w,num,trainable_vars)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return tf.nn.max_pool(act,ksize=[1,pool_k_size,pool_k_size,1],strides=[1,pool_stride_size,pool_stride_size,1],padding='SAME')


def fc_layer(m_input,size_in,size_out,trainable_vars,name,num):
    """
    Creates convolution Layer
    
    args:
        m_input: Input into the dense layer
        size_in: number of input neurons
        size_out: number of output neurons
        trainable_vars: Bool. Specifies if variables created in this dense layer are trainable
        name: Name of the dense layer
        num: dense layer number

    Returns:
        A tensor representing the activations of the layer
    """

    #He et al. Standardeviation
    sdev = np.power(2.0/(size_in*size_out),0.5)
    sdev = np.power(2.0/(size_in+size_out),0.5)
    print("sdev"+name+num+": ",sdev)
    
    with tf.name_scope(name+num):
        #Weights and biases
        w = tf.Variable(tf.truncated_normal([size_in, size_out],stddev=sdev,dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="b{}".format(num))
        z = tf.matmul(m_input,w)
        #act = tf.nn.relu(z+b,name="relu")
        act = tf.nn.leaky_relu(z+b,alpha=0.3,name=("act"+num))
        #act = tf.sigmoid(z+b,name="sigmoid")
        #act = tf.tanh((z+b),name="tanh")
        #Summaries
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return act

def final_linear_layer(m_input,size_in,size_out,trainable_vars,name="final",num="1"):
    sdev = np.power(2.0/(size_in+size_out),.5)
    print("sdev"+name+num+": ",sdev)

    with tf.name_scope(name+num):
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=sdev,dtype=tf.float16),
                        dtype=tf.float16,trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0,shape=[size_out],dtype=tf.float16),
                        dtype=tf.float16,trainable=trainable_vars,
                        name="b{}".format(num))

        act = tf.matmul(m_input,w)+b
        #act = tf.nn.softmax(act)
        tf.summary.histogram("weights",w)
        tf.summary.histogram("biases",b)
        tf.summary.histogram("act",act)
        return act
        
    
    
def get_place_holders():
    """
    Returns: The tensorflow name of the placeholder variables
    """
    a = tf.get_default_graph().get_tensor_by_name("infer_place_holder/x1:0")
    b = tf.get_default_graph().get_tensor_by_name("infer_place_holder/train_bool:0")
    c = tf.get_default_graph().get_tensor_by_name("train_place_holder/seq:0")
    return a,b,c


def build_graph(name,net_in,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,trainable_vars):
    """
    Build the complete tensorflow graph

    args:
        name: The namescope for the graph
        net_in: input to the graph
        conv_count: number of convolution layers
        fc_count: number of dense layers
        conv_feats: List containing the number of kernels per convolution layers
        fc_feats: List containing the number of neurons per dense layer
        conv_k_size: List containing the size of the convolution layers
        conv_strid: List containing the stride for each convolution layer
        trainable_vars: Bool. Specifies if the weights/bias variables in this graph are trainable

    Returns:
        Output Layer of the Graph
    """

    with tf.name_scope(name):
        #names of the convolution and dense layers
        conv_name="conv"
        fcs_name="FC"
        
        #Number of kernels/neurons in the first layer
        conv_feats[0] = 4

        
        #Building Convolution Layers
        with tf.name_scope("Convolution_Layers"):
            convs = []
            convs.append(net_in)
            p = 0

            #For loop calls conv_layer function and adds the returned max pool to convs list
            for i in range(0,conv_count-1):
                convs.append(conv_layer(convs[i],
                                        conv_feats[i],conv_feats[i+1],
                                        conv_k_size[p],conv_k_size[p],
                                        conv_stride[p],
                                        2,2,trainable_vars,
                                        conv_name,str(i+1)))
                p = p+1
            
            shp = convs[conv_count-1].get_shape().as_list()
            dim = np.prod(shp[1:])
            fc_feats[0] = dim
            #Flattening the final layer for input into dense layers
            #flatten = tf.reshape(convs[conv_count-1],[-1,fc_feats[0]])
            flatten = tf.reshape(convs[conv_count-1],[-1,dim])
        with tf.name_scope("Dense_Layers"):
            fcs = []
            fcs.append(flatten)

            #For loop calls fc_layer and add the activations to fcs list
            for i in range(0,fc_count-1):
                fcs.append(fc_layer(fcs[i],
                                    fc_feats[i],fc_feats[i+1],
                                    trainable_vars,fcs_name,str(i+1)))
            output_layer = fcs[len(fcs)-1]
            output_layer = final_linear_layer(output_layer,fc_feats[fc_count-1],4,trainable_vars,name=fcs_name,num=str(fc_count))
    return output_layer

        

def build_train_queue(batch_size,img_shp):
    """
    Builds FIFOQueue used in training

    args:
        batch_size: int. Training batch size
        img_shp: shape of the image
    
    Returns:
        Tensorflow FIFOQueue
    """
    with tf.name_scope("TrainQueue"):
        q = tf.FIFOQueue(capacity=25,
                         dtypes= (tf.uint8,tf.uint8,tf.float16,tf.uint8),
                         shapes= (img_shp,
                                  tf.TensorShape([batch_size,1]),
                                  tf.TensorShape([batch_size,1]),
                                  img_shp),
                         name="tq",shared_name="train_queue")
        
    return q



def standardize_img(img_array):
    """
    Applys a tensorflow map to standardize images
    
    args:
        img_array: A Tensor containing batch of images
    
    returns:
        Tensor containing Images that have been standardized and cast to float16
    """ 
    img_std = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), 
                        img_array,dtype=tf.float32,
                        parallel_iterations=4)
    return tf.cast(img_std,tf.float16)

def get_tensor(name):
    """
    args:
        name: String,Name of tensor
    Returns:
        Tensor
    """
    return tf.get_default_graph().get_tensor_by_name(name)

def build_update_infer_weights_op(conv_name,fc_name,conv_count,fc_count):
    """
    Function builds operations for updating weights of inference graph

    args:
        conv_name: String, Name of the Convolution Layers
        fc_name: String. Name of the Dense Layers
        conv_count: int. Number of convolution Layers
        fc_count: int. Number of dense Layers
    
    returns:
        List containing tensorflow assignment operations
    """

    num_conv = conv_count
    num_fc = fc_count+1
    
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

def infer_weight_update_ops(conv_name,fc_name,conv_count,fc_count):
    """
    Function builds operations for updating weights of inference graph

    args:
        conv_name: String, Name of the Convolution Layers
        fc_name: String. Name of the Dense Layers
        conv_count: int. Number of convolution Layers
        fc_count: int. Number of dense Layers
    
    returns:
        List containing tensorflow assignment operations
    """

    num_conv = conv_count
    num_fc = fc_count+1
    
    infer_conv_w = [get_tensor("Inference/Convolution_Layers/{}{}/w{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    infer_conv_b = [get_tensor("Inference/Convolution_Layers/{}{}/b{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    infer_fc_w = [get_tensor("Inference/Dense_Layers/{}{}/w{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    infer_fc_b = [get_tensor("Inference/Dense_Layers/{}{}/b{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    
    target_conv_w = [get_tensor("Target/Convolution_Layers/{}{}/w{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    target_conv_b = [get_tensor("Target/Convolution_Layers/{}{}/b{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    target_fc_w = [get_tensor("Target/Dense_Layers/{}{}/w{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    target_fc_b = [get_tensor("Target/Dense_Layers/{}{}/b{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]

    assign_ops_conv_w = [tf.assign(a,b) for a,b in zip(infer_conv_w,target_conv_w)]
    assign_ops_conv_b = [tf.assign(a,b) for a,b in zip(infer_conv_b,target_conv_b)]
    assign_ops_fc_w = [tf.assign(a,b) for a,b in zip(infer_fc_w,target_fc_w)]
    assign_ops_fc_b = [tf.assign(a,b) for a,b in zip(infer_fc_b,target_fc_b)]
    return [assign_ops_conv_w,assign_ops_conv_b,assign_ops_fc_w,assign_ops_fc_b]  

def target_weight_update_ops(conv_name,fc_name,conv_count,fc_count):
    """
     Function builds operations to update target network weights and biases

    args:
        conv_name: String, Name of the Convolution Layers
        fc_name: String. Name of the Dense Layers
        conv_count: int. Number of convolution Layers
        fc_count: int. Number of dense Layers
    
    returns:
        List containing tensorflow assignment operations

    """

    num_conv = conv_count
    num_fc = fc_count+1
    
    Target_conv_w = [get_tensor("Target/Convolution_Layers/{}{}/w{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    Target_conv_b = [get_tensor("Target/Convolution_Layers/{}{}/b{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    Target_fc_w = [get_tensor("Target/Dense_Layers/{}{}/w{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    Target_fc_b = [get_tensor("Target/Dense_Layers/{}{}/b{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    
    train_conv_w = [get_tensor("Train/Convolution_Layers/{}{}/w{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    train_conv_b = [get_tensor("Train/Convolution_Layers/{}{}/b{}:0".format(conv_name,i,i)) for i in range(1,num_conv)]
    train_fc_w = [get_tensor("Train/Dense_Layers/{}{}/w{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]
    train_fc_b = [get_tensor("Train/Dense_Layers/{}{}/b{}:0".format(fc_name,i,i)) for i in range(1,num_fc)]

    assign_ops_conv_w = [tf.assign(a,b) for a,b in zip(Target_conv_w,train_conv_w)]
    assign_ops_conv_b = [tf.assign(a,b) for a,b in zip(Target_conv_b,train_conv_b)]
    assign_ops_fc_w = [tf.assign(a,b) for a,b in zip(Target_fc_w,train_fc_w)]
    assign_ops_fc_b = [tf.assign(a,b) for a,b in zip(Target_fc_b,train_fc_b)]
    return [assign_ops_conv_w,assign_ops_conv_b,assign_ops_fc_w,assign_ops_fc_b]


def create_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    """
    This functions creates a the complete model. It is very similar to Infer_model/Train Model. Look under those functions
    for more information. This function is basically an accumulation of those two functions
    """
    if (len(conv_feats) != conv_count):
        return

    # 0 is Trainer....1 is inference

    tf.reset_default_graph()
    
    with tf.name_scope("infer_place_holder"):
        x1 = tf.placeholder(tf.uint8,shape=[1,110,84,4],name="x1")
 
    
    with tf.device("/job:local/task:0"):
        train_q = build_train_queue(batch_size,[None,11,84,4])

   
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
    """
    This function builds a tensorflow graph for the purpose of Inference. It is designed for distributed tensorflow use.

    args:
        learning_rate: Float. Initial Learning Rate
        batch_size: int. Batch size
        conv_count: int. Number of Convolution Layesr
        fc_count: int. Number of dense layers
        conv_feats: List containing the number of kernels at each layer
        fc_feats: List containing the number of neurons at each layer
        conv_k_size: List specifiying the shape fo the convolution kernels at each layer.
        conv_stride: List specifiying the stride for convolution at each layer.
    
    Returns:
        x1: Place holder fo the input image
        action: Tensor. The argmax of the output layer
        Qnext_val: Tensor. The reduce_max of the output layer
    """

    #Saftey check condition
    if (len(conv_feats) != conv_count):
        return
    
    #Placing the operations on worker 0. Not Cheif
    with tf.device("/job:worker/task:0"):
        with tf.name_scope("infer_place_holder"):
            x1 = tf.placeholder(tf.uint8,shape=[None,100,100,4],name="x1")

    
        #Slicing Image for unnecessary frames
        #sl_x1 = x1[:,10:110,:,:]

        std_img = standardize_img(x1)

        #Building graph for inference
        infer_output = build_graph("Inference",std_img,
                                    conv_count,fc_count,
                                    conv_feats,fc_feats,conv_k_size,conv_stride,False)
        
        #Some operations for using the model
        Qnext_val = tf.reduce_max(infer_output,name="Qnext_val")
        q_vals_pr = tf.Print(Qnext_val,[Qnext_val],"Qval: ")
        action = tf.argmax(infer_output,axis=1,name="action")
        return x1,action,Qnext_val
    
    
def train_model(learning_rate,batch_size,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    """
    This function builds a tensorflow graph for the purpose of Training. It is designed for distributed tensorflow use.

    args:
        learning_rate: Float. Initial Learning Rate
        gamma: Discount factor in Q-learning algorithm
        batch_size: int. Batch size
        conv_count: int. Number of Convolution Layesr
        fc_count: int. Number of dense layers
        conv_feats: List containing the number of kernels at each layer
        fc_feats: List containing the number of neurons at each layer
        conv_k_size: List specifiying the shape fo the convolution kernels at each layer.
        conv_stride: List specifiying the stride for convolution at each layer.
        LOGDIR: String. Location of Logs

    Returns:
        writer: Tensorflow summary file writer
        summ: Tensorflow summary merge
        train: Tensorflow train operation that executes training
        enqueue_op: Tensorflow operation for adding values to queue
        q_s1: Tensorflow print operation for printing size of queue
        s_img1: Image1 placeholder
        s_a: Action placeholder
        s_r: Reward Placeholder
        s_img2: Image2 placeholder
        ops: Update operations to update Inference graph weights/biases
    """
    #Safety Check requiremnt
    if (len(conv_feats) != conv_count):
        return
    
    #This model operations live on worker 1. Is Cheif
    with tf.device("/job:worker/task:1"):    
        with tf.name_scope("train_place_holder"):
            s_img1 = tf.placeholder(tf.uint8,shape=[batch_size,100,100,4],name="s_img1")
            s_a = tf.placeholder(tf.uint8,shape=[batch_size,1],name="s_a")
            s_r = tf.placeholder(tf.float16,shape=[batch_size,1],name="s_r")
            s_img2 = tf.placeholder(tf.uint8,shape=[batch_size,100,100,4],name="s_img2")
       
        #with tf.name_scope("slice_image"):
        #    sl_img1 = s_img1[:,10:110,:,:]
        #    sl_img2 = s_img2[:,10:110,:,:]

        #Building Queue Operations
        with tf.name_scope("train_queue"):
            train_q = build_train_queue(batch_size,s_img1.get_shape())
            enqueue_op = train_q.enqueue((s_img1,s_a,s_r,s_img2))
            p_queues = tf.Print(train_q.size(),[train_q.size()],message="Q Size1: ")
            img1,a,r,img2 = train_q.dequeue(name="dequeue")

        #Standardizing Images
        with tf.name_scope("Img_PreProc"):
            std_img1 = standardize_img(img1)
            std_img2 = standardize_img(img2)
            tf.summary.image("std_img1",std_img1)
            tf.summary.image("std_img2",std_img2)
        
        #Building Training model
        train_output = build_graph("Train",std_img1,
                                    conv_count,fc_count,
                                    conv_feats,fc_feats,conv_k_size,conv_stride,True)
        
        #Building Target model
        target_output = build_graph("Target",std_img2,
                                    conv_count,fc_count,
                                    conv_feats,fc_feats,conv_k_size,conv_stride,False)

        #Assignment operations for updating inference graph
        with tf.name_scope("Assignment_Ops"):
            with tf.name_scope("Infer_weight_update_ops"):
                infer_ops = infer_weight_update_ops("conv","FC",conv_count,fc_count)
        
            with tf.name_scope("Target_weight_update_op"):
                target_ops = target_weight_update_ops("conv","FC",conv_count,fc_count)

        global_step = tf.train.create_global_step()

        with tf.name_scope("Q_Algo"):
            qmax_idx = tf.argmax(target_output,axis=1,name="qmax_idx")
            gamma = tf.constant(0.99,tf.float16)
            idxs = tf.concat((tf.transpose([tf.range(0,batch_size,dtype=tf.int64)]),tf.transpose([qmax_idx])),axis=1)
            Qnext = tf.reduce_max(target_output,name="Qnext_target")
            target_q = tf.add(r,tf.multiply(gamma,Qnext),name="y")
            y = tf.Variable(tf.zeros(shape=target_output.get_shape(),dtype=tf.float16),trainable=False)
            assign_y = tf.assign(y,target_output)
            with tf.control_dependencies([assign_y]):
                tf.scatter_nd_update(y,tf.expand_dims(idxs,axis=1),target_q)
                tf.summary.histogram("y",y)

        with tf.name_scope("Trainer"):
            #Creating global step
            loss = tf.losses.huber_loss(y,train_output,delta=1.0)
            tf.summary.scalar("loss",loss)
            opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate,momentum=0.95,epsilon=.01)
            grads = opt.compute_gradients(loss)
            
            train = opt.apply_gradients(grads,global_step=global_step)
            p_delta = tf.Print([grads[1]],[grads[1]],message="grads: ")
        
        p_r = 0

        
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    return writer,summ,train,enqueue_op,p_queues,p_delta,s_img1,s_a,s_r,s_img2,infer_ops,target_ops,p_r,gamma,global_step


def flatten_weights_summarize(w,num,trainable):
    """
    This function flattens the weights of the convolution layers and adds them as images to summary
    """
    if (trainable):
        w_n = tf.transpose(w,perm=[2,0,1,3])
        w_shp = w_n.get_shape().as_list()
        for i in range(0,w_shp[len(w_shp)-1]):
            tf.summary.image("conv_w"+num,tf.expand_dims(w_n[:,:,:,i],axis=3))

def intermediate_summary_img(img,num,trainable):
    s_img = img[0,:,:,:]
    if (trainable):
        s_img_n = tf.expand_dims(tf.transpose(s_img,perm=[2,0,1]),axis=3)
        tf.summary.image("img"+num,s_img_n)
        """
        s_img_1 = tf.expand_dims(img[:,:,:,0],axis=3)
        s_img_2 = tf.expand_dims(img[:,:,:,1],axis=3)
        s_img_3 = tf.expand_dims(img[:,:,:,2],axis=3)
        s_img_4 = tf.expand_dims(img[:,:,:,3],axis=3)

        tf.summary.image("img"+num+"1",s_img_1)
        tf.summary.image("img"+num+"2",s_img_2)
        tf.summary.image("img"+num+"3",s_img_3)
        tf.summary.image("img"+num+"4",s_img_4)
        """
