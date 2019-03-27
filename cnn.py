import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def conv_layer(m_input, size_in, size_out, k_size_w, k_size_h, conv_stride, pool_k_size, pool_stride_size, trainable_vars, name, num):
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
    sdev = np.power(2.0/(k_size_w*k_size_h*size_in), 0.5)
    #Xavier Initialization
    sdev = np.power(2.0/(size_in+size_out), 0.5)
    #print("sdev"+name+num+": ",sdev)

    with tf.name_scope(name+num):

        #Weight and bias initializations
        w = tf.Variable(tf.truncated_normal([k_size_w, k_size_h, size_in, size_out], stddev=sdev, dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0, shape=[size_out], dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="b{}".format(num))

        #Convolution and activations
        conv = tf.nn.conv2d(m_input, w, strides=[
                            1, conv_stride, conv_stride, 1], padding="SAME")
        #act = tf.nn.relu((conv+b),name="relu")
        act = tf.nn.leaky_relu((conv+b), alpha=0.3)
        #act = tf.sigmoid((conv+b),name="sigmoid")
        #act = tf.tanh((conv+b),name="tanh")
        #intermediate_summary_img(act,num,trainable_vars)

        #summaries
        #tf.summary.histogram("weights",w)
        #flatten_weights_summarize(w,num,trainable_vars)
        #tf.summary.histogram("biases",b)
        #tf.summary.histogram("act",act)
        return tf.nn.max_pool(act, ksize=[1, pool_k_size, pool_k_size, 1], strides=[1, pool_stride_size, pool_stride_size, 1], padding='SAME')


def fc_layer(m_input, size_in, size_out, trainable_vars, name, num):
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
    sdev = np.power(2.0/(size_in*size_out), 0.5)
    sdev = np.power(2.0/(size_in+size_out), 0.5)
    #print("sdev"+name+num+": ",sdev)

    with tf.name_scope(name+num):
        #Weights and biases
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=sdev, dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0, shape=[size_out], dtype=tf.float16),
                        dtype=tf.float16,
                        trainable=trainable_vars,
                        name="b{}".format(num))
        z = tf.matmul(m_input, w)
        #act = tf.nn.relu(z+b,name="relu")
        act = tf.nn.leaky_relu(z+b, alpha=0.3, name=("act"+num))
        #act = tf.sigmoid(z+b,name="sigmoid")
        #act = tf.tanh((z+b),name="tanh")
        #Summaries
        #tf.summary.histogram("weights",w)
        #tf.summary.histogram("biases",b)
        #tf.summary.histogram("act",act)
        return act


def final_linear_layer(m_input, size_in, size_out, trainable_vars, name="final", num="1"):
    sdev = np.power(2.0/(size_in+size_out), .5)
    #print("sdev"+name+num+": ",sdev)

    with tf.name_scope(name+num):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=sdev, dtype=tf.float16),
                        dtype=tf.float16, trainable=trainable_vars,
                        name="w{}".format(num))
        b = tf.Variable(tf.constant(0.0, shape=[size_out], dtype=tf.float16),
                        dtype=tf.float16, trainable=trainable_vars,
                        name="b{}".format(num))

        act = tf.matmul(m_input, w)+b
        #act = tf.nn.softmax(act)
        #tf.summary.histogram("weights",w)
        #tf.summary.histogram("biases",b)
        #tf.summary.histogram("act",act)
        return act


def build_graph(name, net_in, conv_count, fc_count, conv_feats, fc_feats, conv_k_size, conv_stride, trainable_vars):
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
        conv_name = "conv"
        fcs_name = "FC"

        #Number of kernels/neurons in the first layer
        #conv_feats[0] = 10

        #Building Convolution Layers
        with tf.name_scope("Convolution_Layers"):
            convs = []
            convs.append(net_in)
            p = 0

            #For loop calls conv_layer function and adds the returned max pool to convs list
            for i in range(0, conv_count-1):
                convs.append(conv_layer(convs[i],
                                        conv_feats[i], conv_feats[i+1],
                                        conv_k_size[p], conv_k_size[p],
                                        conv_stride[p],
                                        2, 2, trainable_vars,
                                        conv_name, str(i+1)))
                p = p+1

            shp = convs[conv_count-1].get_shape().as_list()
            dim = np.prod(shp[1:])
            fc_feats[0] = dim
            #Flattening the final layer for input into dense layers
            #flatten = tf.reshape(convs[conv_count-1],[-1,fc_feats[0]])
            flatten = tf.reshape(convs[conv_count-1], [-1, dim])
        with tf.name_scope("Dense_Layers"):
            fcs = []
            fcs.append(flatten)

            #For loop calls fc_layer and add the activations to fcs list
            for i in range(0, fc_count-1):
                print(fcs_name+str(i+1)+" "+str(fc_feats[i]), trainable_vars)
                fcs.append(fc_layer(fcs[i],
                                    fc_feats[i], fc_feats[i+1],
                                    trainable_vars, fcs_name, str(i+1)))

            output_layer = fcs[len(fcs)-1]
            output_layer = final_linear_layer(
                output_layer, fc_feats[fc_count-1], 3, trainable_vars, name=fcs_name, num=str(fc_count))
    return output_layer


def standardize_img(img_array):
    """
    Applys a tensorflow map to standardize images
    
    args:
        img_array: A Tensor containing batch of images
    
    returns:
        Tensor containing Images that have been standardized and cast to float16
    """
    img_std = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),
                        img_array, dtype=tf.float32,
                        parallel_iterations=4)

    return tf.cast(img_std, tf.float16)


def get_tensor(name):
    """
    args:
        name: String,Name of tensor
    Returns:
        Tensor
    """
    return tf.get_default_graph().get_tensor_by_name(name)


def build_update_infer_weights_op(conv_name, fc_name, conv_count, fc_count):
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

    infer_conv_w = [get_tensor(
        "Inference/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    infer_conv_b = [get_tensor(
        "Inference/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    infer_fc_w = [get_tensor(
        "Inference/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    infer_fc_b = [get_tensor(
        "Inference/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    train_conv_w = [get_tensor(
        "Train/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    train_conv_b = [get_tensor(
        "Train/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    train_fc_w = [get_tensor(
        "Train/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    train_fc_b = [get_tensor(
        "Train/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    assign_ops_conv_w = [tf.assign(a, b)
                         for a, b in zip(infer_conv_w, train_conv_w)]
    assign_ops_conv_b = [tf.assign(a, b)
                         for a, b in zip(infer_conv_b, train_conv_b)]
    assign_ops_fc_w = [tf.assign(a, b) for a, b in zip(infer_fc_w, train_fc_w)]
    assign_ops_fc_b = [tf.assign(a, b) for a, b in zip(infer_fc_b, train_fc_b)]
    return [assign_ops_conv_w, assign_ops_conv_b, assign_ops_fc_w, assign_ops_fc_b]


def infer_weight_update_ops(conv_name, fc_name, conv_count, fc_count):
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

    infer_conv_w = [get_tensor(
        "Inference/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    infer_conv_b = [get_tensor(
        "Inference/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    infer_fc_w = [get_tensor(
        "Inference/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    infer_fc_b = [get_tensor(
        "Inference/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    target_conv_w = [get_tensor(
        "Target/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    target_conv_b = [get_tensor(
        "Target/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    target_fc_w = [get_tensor(
        "Target/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    target_fc_b = [get_tensor(
        "Target/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    assign_ops_conv_w = [tf.assign(a, b)
                         for a, b in zip(infer_conv_w, target_conv_w)]
    assign_ops_conv_b = [tf.assign(a, b)
                         for a, b in zip(infer_conv_b, target_conv_b)]
    assign_ops_fc_w = [tf.assign(a, b)
                       for a, b in zip(infer_fc_w, target_fc_w)]
    assign_ops_fc_b = [tf.assign(a, b)
                       for a, b in zip(infer_fc_b, target_fc_b)]
    return [assign_ops_conv_w, assign_ops_conv_b, assign_ops_fc_w, assign_ops_fc_b]


def get_all_weights_bias_tensors(conv_count, fc_count):
    """
    Function returns the tensors of all weights and biases in network

    args: None

    Returns:
        A list of all tensors that are weights and biases for the network
    """

    conv_name = "conv"
    fc_name = "FC"
    num_conv = conv_count
    num_fc = fc_count+1

    rtn_list = []

    infer_conv_w = [get_tensor(
        "Inference/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    infer_conv_b = [get_tensor(
        "Inference/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    infer_fc_w = [get_tensor(
        "Inference/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    infer_fc_b = [get_tensor(
        "Inference/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    target_conv_w = [get_tensor(
        "Target/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    target_conv_b = [get_tensor(
        "Target/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    target_fc_w = [get_tensor(
        "Target/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    target_fc_b = [get_tensor(
        "Target/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    train_conv_w = [get_tensor(
        "Train/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    train_conv_b = [get_tensor(
        "Train/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    train_fc_w = [get_tensor(
        "Train/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    train_fc_b = [get_tensor(
        "Train/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    rtn_list = infer_conv_w + infer_conv_b + infer_fc_w + infer_fc_b
    rtn_list = rtn_list + target_conv_w + target_conv_b + target_fc_w + target_fc_b
    rtn_list = rtn_list + train_conv_w + train_conv_b + train_fc_w + train_fc_b
    return rtn_list


def target_weight_update_ops(conv_name, fc_name, conv_count, fc_count):
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

    Target_conv_w = [get_tensor(
        "Target/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    Target_conv_b = [get_tensor(
        "Target/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    Target_fc_w = [get_tensor(
        "Target/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    Target_fc_b = [get_tensor(
        "Target/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    train_conv_w = [get_tensor(
        "Train/Convolution_Layers/{}{}/w{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    train_conv_b = [get_tensor(
        "Train/Convolution_Layers/{}{}/b{}:0".format(conv_name, i, i)) for i in range(1, num_conv)]
    train_fc_w = [get_tensor(
        "Train/Dense_Layers/{}{}/w{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]
    train_fc_b = [get_tensor(
        "Train/Dense_Layers/{}{}/b{}:0".format(fc_name, i, i)) for i in range(1, num_fc)]

    assign_ops_conv_w = [tf.assign(a, b)
                         for a, b in zip(Target_conv_w, train_conv_w)]
    assign_ops_conv_b = [tf.assign(a, b)
                         for a, b in zip(Target_conv_b, train_conv_b)]
    assign_ops_fc_w = [tf.assign(a, b)
                       for a, b in zip(Target_fc_w, train_fc_w)]
    assign_ops_fc_b = [tf.assign(a, b)
                       for a, b in zip(Target_fc_b, train_fc_b)]
    return [assign_ops_conv_w, assign_ops_conv_b, assign_ops_fc_w, assign_ops_fc_b]


def pdqn(learning_rate, gamma, batch_size, seq_len, conv_feats, fc_feats, conv_k_size, conv_stride, LOGDIR):
    """
    Priority DQN implementation with keras
    """
    state_one = tf.keras.Input(
        shape=(84, 84, seq_len), batch_size=batch_size, dtype=tf.uint8)
    state_two = tf.keras.Input(
        shape=(84, 84, seq_len), batch_size=batch_size, dtype=tf.uint8)

    inference = tf.keras.models.Sequential()

    #train = tf.keras.model.Sequential()
    #Conv_layers
    with tf.variable_scope("Inference"):
        for i in range(0, len(conv_feats)):
            if i == 0:
                layer = tf.keras.layers.Conv3D(filters=conv_feats[i],
                                               input_shape=state_one.shape,
                                               kernel_size=conv_k_size[i],
                                               strides=conv_stride[i],
                                               data_format="channels_last",
                                               activation=tf.nn.relu,
                                               use_bias=True,
                                               kernel_initializer=tf.keras.initializers.glorot_normal,
                                               bias_initializer=tf.keras.initializers.Zeros)
            else:
                layer = tf.keras.layers.Conv3D(filters=conv_feats[i],
                                               kernel_size=conv_k_size[i],
                                               strides=conv_stride[i],
                                               data_format="channels_last",
                                               activation=tf.nn.relu,
                                               use_bias=True,
                                               kernel_initializer=tf.keras.initializers.glorot_normal,
                                               bias_initializer=tf.keras.initializers.Zeros)
            inference.add(layer)

        inference.add(tf.keras.layers.Flatten())

        for i in range(1, len(fc_feats)):
            layer = tf.keras.layers.Dense(fc_feats[i],
                                          activation=tf.nn.relu,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.glorot_normal,
                                          bias_initializer=tf.keras.initializers.Zeros)
            inference.add(layer)

        inference.add(tf.keras.layers.Dense(3, activation=None, use_bias=True,
                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                            bias_initializer=tf.keras.initializers.Zeros))

    tf.keras.callbacks.TensorBoard(
        log_dir=LOGDIR, write_graph=True, histogram_freq=0).set_model(inference)
    return inference
        
    


def construct_two_network_model(learning_rate,gamma,batch_size,seq_len,conv_count,fc_count,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR):
    """
    This functions creates a non-distributed tensorflow model for inference and training with only two networks

    args:
        learning_rate: float. Learning rate for the tensorflow model
        gamma: float. Gamma in q learning
        batch_size: Size of batch
        seq_len: int. Batch size for training
        conv_count: int. number of convolution layers
        fc_count: int. number of dense layers

    returns:
        rtn_vals: Dictionary of required returns. Operations and tensors!
    """

    #Global step for training purposes
    global_step = tf.train.create_global_step()
    writer = tf.summary.FileWriter(LOGDIR)
    
    #Safety Conditional check
    if (len(conv_feats) != conv_count):
        return

    #return values
    rtn_vals = {}

    #Creating placeholders
    with tf.name_scope("place_holders"):
        s1 = tf.placeholder(tf.uint8,shape=[None,84,84,None],name='s1')
        s2 = tf.placeholder(tf.uint8,shape=[None,84,84,None],name='s2')
        r = tf.placeholder(tf.float16,shape=[None,1],name="r")
        IS_weights = tf.placeholder(tf.float16,shape=[None,1],name="IS_weights")

    tf.summary.histogram("rewards",r)

    #Standardizing images
    with tf.name_scope("image_pre_proc"):
        std_img_s1 = standardize_img(s1)
        std_img_s2 = standardize_img(s2)
        #tf.summary.image("std_img_1",std_img_s1)
        #tf.summary.image("std_img_2",std_img_s2)


    #pad tensor with fake batches if it's inference [Because inference is only single image]
    paddings = tf.constant([[0,batch_size-1],[0,0],[0,0],[0,0]])
    def fn_true():
        #If it is inference pad the image
        padded_img = tf.pad(std_img_s1,paddings,mode='CONSTANT',constant_values=0)
        sum_img = padded_img[0][:,:,seq_len-1]
        sum_img = tf.expand_dims(sum_img,0)
        sum_img = tf.expand_dims(sum_img,3)
        return [padded_img,sum_img]
    

    #input_img,sum_img = tf.cond(tf.equal(tf.shape(s1)[0],1),fn_true,lambda: [std_img_s2,tf.zeros(shape=[1,100,100,1],dtype=tf.float16)])
    input_img,sum_img = tf.cond(tf.equal(tf.shape(s1)[0],1),fn_true,lambda: [std_img_s2,tf.expand_dims(tf.expand_dims(std_img_s2[0][:,:,seq_len-1],2),0)])
    tf.summary.image("Image",sum_img)

    #Building graph for inference
    inference_out = build_graph("Target",input_img,
                                conv_count,fc_count,
                                conv_feats,fc_feats,conv_k_size,conv_stride,"False")
    
    #Building graph for training
    train_out = build_graph("Train",std_img_s2,
                            conv_count,fc_count,
                            conv_feats,fc_feats,conv_k_size,conv_stride,"True")
    

    #Assignment operations for updating inference graph
    with tf.name_scope("Assignment_Ops"):
        with tf.name_scope("Target_weight_update_op"):
            target_ops = target_weight_update_ops("conv","FC",conv_count,fc_count)

    with tf.name_scope("action"):
        action = tf.argmax(inference_out[0],axis=0,name="action")
        #tf.summary.scalar("Action: ",action)

    #implementing Q algorithm
    with tf.name_scope("Q_Algo"):
        qmax_idx = tf.argmax(inference_out,axis=1,name="qmax_idx", output_type=tf.int64)
        inf_shape = tf.shape(inference_out,out_type=tf.int64)
        qm_shape = tf.shape(qmax_idx,out_type=tf.int64)
        idxs = tf.concat((tf.transpose([tf.range(0,qm_shape[0],dtype=tf.int64)]),tf.transpose([qmax_idx])),axis=1)
        gamma = tf.constant(0.99,shape=[batch_size],dtype=tf.float16)#([seq_len],0.99)
        gamma_sparse = tf.SparseTensor(idxs,gamma,dense_shape=inf_shape)
        reward_sparse = tf.SparseTensor(idxs,tf.squeeze(r,axis=1),dense_shape=inf_shape)
        gamma_dense = tf.sparse_tensor_to_dense(gamma_sparse,1)
        reward_dense = tf.sparse_tensor_to_dense(reward_sparse,0)
        y = tf.add(reward_dense,tf.multiply(gamma_dense,inference_out))
        
    

    with tf.name_scope("Trainer"):
        TD_error = tf.reduce_max(y,axis=1)-tf.reduce_max(train_out,axis=1)
        ct = tf.concat([y,train_out],axis=0)
        
        #loss = tf.losses.huber_loss(y,train_out,delta=1.0,weights=IS_weights)
        loss = tf.losses.huber_loss(y,train_out,delta=1.0)
        tf.summary.scalar("loss",loss)
        opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate,momentum=0.95,epsilon=.01)
        grads = opt.compute_gradients(loss)
        train = opt.apply_gradients(grads,global_step=global_step,name='train')
        prt1 = tf.Print(train_out,[train_out],"trainout: ",name="prt1",summarize=100)  
        prt2 = tf.Print(y,[y],"y: ",name="prt2",summarize=100)

    summ = tf.summary.merge_all()
    #tf.summary.merge([summary_var1, summary_var2])

    rtn_vals['s1'] = s1
    rtn_vals['s2'] = s2
    rtn_vals['r'] = r
    rtn_vals['IS_weights'] = IS_weights
    rtn_vals['target_ops'] = target_ops
    rtn_vals['loss'] = loss
    rtn_vals['TD_error'] = TD_error
    
    rtn_vals['train'] = train
    rtn_vals['print1'] = prt1
    rtn_vals['print2'] = prt2

    rtn_vals['action'] = action
    #rtn_vals['op'] = op[0]
    
    rtn_vals['summ'] = summ
    rtn_vals['writer'] = writer
    rtn_vals['global_step'] = global_step
    rtn_vals['gamma'] = gamma

    return rtn_vals

    





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
        with tf.name_scope("inter_imgs"):
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
