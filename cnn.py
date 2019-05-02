import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf
import h5py
from tensorflow.contrib.tensorboard.plugins import projector


class pdqn(tf.keras.Model):

    def __init__(self,seq_len, conv_feats, fc_feats, conv_k_size,
                    conv_stride, LOGDIR,gamma=0, batch_size=0, learning_rate = 0):
        """
        Function initializes the model
        args:
            learning_rate: Int. Learning rate associated with the model. Default model is inferenc only
            gamma: Float. Temporal difference learning discount parameter
            batch_size: int.
            seq_len: int.
            conv_feats: List of Ints. Specifies the number of kernels for each layer
            fc_feats: List of Ints. Specifies the number of nodes for each leayer.
            conv_k_size: List of ints. Specifies the cnn kernel size
            conv_stride: List of ints. Specifies the stride for each layer.
            LOGDIR: String. Tensorboard log dir

        """
        super(pdqn, self).__init__()
        self.layer_dict = {}
        self.s_writer = tf.contrib.summary.create_file_writer(
            logdir = LOGDIR
        )


        #Building layers
        self.build_layers("Tar",seq_len,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR)
        for i in self.layer_dict.keys():
            self.layer_dict[i].trainable=False
        self.build_layers("Tra",seq_len,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR)

        self.loss_fun = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                    momentum=0.95,
                                                    )
        # self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        return

    def update_target_weights(self):
        print("##### UPDATE PARAMS #####")
        self.layer_dict["Tar_cnn_layer0"].set_weights(self.layer_dict["Tra_cnn_layer0"].get_weights())
        self.layer_dict["Tar_cnn_layer1"].set_weights(self.layer_dict["Tra_cnn_layer1"].get_weights())
        self.layer_dict["Tar_cnn_layer2"].set_weights(self.layer_dict["Tra_cnn_layer2"].get_weights())
        self.layer_dict["Tar_dense0"].set_weights(self.layer_dict["Tra_dense0"].get_weights())
        self.layer_dict["Tar_dense1"].set_weights(self.layer_dict["Tra_dense1"].get_weights())
        self.layer_dict["Tar_fdense"].set_weights(self.layer_dict["Tra_fdense"].get_weights())


    def infer(self,inputs):
        print("##### INFERING #####")
        inputs[0] = inputs[0].astype(np.float32)
        inputs[0] = tf.image.per_image_standardization(inputs[0])
        Tar_d3,a = self.__call__(inputs = inputs)
        return Tar_d3,a

    def train(self,inputs,IS_weights,r):
        print("##### TRAINING #####")
        #Flip states before passing to network
        stdzd = (
            tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),inputs[3].astype(np.float32)),
            inputs[1].astype(np.int64),
            np.transpose(inputs[2].astype(np.float32)),
            tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),inputs[0].astype(np.float32))
        )
        
        with self.s_writer.as_default():
            with tf.contrib.eager.GradientTape() as tape:
                y,Tra_d3 = self.__call__(inputs=stdzd,training=True)
                loss = self.loss_fun(y,Tra_d3,IS_weights)
            
            grads = tape.gradient(loss,self.trainable_variables)
            grads_clip,global_norm = tf.clip_by_global_norm(grads,clip_norm = 1)
            self.optimizer.apply_gradients(
                zip(grads_clip,self.trainable_variables),
                global_step=tf.train.get_global_step()
            )
            
            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                tf.contrib.summary.scalar('Loss',loss)

        return y,Tra_d3

    def build_layers(self,model_pre, seq_len, conv_feats, fc_feats, conv_k_size,
                conv_stride, LOGDIR,gamma=0, batch_size=0, learning_rate = 0):

        for i in range(0, len(conv_feats)):
            layer = tf.keras.layers.Conv2D(filters=conv_feats[i],
                                        kernel_size=conv_k_size[i],
                                        strides=conv_stride[i],
                                        data_format="channels_last",
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        kernel_initializer='glorot_normal',
                                        bias_initializer='zeros')
            self.layer_dict[model_pre+"_cnn_layer"+str(i)] = layer

        self.layer_dict[model_pre+"_flatten"] = tf.keras.layers.Flatten()

        for i in range(0, len(fc_feats)):
            layer = tf.keras.layers.Dense(fc_feats[i],
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        kernel_initializer='glorot_normal',
                                        bias_initializer='zeros')
            self.layer_dict[model_pre+"_dense"+str(i)] = layer

        self.layer_dict[model_pre+"_fdense"] = tf.keras.layers.Dense(3,activation=None,use_bias=True,
                                                        kernel_initializer='glorot_normal',
                                                        bias_initializer='zeros')

    def set_model_weights(self,sdir):
        f = h5py.File(sdir,'r')
        lb = f['conv2d']['pdqn']['conv2d']['bias:0'][()]
        lw = f['conv2d']['pdqn']['conv2d']['kernel:0'][()]
        l1b = f['conv2d_1']['pdqn']['conv2d_1']['bias:0'][()]
        l1w = f['conv2d_1']['pdqn']['conv2d_1']['kernel:0'][()]
        l2b = f['conv2d_2']['pdqn']['conv2d_2']['bias:0'][()]
        l2w = f['conv2d_2']['pdqn']['conv2d_2']['kernel:0'][()]
        l3b = f['conv2d_3']['pdqn']['conv2d_3']['bias:0'][()]
        l3w = f['conv2d_3']['pdqn']['conv2d_3']['kernel:0'][()]
        l4b = f['conv2d_4']['pdqn']['conv2d_4']['bias:0'][()]
        l4w = f['conv2d_4']['pdqn']['conv2d_4']['kernel:0'][()]
        l5b = f['conv2d_5']['pdqn']['conv2d_5']['bias:0'][()]
        l5w = f['conv2d_5']['pdqn']['conv2d_5']['kernel:0'][()]

        db = f['dense']['pdqn']['dense']['bias:0'][()]
        dw = f['dense']['pdqn']['dense']['kernel:0'][()]
        d1b = f['dense_1']['pdqn']['dense_1']['bias:0'][()]
        d1w = f['dense_1']['pdqn']['dense_1']['kernel:0'][()]
        d2b = f['dense_2']['pdqn']['dense_2']['bias:0'][()]
        d2w = f['dense_2']['pdqn']['dense_2']['kernel:0'][()]
        d3b = f['dense_3']['pdqn']['dense_3']['bias:0'][()]
        d3w = f['dense_3']['pdqn']['dense_3']['kernel:0'][()]
        d4b = f['dense_4']['pdqn']['dense_4']['bias:0'][()]
        d4w = f['dense_4']['pdqn']['dense_4']['kernel:0'][()]
        d5b = f['dense_5']['pdqn']['dense_5']['bias:0'][()]
        d5w = f['dense_5']['pdqn']['dense_5']['kernel:0'][()]
        f.close()

        #Target network
        self.layer_dict["Tar_cnn_layer0"].set_weights([lw,lb])
        self.layer_dict["Tar_cnn_layer1"].set_weights([l1w,l1b])
        self.layer_dict["Tar_cnn_layer2"].set_weights([l2w,l2b])
        self.layer_dict["Tar_dense0"].set_weights([dw,db])
        self.layer_dict["Tar_dense1"].set_weights([d1w,d1b])
        self.layer_dict["Tar_fdense"].set_weights([d2w,d2b])


        #Training network
        self.layer_dict["Tra_cnn_layer0"].set_weights([l3w,l3b])
        self.layer_dict["Tra_cnn_layer1"].set_weights([l4w,l4b])
        self.layer_dict["Tra_cnn_layer2"].set_weights([l5w,l5b])
        self.layer_dict["Tra_dense0"].set_weights([d3w,d3b])
        self.layer_dict["Tra_dense1"].set_weights([d4w,d4b])
        self.layer_dict["Tra_fdense"].set_weights([d5w,d5b])


        return

    def call(self, inputs,training=False):
        print("Maximum",np.max(inputs[0].numpy()))

        norm_Tar_s1 = tf.convert_to_tensor(inputs[0])

        Tar_cl1 = self.layer_dict["Tar_cnn_layer0"](norm_Tar_s1)
        Tar_cl2 = self.layer_dict["Tar_cnn_layer1"](Tar_cl1)
        Tar_cl3 = self.layer_dict["Tar_cnn_layer2"](Tar_cl2)
        Tar_fl = self.layer_dict["Tar_flatten"](Tar_cl3)
        Tar_d1 = self.layer_dict["Tar_dense0"](Tar_fl)
        Tar_d2 = self.layer_dict["Tar_dense1"](Tar_d1)
        Tar_d3 = self.layer_dict["Tar_fdense"](Tar_d2)

        #if not training return this mode
        if (not training):
            return Tar_d3,tf.keras.backend.argmax(Tar_d3)

        norm_Tra_s2 = tf.convert_to_tensor(inputs[3])
        r = tf.convert_to_tensor(inputs[2])
        
        Tra_cl1 = self.layer_dict["Tra_cnn_layer0"](norm_Tra_s2)
        Tra_cl2 = self.layer_dict["Tra_cnn_layer1"](Tra_cl1)
        Tra_cl3 = self.layer_dict["Tra_cnn_layer2"](Tra_cl2)
        Tra_fl = self.layer_dict["Tra_flatten"](Tra_cl3)
        Tra_d1 = self.layer_dict["Tra_dense0"](Tra_fl)
        Tra_d2 = self.layer_dict["Tra_dense1"](Tra_d1)
        Tra_d3 = self.layer_dict["Tra_fdense"](Tra_d2)


        Qmax = tf.keras.backend.max(Tar_d3,axis=1)
        y = r+0.99*Qmax
        update_idx = [[i,a[0]] for i,a in zip(range(0,len(inputs[1])),inputs[1])]
        y = tf.clip_by_value(y,-1,1)
        print("Before y",y)

        y = y.numpy()[0]
        y = [inputs[2][0][i] if inputs[2][0][i] == -1 else y[i] for i in range(0,len(inputs[2][0]))]         
 
        y_scr = tf.scatter_nd_update(tf.Variable(Tra_d3),indices=update_idx,updates=y)
        # print("Update idx", update_idx)
        # print("Qmax",Qmax)
        # print("reward",r)
        # print("y",y)
        # print("Targ",Tar_d3.numpy())
        # print("Tarin",Tra_d3.numpy())
        # print("Yscr",y_scr.numpy())         

        #with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            #tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,0],-1))
            #tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,1],-1))
            #tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,2],-1))
            #tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,3],-1))

        return y_scr,Tra_d3
