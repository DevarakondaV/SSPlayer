import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf
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



        self.build_layers("Tar",seq_len,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR)

        for i in self.layer_dict.keys():
            self.layer_dict[i].trainable=False

        self.build_layers("Tra",seq_len,conv_feats,fc_feats,conv_k_size,conv_stride,LOGDIR)

        return

    def build_layers(self,model_pre, seq_len, conv_feats, fc_feats, conv_k_size,
                conv_stride, LOGDIR,gamma=0, batch_size=0, learning_rate = 0):

        for i in range(0, len(conv_feats)):
            layer = tf.keras.layers.Conv2D(filters=conv_feats[i],
                                        kernel_size=conv_k_size[i],
                                        strides=conv_stride[i],
                                        data_format="channels_last",
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Zeros)
            self.layer_dict[model_pre+"_cnn_layer"+str(i)] = layer

        self.layer_dict[model_pre+"_flatten"] = tf.keras.layers.Flatten()

        for i in range(0, len(fc_feats)):
            layer = tf.keras.layers.Dense(fc_feats[i],
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Zeros)
            self.layer_dict[model_pre+"_dense"+str(i)] = layer

        self.layer_dict[model_pre+"_fdense"] = tf.keras.layers.Dense(3,activation=None,use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                                        bias_initializer=tf.keras.initializers.Zeros)

    def call(self, inputs,training=False):
        Tar_s1 = tf.convert_to_tensor(inputs[0],dtype=tf.float16)
        norm_Tar_s1 = tf.keras.utils.normalize(Tar_s1)
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

        Tra_s2 = tf.convert_to_tensor(inputs[3])
        norm_Tra_s2 = tf.keras.utils.normalize(Tra_s2)

        r = tf.convert_to_tensor(inputs[2])
        
        Tra_cl1 = self.layer_dict["Tra_cnn_layer0"](norm_Tra_s2)
        Tra_cl2 = self.layer_dict["Tra_cnn_layer1"](Tra_cl1)
        Tra_cl3 = self.layer_dict["Tra_cnn_layer2"](Tra_cl2)
        Tra_fl = self.layer_dict["Tra_flatten"](Tra_cl3)
        Tra_d1 = self.layer_dict["Tra_dense0"](Tra_fl)
        Tra_d2 = self.layer_dict["Tra_dense1"](Tra_d1)
        Tra_d3 = self.layer_dict["Tra_fdense"](Tra_d2)


        Qmax = tf.keras.backend.max(Tar_d3,axis=0)
        y = r+0.99*Qmax
        return y,Tra_d3
