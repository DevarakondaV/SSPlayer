import numpy as np
from scipy.stats import truncnorm as tn
import tensorflow as tf
import h5py


class pdqn(tf.keras.Model):

    def __init__(self, seq_len, conv_feats, fc_feats, conv_k_size,
                 conv_stride, LOGDIR, gamma=0, batch_size=0, learning_rate=0):
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
            logdir=LOGDIR
        )

        # Building layers
        self.build_layers("Tar", seq_len, conv_feats, fc_feats,
                          conv_k_size, conv_stride, LOGDIR)
        for i in self.layer_dict.keys():
            self.layer_dict[i].trainable = False
        self.build_layers("Tra", seq_len, conv_feats, fc_feats,
                          conv_k_size, conv_stride, LOGDIR)

        self.loss_fun = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        return

    def update_target_weights(self):
        print("##### UPDATE PARAMS #####")

        Tra_weights = {}
        for key in self.layer_dict:
            if "Tar" in key:
                continue
            Tra_weights[key] = self.layer_dict[key].get_weights()

        for key in Tra_weights:
            Tar_key = key.replace("Tra", "Tar")
            print(key, Tar_key)
            self.layer_dict[Tar_key].set_weights(Tra_weights[key])

    def infer(self, inputs, TSNE=False):
        print("##### INFERING #####")
        inputs[0] = inputs[0].astype(np.float32)
        inputs[0] = tf.image.per_image_standardization(inputs[0])
        if not TSNE:
            Tar_d3, a = self.__call__(inputs=inputs)
            return Tar_d3, a
        else:
            Tar_d3, a, Tar_d2 = self.__call__(inputs=inputs, TSNE=TSNE)
            return Tar_d3, a, Tar_d2

    def train(self, inputs, IS_weights, r):
        print("##### TRAINING #####")
        # Flip states before passing to network
        stdzd = (
            tf.map_fn(lambda frame: tf.image.per_image_standardization(
                frame), inputs[3].astype(np.float32)),
            inputs[1].astype(np.int64),
            np.transpose(inputs[2].astype(np.float32)),
            tf.map_fn(lambda frame: tf.image.per_image_standardization(
                frame), inputs[0].astype(np.float32))
        )

        with self.s_writer.as_default():
            with tf.contrib.eager.GradientTape() as tape:
                y, Tra_d3 = self.__call__(inputs=stdzd, training=True)
                loss = self.loss_fun(y, Tra_d3, IS_weights)

            grads = tape.gradient(loss, self.trainable_variables)
            grads_clip, global_norm = tf.clip_by_global_norm(
                grads, clip_norm=1)
            self.optimizer.apply_gradients(
                zip(grads_clip, self.trainable_variables),
                global_step=tf.train.get_global_step()
            )

            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                tf.contrib.summary.scalar('Loss', loss)

        return y, Tra_d3

    def build_layers(self, model_pre, seq_len, conv_feats, fc_feats, conv_k_size,
                     conv_stride, LOGDIR, gamma=0, batch_size=0, learning_rate=0):

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

        self.layer_dict[model_pre+"_fdense"] = tf.keras.layers.Dense(3, activation=None, use_bias=True,
                                                                     kernel_initializer='glorot_normal',
                                                                     bias_initializer='zeros')

    def set_model_weights(self, sdir):
        tar_count_cnn = len(
            [key for key in self.layer_dict.keys() if "Tar_cnn_layer" in key])
        tra_count_cnn = len(
            [key for key in self.layer_dict.keys() if "Tra_cnn_layer" in key])
        tar_count_dense = len(
            [key for key in self.layer_dict.keys() if "Tar_dense" in key])
        tra_count_dense = len(
            [key for key in self.layer_dict.keys() if "Tra_dense" in key])

        f = h5py.File(sdir, 'r')

        conv2d = "conv2d"
        for i in range(0, tar_count_cnn):
            if i != 0:
                conv2d = "conv2d_"+str(i)
            self.layer_dict["Tar_cnn_layer"+str(i)].set_weights([
                f[conv2d]['pdqn'][conv2d]['kernel:0'][()],
                f[conv2d]['pdqn'][conv2d]['bias:0'][()]
            ])

        conv2d = "conv2d"
        for i in range(0, tra_count_cnn):
            if i != 0:
                conv2d = "conv2d_"+str(i)
            self.layer_dict["Tra_cnn_layer"+str(i)].set_weights([
                f[conv2d]['pdqn'][conv2d]['kernel:0'][()],
                f[conv2d]['pdqn'][conv2d]['bias:0'][()]
            ])

        dense = "dense"
        for i in range(0, tar_count_dense+1):
            if i != 0:
                dense = "dense_"+str(i)
            layer_key = "Tar_dense" + \
                str(i) if i < tar_count_dense else "Tar_fdense"
            self.layer_dict[layer_key].set_weights([
                f[dense]['pdqn'][dense]['kernel:0'][()],
                f[dense]['pdqn'][dense]['bias:0'][()]
            ])

        dense = "dense"
        for i in range(0, tra_count_dense+1):
            if i != 0:
                dense = "dense_"+str(i)
            layer_key = "Tra_dense" + \
                str(i) if i < tra_count_dense else "Tra_fdense"
            self.layer_dict[layer_key].set_weights([
                f[dense]['pdqn'][dense]['kernel:0'][()],
                f[dense]['pdqn'][dense]['bias:0'][()]
            ])

        f.close()
        return

    def call(self, inputs, training=False, TSNE=False):

        Tar_inp = tf.convert_to_tensor(inputs[0])
        for key in self.layer_dict:
            if "Tar" not in key:
                continue
            Tar_out = self.layer_dict[key](Tar_inp)
            Tar_inp = Tar_out
            if TSNE and key == "Tar_dense1":
                Tar_d2 = Tar_out
            # print("output",Tar_out.shape)

        # if not training return this mode
        if (not training):
            if not TSNE:
                return Tar_out, tf.keras.backend.argmax(Tar_out)
            else:
                return Tar_out, tf.keras.backend.argmax(Tar_out), Tar_d2

        Tra_inp = tf.convert_to_tensor(inputs[3])
        r = tf.convert_to_tensor(inputs[2])
        for key in self.layer_dict:
            if "Tra" not in key:
                continue
            Tra_out = self.layer_dict[key](Tra_inp)
            Tra_inp = Tra_out

        Qmax = tf.keras.backend.max(Tar_out, axis=1)
        y = r+0.99*Qmax
        update_idx = [[i, a[0]]
                      for i, a in zip(range(0, len(inputs[1])), inputs[1])]
        y = tf.clip_by_value(y, -1, 1)
        y = y.numpy()[0]
        y = [inputs[2][0][i] if inputs[2][0][i] == -1 else y[i]
             for i in range(0, len(inputs[2][0]))]

        y_scr = tf.scatter_nd_update(tf.Variable(
            Tra_out), indices=update_idx, updates=y)
        # print("Update idx", update_idx)
        # print("Qmax",Qmax)
        # print("reward",r)
        # print("y",y)
        # print("Targ",Tar_d3.numpy())
        # print("Tarin",Tra_d3.numpy())
        # print("Yscr",y_scr.numpy())

        # with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        # tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,0],-1))
        # tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,1],-1))
        # tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,2],-1))
        # tf.contrib.summary.image('s1',tf.expand_dims(norm_Tar_s1[:,:,:,3],-1))

        return y_scr, Tra_out
