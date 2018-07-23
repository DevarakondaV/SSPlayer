import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
server = tf.train.Server.create_local_server(config=config)
print(server.target)
server.join()