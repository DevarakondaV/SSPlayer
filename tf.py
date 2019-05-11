import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly() 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)



import sys
import numpy as np
from cnn import *
from trainer import *
import json
import os

#os.chdir(r"C:\Users\devar\Documents\EngProj\SSPlayer\\")

with open("meta.json","r") as params_file:
    data = json.load(params_file)
    #print(data)

LOGDIR = data["logdir"] if data["pc"] == 1 else data["plogdir"]
save_steps = data["save_steps"]

#network params
batch_size = data["batchsize"]
seq_len = data["seq_len"]
conv_k_size = [i["l"+str(idx+1)] for i,idx in zip(data["conv_k_size"],range(0,len(data["conv_k_size"])))]
conv_stride = [i["l"+str(idx+1)] for i,idx in zip(data["conv_stride"],range(0,len(data["conv_stride"])))]
conv = [i["l"+str(idx+1)] for i,idx in zip(data["conv"],range(0,len(data["conv"])))]
fclyr = [i["l"+str(idx+1)] for i,idx in zip(data["fclyr"],range(0,len(data["fclyr"])))]
learning_rate = data["learing_rate"]
gamma = np.array([data["gamma"]]).astype(np.float16)
load_weights = True if data["load_weights"] == 1 else  False

net = pdqn(seq_len,conv,fclyr,conv_k_size,conv_stride,LOGDIR,gamma=gamma,batch_size=batch_size,learning_rate=learning_rate)

if (load_weights):
    T1 = np.zeros(shape=(1,84,84,seq_len))
    infer_dummy = [T1]
    train_dummy = [np.vstack([T1,T1]),
                    np.asarray([[1],[0]]),
                    np.asarray([.5,-1.0]).reshape((2,1)),
                    np.vstack([T1,T1])]
    net.infer(infer_dummy)
    net.train(inputs=train_dummy,IS_weights=np.ones(shape=(2,1)),r=[0,0])
    net.set_model_weights(r"C:\\Users\\vishnu\\Documents\\EngProj\\SSPlayer\\sweights\\b5weights2.hdf5")
game = snake(data["pc"])


run_type = input("Run type?(r=run, t=testing,p=play): ")
if (run_type == "p"):
    num_times = int(input("Play_times?: "))
    game_trainer = Trainer(1)
    game_trainer.play(net,game,seq_len,num_times,TSNE=False,TSNE_size=5000)
elif (run_type == "r"):
    num_times = 5000000
    greed_frames = 1000000
    max_exp_len = 1000000
    min_exp_len_train = 25000 #30000
    n = 500000
    game_trainer = Trainer(1)
    game_trainer.play_train(net,game,learning_rate,seq_len,batch_size,num_times,greed_frames,max_exp_len,min_exp_len_train,1,n,15,LOGDIR)
elif (run_type == "t"):
    num_times = 20000
    greed_frames = 1000
    max_exp_len = 100
    min_exp_len_train = 10
    n = 1000
    game_trainer = Trainer(1)
    game_trainer.play_train(net,game,learning_rate,seq_len,batch_size,num_times,greed_frames,max_exp_len,min_exp_len_train,1,n,15,LOGDIR)
else :
    print("INVALID PLAY OPTION")

game.kill()
exit()
