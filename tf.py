import tensorflow as tf
import sys
import numpy as np
from cnn import *
#from gsheets import *
tf.enable_eager_execution()
tf.executing_eagerly() 
from snake_trainer import *
from trainer import *
import json
import os

os.chdir(r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\\")

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


net = pdqn(seq_len,conv,fclyr,conv_k_size,conv_stride,LOGDIR,gamma=gamma,batch_size=batch_size,learning_rate=learning_rate)
game = snake(data["pc"])


run_type = input("Run type?(r=run, t=testing,p=play): ")
if (run_type == "p"):
    num_times = int(input("Play_times?: "))
    game_trainer = Trainer(1)
    game_trainer.play(net,game,seq_len,num_times)
elif (run_type == "r"):
    num_times = 1000000
    greed_frames = 100000
    max_exp_len = 100000
    min_exp_len_train = 25000 #30000
    n = 1000
    # game_trainer = Trainer(net,game,num_times,greed_frames,max_exp_len,min_exp_len_train,10,batch_size,ops_and_tens,g_sheets,1)
    game_trainer = Trainer(1)
    game_trainer.play_train(net,game,learning_rate,seq_len,batch_size,num_times,greed_frames,max_exp_len,min_exp_len_train,1,n,15)
elif (run_type == "t"):
    num_times = 1000
    greed_frames = 100
    max_exp_len = 100
    min_exp_len_train = 10
    n = 100
    game_trainer = Trainer(1)
    game_trainer.play_train(net,game,learning_rate,seq_len,batch_size,num_times,greed_frames,max_exp_len,min_exp_len_train,1,n,15)
else :
    print("INVALID PLAY OPTION")

game.kill()
exit()