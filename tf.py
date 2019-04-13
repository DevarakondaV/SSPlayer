import tensorflow as tf
import sys
import numpy as np
from cnn import *
#from gsheets import *
import tensorflow.contrib.graph_editor as ge
from snake_trainer import *
from trainer import *

import json

with open("meta.json","r") as params_file:
    data = json.load(params_file)
    print(data)

LOGDIR = data["logdir"] if data["pc"] == 1 else data["plogdir"]
save_steps = data["save_steps"]

#network params
batch_size = data["batchsize"]
seq_len = data["seq_len"]
conv_k_size = [i["l"+str(idx+1)] for i,idx in zip(data["conv_k_size"],range(0,len(data["conv_k_size"])))]
conv_stride = [i["l"+str(idx+1)] for i,idx in zip(data["conv_stride"],range(0,len(data["conv_stride"])))]
conv = [data["conv"][0]["in"]] + [data["conv"][i]["l"+str(i)] for i in range(1,len(data["conv"]))]
fclyr = [i["l"+str(idx+1)] for i,idx in zip(data["fclyr"],range(0,len(data["fclyr"])))]
learning_rate = data["learing_rate"]
gamma = np.array([data["gamma"]]).astype(np.float16)


#Session configuration parameters
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

conv_count = len(conv)
fc_count = len(fclyr)

summary_dir = LOGDIR+r"\log"
chkpt_dir = LOGDIR+"\ckp" 

ops_and_tens = construct_two_network_model(learning_rate,gamma,batch_size,seq_len,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)
print(ops_and_tens.keys())
writer = ops_and_tens['writer']
summ = ops_and_tens['summ']
saver = tf.train.Saver()

scaffold = tf.train.Scaffold(summary_op=summ,saver=saver,ready_for_local_init_op=None)

chief_session = tf.train.ChiefSessionCreator(scaffold=scaffold,config=config, checkpoint_dir=chkpt_dir)

#Hooks for session
summary_hook = tf.train.SummarySaverHook(   save_steps=save_steps,save_secs=None,
                                                output_dir=summary_dir,summary_writer=None,
                                                scaffold=None,summary_op=summ)

saver_hook = tf.train.CheckpointSaverHook(  checkpoint_dir=chkpt_dir,
                                                save_secs=None,save_steps=save_steps,
                                                saver=saver,checkpoint_basename='model.ckpt',
                                                scaffold=None)

#Session param



g_sheets = 0

game = snake(data["pc"])



wait_for(1)
with tf.train.MonitoredSession(session_creator=chief_session,hooks=[saver_hook, summary_hook]) as sess:
    
    if data["pc"] == 1 or data["pc"] == 2:
        test_run = input("Test run or pure trainig:(t/p)")
        if (test_run == "p"):
            num_times = 5000000
            greed_frames = 1000000
            max_exp_len = 100000
            min_exp_len_train = 25000 #30000
            n = 1000000
        else :
            num_times = 1000
            greed_frames = 100
            max_exp_len = 100
            min_exp_len_train = 10
            n = 100
        run_play = input("Run Training or play:(r/p) ")
        game_trainer = Trainer(sess,game,num_times,greed_frames,max_exp_len,min_exp_len_train,seq_len,batch_size,ops_and_tens,g_sheets,1)
        if (run_play == "r"):
            game_trainer.play_train(n,15)
            
        else:
            game_trainer.play(15)
    else:
        Testing_Desktop = input("Testing Desktop?(1-yes):  ")

        if (Testing_Desktop == "1"):
            num_times = 100
            greed_frames = 10
            max_exp_len = 20
            min_exp_len_train = 10

            game_trainer = Trainer(sess,game,num_times,greed_frames,max_exp_len,min_exp_len_train,10,batch_size,ops_and_tens,g_sheets,1)
            game_trainer.play_train(10,5)


        elif Testing_Desktop == "0":
            train_or_play = input("T for train,P for play,E for end: T/P/E: ")
            num_times = int(input("Number frames to Process?: "))
            greed_frames = int(input("Greed Frames Limit: "))
            max_exp_len = 10
            min_exp_len_train = 10
            game_trainer = Trainer(sess,game,num_times,greed_frames,max_exp_len,min_exp_len_train,10,batch_size,ops_and_tens,g_sheets,1)

            while (train_or_play is not "E"):
                if (train_or_play == "T" or train_or_play == "t"):
                    game_trainer.play_train(10,1)
                elif train_or_play is "P" or train_or_play is "p":
                    game_trainer.play(3)


                #See if train again    
                train_or_play = input("T for train,P for play,E for end: T/P/E: ")
                if (train_or_play != "E"):
                    num_times = int(input("Number of times? : "))
                    greed_frames = int(input("Greed Frames Limit: "))
        elif Testing_Desktop == "p":
            num_times = int(input("How many times?: "))
            greed_frames = 10
            max_exp_len = 20
            min_exp_len_train = 10
            game_trainer = Trainer(sess,game,num_times,greed_frames,max_exp_len,min_exp_len_train,10,batch_size,ops_and_tens,g_sheets,1)
            game_trainer.play(num_times)

        else:
            exit()
