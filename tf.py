import tensorflow as tf
import sys
import numpy as np
from cnn import *
#from gsheets import *
import tensorflow.contrib.graph_editor as ge
from snake_trainer import *
from trainer import *


pc = 1  #1 for desktop, 2 for laptop
if pc == 1:
    LOGDIR = r"E:\vishnu\SSPlayer\one"
    save_steps = 5000
else:
    LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log2"
    save_steps = 1
#Session configuration parameters
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# game = snake(pc)
# take_shot(game)
# game.click_play()
# take_shot(game)
# exit()

#Network params
batch_size = 10
seq_len = 4

conv_k_size = [8,4,3]
conv_stride = [4,2,1]



conv = [seq_len,32,64,64]
fclyr = [0,512] #5
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 0.00025
gamma = np.array([.9]).astype(np.float16)



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


game = snake(pc)
wait_for(1)
with tf.train.MonitoredSession(session_creator=chief_session,hooks=[saver_hook, summary_hook]) as sess:
    
    if pc == 1:
        run_play = input("Run Training or play:(r/p) ")
        num_times = 1000000
        greed_frames = 100000
        max_exp_len = 100000
        min_exp_len_train = 25000 #30000
        game_trainer = Trainer(sess,game,num_times,greed_frames,max_exp_len,min_exp_len_train,seq_len,batch_size,ops_and_tens,g_sheets,1)
        if (run_play == "r"):
            #game_trainer.play_train(100,1)
            game_trainer.play_train(10000,15)
            
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