import tensorflow as tf
import sys
import numpy as np
from cnn import *
from t048_trainer import *
#from gsheets import *



x = np.random.rand(1,100,100,4)
i = np.random.rand(10,100,100,4).astype(np.uint8)
ia = np.random.rand(10,1).astype(np.uint8)
ir = np.random.rand(10,1).astype(np.float16)

x = (x*250).astype(np.uint8)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

conv_k_size = [8,4,3]
conv_stride = [4,2,1]
conv = [0,32,64,64]
fclyr = [0,512] #5
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 0.00025
gamma = np.array([.9]).astype(np.float16)
batch_size = 10
LOGDIR = r"c:\Users\devar\Documents\EngProj\SSPlayer\log3"

summary_dir = LOGDIR
chkpt_dir = LOGDIR = r"c:\Users\devar\Documents\EngProj\SSPlayer\log3"

ops_and_tens = construct_two_network_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)
print(ops_and_tens.keys())
writer = ops_and_tens['writer']
summ = ops_and_tens['summ']

saver = tf.train.Saver(save_relative_paths=True)


summary_hook = tf.train.SummarySaverHook(   save_steps=1,save_secs=None,
                                                output_dir=summary_dir,summary_writer=None,
                                                scaffold=None,summary_op=summ)

saver_hook = tf.train.CheckpointSaverHook(  checkpoint_dir=chkpt_dir,
                                                save_secs=None,save_steps=500,
                                                saver=saver,checkpoint_basename='model.ckpt',
                                                scaffold=None)

chief_session = tf.train.ChiefSessionCreator(scaffold=None,config=config, checkpoint_dir=chkpt_dir)


#Launch game
g_sheets = 0
game = t048(1)
wait_for(1)
with tf.train.MonitoredSession(session_creator=chief_session,hooks=[saver_hook, summary_hook]) as sess:
    train_or_play = "E"
    num_times = 100000
    greed_frames = 10000

    game_trainer = Trainer(sess,game,num_times,greed_frames,10,batch_size,ops_and_tens,g_sheets,1)
    game_trainer.play_train(1000,25)
    