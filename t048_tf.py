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
LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log2"

summary_dir = LOGDIR
chkpt_dir = LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log2"

ops_and_tens = construct_two_network_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)
print(ops_and_tens.keys())
writer = ops_and_tens['writer']
summ = ops_and_tens['summ']

saver = tf.train.Saver()


summary_hook = tf.train.SummarySaverHook(   save_steps=1,save_secs=None,
                                                output_dir=summary_dir,summary_writer=None,
                                                scaffold=None,summary_op=summ)

saver_hook = tf.train.CheckpointSaverHook(  checkpoint_dir=chkpt_dir,
                                                save_secs=None,save_steps=5,
                                                saver=saver,checkpoint_basename='model.ckpt',
                                                scaffold=None)

chief_session = tf.train.ChiefSessionCreator(scaffold=None,config=config, checkpoint_dir=chkpt_dir)


#Launch game
g_sheets = 0
game = t048(2)
wait_for(1)
with tf.train.MonitoredSession(session_creator=chief_session,hooks=[saver_hook, summary_hook]) as sess:
    train_or_play = input("T for train,P for play,E for end: T/P/E: ")
    num_times = int(input("Number frames to Process?: "))
    greed_frames = int(input("Greed Frames Limit: "))

    game_trainer = Trainer(sess,game,num_times,greed_frames,10,batch_size,ops_and_tens,g_sheets)

    while (train_or_play is not "E"):
        if (train_or_play == "T" or train_or_play == "t"):
            game_trainer.play_train(100,1)
        elif train_or_play is "P" or train_or_play is "p":
            game_trainer.play(1)


        #See if train again    
        train_or_play = input("T for train,P for play,E for end: T/P/E: ")
        if (train_or_play != "E"):
            frames_or_iter = input("Frames or Iter: F/I: ")
            num_times = int(input("Number of times? : "))
            greed = float(input("Greed: "))
            greed_frames = int(input("Greed Frames Limit: "))
    
# ## Get working directory
# PATH = os.getcwd()

# ## Path to save the embedding and checkpoints generated
# LOGDIR = PATH + '/project-tensorboard/log-1/'
# tf.reset_default_graph()
# vec = []
# for i in game_trainer.em_vec:
#     print("img: ",np.shape(i))
#     i = np.expand_dims(i,0)
#     vec.append(i)

# vec = np.concatenate(vec)
# images = tf.Variable(vec)
# metadata = os.path.join(LOGDIR, 'labels.tsv')
# with tf.Session() as sess:
#     saver = tf.train.Saver([images])

#     sess.run(images.initializer)
#     saver.save(sess, os.path.join(LOGDIR, 'images.ckpt'))

#     config = projector.ProjectorConfig()
#     # One can add multiple embeddings.
#     embedding = config.embeddings.add()
#     embedding.tensor_name = images.name
#     # Link this tensor to its metadata file (e.g. labels).
#     embedding.metadata_path = metadata
#     # Saves a config file that TensorBoard will read during startup.
#     projector.visualize_embeddings(tf.summary.FileWriter(LOGDIR), config) 