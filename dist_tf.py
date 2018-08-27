import tensorflow as tf
import sys
import numpy as np
from dist_cnn import *
from snake_trainer import *
#from gsheets import *


s_name = str(sys.argv[1])
t_num = int(sys.argv[2])

cl_spec = tf.train.ClusterSpec({
    "worker": [
        "localhost:2223",
        "localhost:2224"
    ],
    "ps": [
        "localhost:2222"
    ]
})


x = np.random.rand(1,110,84,4).astype(np.uint8)
i = np.random.rand(10,110,84,4).astype(np.uint8)
ia = np.random.rand(10,1).astype(np.uint8)
ir = np.random.rand(10,1).astype(np.float16)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

conv_k_size = [20,10,4]
conv_stride = [2,2,1]
conv = [0,16,32,64]
fclyr = [0,125,45] #5
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 0.00025
gamma = np.array([.9]).astype(np.float16)
batch_size = 32
LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log"    

if s_name == "ps":
    server = tf.train.Server(cl_spec,job_name="ps",task_index=0,config=config)
    server.join()
else:
    server = tf.train.Server(cl_spec,job_name="worker",task_index=t_num,config=config)
    with tf.device(tf.train.replica_device_setter(cluster=cl_spec)):
        x1,a,q_vals_pr = infer_model(learning_rate,batch_size,
                    conv_count,fc_count,
                    conv,fclyr,
                    conv_k_size,conv_stride,LOGDIR)
        
        writer,summ,train,enqueue_op,p_queues,p_delta,s_img1,s_a,s_r,s_img2,uwb,p_r,gamma = train_model(learning_rate,
                                                     batch_size,conv_count,
                                                     fc_count,conv,
                                                     fclyr,conv_k_size,
                                                     conv_stride,LOGDIR)
        
        saver = tf.train.Saver()
    ops = {
        'action': a,'enqueue_op': enqueue_op,
        'train': train,'uwb': uwb,
        'p_queues' : p_queues, 'q_vals_pr': q_vals_pr,
        'p_r': p_r, 'gamma': gamma,
        'p_delta': p_delta,
    }
    
    phs = {
        'x1': x1, 's_img1': s_img1,
        's_a': s_a,'s_r': s_r,
        's_img2': s_img2
    }

    if (t_num == 0):
        g_sheets = 0
        game = snake(2)
        wait_for(1)
        print(server.target)
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(t_num == 1),
                                               config=config) as sess:
            train_or_play = input("T for train,P for play,E for end: T/P/E: ")
            frames_or_iter = input("Frames or Iter: F/I: ")
            num_times = int(input("Number of times? : "))
            greed = float(input("Greed: "))
            greed_frames = int(input("Greed Frames Limit: "))
            while (train_or_play is not "E"):
                if (train_or_play == "T" or train_or_play == "t"):
                    if (frames_or_iter is "I"):
                        dist_run(sess,game,greed,num_times,batch_size,ops,phs)
                    elif frames_or_iter is "F":
                        frame_train_reward(sess,game,num_times,greed_frames,batch_size,ops,phs,g_sheets)
                elif train_or_play is "P" or train_or_play is "p":
                    dist_play(sess,game,num_times,ops,phs)
                train_or_play = input("T for train,P for play,E for end: T/P/E: ")
                frames_or_iter = input("Frames or Iter: F/I: ")
                num_times = int(input("Number of times? : "))
                greed = float(input("Greed: "))
                greed_frames = int(input("Greed Frames Limit: "))

    else:
        #3600 saver
        #summ  = 300
        lap_dir = r'C:\Users\Vishnu\Documents\EngProj\SSPlayer\log'
        dsk_chk_dir = r"E:\TFtmp\test\model"
        dsk_sum_dir = r"E:\TFtmp\test\sum"

        saver_hook = tf.train.CheckpointSaverHook(  checkpoint_dir=lap_dir,
                                                    save_secs=3600,save_steps=None,
                                                    saver=tf.train.Saver(),checkpoint_basename='model.ckpt',
                                                    scaffold=None)
        summary_hook = tf.train.SummarySaverHook(   save_steps=1,save_secs=None,
                                                    output_dir=lap_dir,summary_writer=None,
                                                    scaffold=None,summary_op=summ)
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(t_num == 1),
                                                hooks = [saver_hook,summary_hook],
                                                save_summaries_steps=1,config=config) as sess:
            while not sess.should_stop():
                tt = sess.run([train,p_queues,p_delta],{x1: np.random.rand(1,110,142,4).astype(np.uint8)})
                #print(tt[2])

    
