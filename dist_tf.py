import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sys
import numpy as np
from dist_cnn import *
from t048_trainer import *
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

#[8,4]
#lr = .01
#16,8
conv_k_size = [8,4]
conv_stride = [4,2]
conv = [0,16,32]
fclyr = [0,256,128,64] #5
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 0.00025
gamma = np.array([.9]).astype(np.float16)
batch_size = 5
#LOGDIR = r"c:\Users\devar\Documents\EngProj\SSPlayer\log"    
LOGDIR = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log"

if s_name == "ps":
    server = tf.train.Server(cl_spec,job_name="ps",task_index=0,config=config)
    server.join()
else:
    server = tf.train.Server(cl_spec,job_name="worker",task_index=t_num,config=config)
    #dev_rep_setter = tf.train.replica_device_setter( worker_device="/job:worker/task:%d" % t_num,cluster=cl_spec)
    #dev_rep_setter = tf.train.replica_device_setter(cluster=cl_spec)
    #print("#################",dev_rep_setter)
    #with tf.device(dev_rep_setter):
    x1,a,q_vals_pr = infer_model(learning_rate,batch_size,
                conv_count,fc_count,
                conv,fclyr,
                conv_k_size,conv_stride,LOGDIR)

    writer,summ,train,enqueue_op,p_delta,s_img1,s_a,s_r,s_img2,infer_ops,target_ops,p_r,gamma,global_step = train_model(learning_rate,
                                                    batch_size,conv_count,
                                                    fc_count,conv,
                                                    fclyr,conv_k_size,
                                                    conv_stride,LOGDIR)

        #saver = tf.train.Saver()
    

    ops = {
        'action': a,'enqueue_op': enqueue_op,
        'train': train,'infer_ops': infer_ops, 'q_vals_pr': q_vals_pr,
        'p_r': p_r, 'gamma': gamma,
        'p_delta': p_delta, 'target_ops': target_ops,
        'global_step': global_step
    }
    
    phs = {
        'x1': x1, 's_img1': s_img1,
        's_a': s_a,'s_r': s_r,
        's_img2': s_img2
    }

    

    summary_dir = r'C:\Users\Vishnu\Documents\EngProj\SSPlayer\log\sum'
    ckpt_dir = r'C:\Users\Vishnu\Documents\EngProj\SSPlayer\log\ckpt'

    lap_dir = r'C:\Users\Vishnu\Documents\EngProj\SSPlayer\log'
    dsk_chk_dir = r"E:\TFtmp\test\model"
    dsk_sum_dir = r"E:\TFtmp\test\sum"

    #Saver
    #saver = tf.train.Saver(
    #    var_list = get_all_weights_bias_tensors(conv_count,fc_count)
    #)
    saver = tf.train.Saver()

    summary_hook = tf.train.SummarySaverHook(   save_steps=1,save_secs=None,
                                                    output_dir=summary_dir,summary_writer=None,
                                                    scaffold=None,summary_op=summ)

    saver_hook = tf.train.CheckpointSaverHook(  checkpoint_dir=ckpt_dir,
                                                    save_secs=3600,save_steps=None,
                                                    saver=saver,checkpoint_basename='model.ckpt',
                                                    scaffold=None)

    if (t_num == 0):
        # THis is master thread

        g_sheets = 0
        game = t048(2)
        wait_for(1)
        print(server.target)

        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=True, 
                                                hooks=[saver_hook,summary_hook], checkpoint_dir=ckpt_dir,
                                                config=config,save_summaries_steps=1,
                                                save_checkpoint_steps=1) as sess:   
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
                        #frame_train_reward_1(sess,game,num_times,greed_frames,batch_size,ops,phs)
                elif train_or_play is "P" or train_or_play is "p":
                    dist_play(sess,game,num_times,ops,phs)
                train_or_play = input("T for train,P for play,E for end: T/P/E: ")
                if (train_or_play != "E"):
                    frames_or_iter = input("Frames or Iter: F/I: ")
                    num_times = int(input("Number of times? : "))
                    greed = float(input("Greed: "))
                    greed_frames = int(input("Greed Frames Limit: "))

    else:
        
        #3600 saver
        #summ  = 300
        print(server.target)
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=False,
                                                save_summaries_steps=1,config=config,
                                                save_checkpoint_steps=1) as sess:
            #writer.add_graph(sess.graph)
            while not sess.should_stop():
                #print("Active")
                
                """
                tt = sess.run([train,p_delta,global_step],{s_img1: np.random.rand(5,100,100,4),
                                                            s_a: np.random.rand(5,1).astype(np.float16), 
                                                            s_r: np.random.rand(5,1).astype(np.float16),
                                                            s_img2: np.random.rand(5,100,100,4),
                                                            x1: np.random.rand(1,100,100,4).astype(np.uint8)})
                """
                tt = sess.run([train,p_delta,global_step],{x1: np.random.rand(1,100,100,4).astype(np.uint8)})
                
                
                if tt[2] % 10 == 0:
                    print("global_step",tt[2])
                    sess.run([infer_ops,target_ops],{x1: np.random.rand(1,100,100,4).astype(np.uint8)})

    
