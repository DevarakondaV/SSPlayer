import time
import numpy as np
#np.set_printoptions(threshold=np.nan)
from PIL import Image
from snake_con import *
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
#import cv2
from collections import deque
#import objgraph

from pynput import keyboard
from threading import Thread
from exp import experience
import gc

class Trainer:
    """
    Class is responsible for training
    """

    #Main Methods
    def __init__(self,log):
        """
        Constructor
        args:
            sess:               Tensorflow session object
            game:               Game object
            frame_limit:        int. Number of frames to play
            greed_frames:       int. Number of frames when greed is active
            max_exp_len:        int. Maximum lenght of experience vector
            min_exp_len_train:  int. Minimum lenght of exp before training
            seq_len:            int. Sequence size which determines state
            batch_size:         int. Determines batchsize of training operation
            ops_and_tens:       Dictionary. Contains the relevent tensorflow operations and tensors
            gsheets:            gsheets object. Used to post results to google sheets.
            log:                Bool. True to activate console log.
        returns:
            Null
        """

        self.force_kill = False                     #Bool param determines if training should be forced to stop
        
        #variables required for training
        #self.exp = deque(maxlen = max_exp_len)      #experience vector  
        self.exp = experience(100000) #100000
        self.total_frames = 0                       #Total frames runduring entire training session
        self.process_frames = 0                     #Number of frames to process

        #Params used while trianing
        self.game_play_iteration = 0                #Number of iterations of game play
        self.num_train_ops = 0                      #Number of trianing operations
        self.log = log
        self.pause = False




    def store_exp(self,error,seq):
        """
            Function stores new exp into experience buffer
            args:
                seq: tuple. new experience to be added to buffer
            returns:
        """

        self.exp.store(error,seq)
        self.process_frames = self.process_frames +  1 
        self.total_frames = self.total_frames + 1
        return

    def get_greed(self,processed_frames,greed_frames):
        """
            Function returns greed based on linear relationship
            
            return:
                float. Greed value that linearly ranges from 1->.1
        """

        
        if processed_frames > greed_frames:
            return 0.05
        return (((.1-1)/greed_frames)*processed_frames)+1

    def update_exp(self,leaf_idx,td):
        """
            Function updates the experience tree priorities

        """
        self.exp.update(leaf_idx,td)


    def process_seq(self,seq,seq_len):
        """
            Function process the last seq frames and stacks them for the network

            args:
                seq: List. Containg SARSA for game iteration
                seq_len: int. Sequence length
            returns:
                np_f: numpy array: Stacked Last four frames of the sequence
        """

        if (len(seq) < seq_len):
            frames = [np.zeros(shape=[84,84,1]).astype(np.uint8) for i in range(0,seq_len-len(seq))]+ [i for i in seq]
        else:
            frames = seq
        
        return np.expand_dims(np.stack(frames,axis=2).squeeze(),axis=0)
        
        # np_f = frames[0]
        # for i in range(1,len(frames)):
        #     np_f = np.concatenate((np_f,frames[i]),axis=2)
        # return np_f
    
            
    def create_reward_plot(self,r_iter,save_dir):
        """
        Funciton creates a plot of the reward vs game iteration.

        args:
            g_iter: int. NNumber of total games played
            r_iter: List of ints. List containing reward for each game iterstion
            save_dir:   String. Directory where plot is saved
        """

        len_re = len(r_iter)
        if len_re == 0 or len_re == 1:
            return

        #Games vec
        g_vec = np.arange(0,len(r_iter))

        plt.plot(g_vec,r_iter)
        plt.title("Reward for game iteration")
        plt.savefig(save_dir+"\\rewardplt.png")


    def key_manager(self,key):
        """
        Function manages pausing and killing game play
        """

        if key == keyboard.Key.esc:
            self.pause = not self.pause
        elif key == keyboard.Key.f1:
            self.pause = False
            self.force_kill = True

    def esc_wrap(self,function):
        """
            Function wraps function in escape key to end training

            args:
                function: Function to wrap with keyboard listener

            returns:
                rtn_val: Return value of function
        """
        
        #Wrapping trainer inside a listener to stop training if esc key is pressed
        with keyboard.Listener(on_press=self.key_manager) as listener:
            #perform the Q algorithm with experience replay
            rtn_val = function()
            listener.stop()


        return rtn_val

    def play_train(self, net, game, learning_rate,
                    seq_len, batch_size, num_times, greed_frames,
                    min_exp_len, max_exp_len_train, log, n, x,LOGDIR):
        """
        args:
            net:                Net object.
            game:               Game object
            learning_rate:      Float
            seq_len:            int. Sequence size which determines state
            batch_size:         int. Determines batchsize of training operation
            num_times:          int. Total number of transitions to process
            greed_frames:       int. Number of frames when greed is active
            max_exp_len:        int. Maximum lenght of experience vector
            min_exp_len_train:  int. Minimum lenght of exp before training
            log:                Bool. True to activate console log.
            n: int. Number of process frames to run before running play operations
            x: Number of times to play the game
            LOGDIR:             String. Directory where model is saved
        """

        weights_save_dir = LOGDIR+"\\weights\\"
        if not os.path.exists(weights_save_dir):
            os.makedirs(weights_save_dir)


        run_times = int(num_times/n)
        reward_list = []

        for i in range(0,run_times):
            print("Iteration {} of {}".format(i+1,run_times))
            if not self.force_kill:
                self.train(net, game, n, seq_len, batch_size, greed_frames, max_exp_len_train,learning_rate,i)
                net.save_weights(weights_save_dir+"weights{}.h5".format(i),save_format='h5')
                r = self.play(net, game, seq_len, x)
                
                reward_list.append(r)
        
        self.con_log("Reward List: ",reward_list)
        self.create_reward_plot(reward_list,LOGDIR)

    def train(self, net, game, n, seq_len, 
                batch_size, greed_frames, min_exp_len_train,
                learning_rate,i):
        """
        Function trains network

        args:
            net: Network
            game: Game
            n: int. Number of frames to train
            greed_frames: int. greed
            min_exp_len_train: int. Determines if train.
            i: int. I'th training Iteration
        returns:
            null
        """




        dummy_in = [np.zeros(shape=(10,84,84,5)).astype(np.float16),
                    np.zeros(shape=[1]).astype(np.float16),
                    np.zeros(shape=(10,1)).astype(np.float16),
                    np.zeros(shape=(10,84,84,5)).astype(np.float16)]

        current_tran = 1
        def fun():
            nonlocal current_tran
            while current_tran < n:
                

                seq = []
                game.click_play()
                frame = game.get_frame()
                seq.append(frame)
                phi1 = self.process_seq(seq, seq_len)
                while not game.stop_play:
                    os.system('cls')
                    print("TRAINSITION",current_tran+(i*n))
                    
                    current_tran+=1
                    greed = self.get_greed(current_tran+(i*n),greed_frames)
                    print("GREED",greed)
                    if  greed > np.random.rand(1)[0]:
                        a = np.random.randint(0,2,size=[1])[0]
                        e = 1
                        kill_play = game.perform_action(a)
                    else:
                        e,a = net.infer([phi1])
                        kill_play = game.perform_action(a.numpy()[0])

                    
                    r = game.get_reward()
                    #print("REWARD FOR ACTION", r)
                    frame = game.get_frame()

                    if (len(seq) >= seq_len):
                        seq.pop(0)
                    seq.append(frame)
                    
                    phi2 = self.process_seq(seq,seq_len)
                    self.store_exp(np.max(e.numpy()[0]) if type(e) is not int else e,
                                    (phi1, np.array(a).astype(np.uint8),
                                    np.array(r).astype(np.float16),
                                    phi2))
                    
                    phi1 = phi2
                    
                    if (len(self.exp) > batch_size and len(self.exp) > 10000):
                        leaf_idx,IS_weights,seq_n = self.exp.sample(batch_size)
                        IS_weights = np.reshape(IS_weights,(batch_size,1))
                        y,Tra_d3 = net.train(seq_n,IS_weights,r)
                        self.update_exp(leaf_idx,np.amax((y-Tra_d3).numpy(),axis=1))

                        if current_tran % 10 == 0:
                            net.update_target_weights()
                    
                    if (kill_play):
                        print("END Play")

                    if self.force_kill:
                        break

                    while self.pause:
                        pass
                
                if self.force_kill:
                    break

        return self.esc_wrap(fun)


    def play(self,net,game,seq_len,num_times,TSNE=False,TSNE_size=0):
        """
        Fucntion plays the game

        args:
            net: pdqn instance
            game: game instance
            seq_len: int. Sequence length
            num_times: int. number of times to play the game
            TSNE: Bool. If True, function will write files for performing tsne
            TSNE_size: int. Number of transitions to save for tsne

        returns:
            rtn_val: double. Average reward for n iterations
        """

        #net([np.zeros(shape=(1,84,84,5)).astype(np.float16)])
        #net.infer([np.zeros(shape=(1,84,84,5)).astype(np.float16)])

        #Create memap objects if TSNE
        if (TSNE):
            tarmmep = np.memmap("tsne/tar",dtype=np.float16,mode='w+',shape=(TSNE_size,512))
            vmmep = np.memmap("tsne/v",dtype=np.float16,mode='w+',shape=(TSNE_size,))
            ammep = np.memmap("tsne/a",dtype=np.float16,mode='w+',shape=(TSNE_size,))
            
            t_i = 0

        def fun():
            if (TSNE):
                nonlocal t_i,tarmmep,vmmep,ammep
            reward_list = []
            for i in range(0,num_times):
                self.con_log('Play Iteration: ',i+1)
                if (i > 5000):
                    self.con_log('Play Iteration To High: ',i+1)                    
                    break
                game.click_play()
                reward = 0
                seq = []
                frame = game.get_frame()
                seq.append(frame)
                phi1 = self.process_seq(seq,seq_len)
                while not game.stop_play:
                    os.system('cls')
                    phi1 = phi1.astype(np.float16)
                    if not TSNE:
                        Tar_d3,a = net.infer([phi1])
                    else:
                        Tar_d3,a,Tar_d2 = net.infer([phi1],TSNE=TSNE)
                        print("TSNE: ",t_i)
                        tarmmep[t_i:,:] = Tar_d2.numpy().astype(np.float16)
                        vmmep[t_i:] = np.amax(Tar_d3.numpy()).astype(np.float16)
                        ammep[t_i:] = a.numpy()[0].astype(np.uint8)
                        t_i+=1
                        if (t_i % 100 == 0):
                            self.save_transition(phi1,name=str(t_i))
                        if (t_i == TSNE_size):
                            del tarmmep
                            del ammep
                            self.force_kill = True
                    kill_play = game.perform_action(a.numpy()[0])
                    if (kill_play):
                        print("END PLAY")
                        break
                    r = game.get_reward()
                    frame = game.get_frame()
                    reward += r

                    if (len(seq) >= seq_len):
                        seq.pop(0)
                    seq.append(frame)
                    phi2 = self.process_seq(seq,seq_len)
                    phi1 = phi2

                    if self.force_kill:
                        break

                    while self.pause:
                        pass
                
                if self.force_kill:
                    break
            
                reward_list.append(reward)
            return np.mean(reward_list)

        return self.esc_wrap(fun)

    def con_log(self,msg_n,msg = ""):
        """
        Functions console logs messages based on log_or_no

        args:
            msg_n: String. Title for the message
            msg:    Object to log
        returns:
            null
        """

        if self.log:
            print(msg_n,msg)
        return

    #METHODS FOR DEBUGGING AND PROGRESS
    def print_progress(self,greed):
        """
        Function prints, relevent progress metrics

        args:
            gp: Int. Game Play iteration number
            greed: Float. Greed value
        """

        if (self.game_play_iteration % 50 == 0):
            self.con_log("Exp size: ", len(self.exp))
            self.con_log("Number process Frames: ", self.process_frames)
            self.con_log("greed: ",greed)


    def save_seq_img(self,seq):
        for i in range(0,len(seq)):
            Image.fromarray(np.squeeze(seq[i])).save("imgs/test"+str(i)+".png")

    
    def save_transition(self,transition,name):
        """
        Function writes transition as image.
        If transition channel is greater than 3 it will drop the remaining
        args:
            transition: Numpy array of shape [1,img_w,img_h,channels]
            name: the name to write the file as
        """
        write_loc = "tsne/imgs/"+name
        r_img = transition.squeeze()[:,:,4].astype(np.uint8)
        img = Image.fromarray(r_img)
        path = write_loc+".jpeg"
        img.save(path, "JPEG")


