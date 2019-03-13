import time
from timeit import timeit,Timer
import numpy as np
np.set_printoptions(threshold=np.nan)
from PIL import Image
from snake_con import *
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
#import cv2
from collections import deque

from pynput import keyboard
from threading import Thread

from exp import experience

class Trainer:
    """
    Class is responsible for training
    """

    #Main Methods
    def __init__(self,sess,game,frame_limit,greed_frames,max_exp_len,min_exp_len_train,seq_len,batch_size,ops_and_tens,gsheets,log = 0):
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

        #Declaring some variables
        
        #Tensorflow variables
        self.sess = sess                            #Tensorflow session
        self.ops_and_tens = ops_and_tens            #Tensorflow operations and tensors

        #Game controller variables
        self.game = game                            #Game object
        self.force_kill = False                     #Bool param determines if training should be forced to stop
        

        #variables required for training
        #self.exp = deque(maxlen = max_exp_len)      #experience vector  
        self.exp = experience(100000) #100000
        self.total_frames = 0                       #Total frames runduring entire training session
        self.process_frames = 0                     #Number of frames to process
        self.max_exp_len = max_exp_len              #The maximum length of the experience vector before removing experience      
        self.min_exp_len_train = min_exp_len_train  #Minimum lenght of exp required for training operations    
        self.frame_limit = frame_limit              #Maximum number of frame sto play  
        self.greed_frames = greed_frames            #Maximum number of frames when greed is calculated
        
       
        self.batch_size = batch_size                #Determines batchsize of training operation
        self.seq_len = seq_len                      #The number of frames which determines a state

        #Params used while trianing
        self.game_play_iteration = 0                #Number of iterations of game play
        self.num_train_ops = 0                      #Number of trianing operations
        self.gsheets = gsheets                      #gsheets object ot post to sheets
        
        #self.tsv_file = r"c:\Users\Vishnu\Documents\EngProj\SSPlayer\log2\labels.tsv"
        self.em_vec = []
        self.log = log

        


    #Helper Methods
    def infer_action(self,frames):
        """
            Function infers action from the inference network
        
            args:
                frames: numpy array. Frames that are passed to the inference network
            return:
                a: int. action predicted by the inference network
        """

        #Grabbing relevent objects for inference
        s1 = self.ops_and_tens['s1']
        s2 = self.ops_and_tens['s2']
        r = self.ops_and_tens['r']
        weights = self.ops_and_tens['IS_weights']
        weights_tmp = np.zeros(shape=[self.batch_size,1])
        action = self.ops_and_tens['action']
        TD_error = self.ops_and_tens['TD_error']

        #em = self.sess.graph.get_tensor_by_name("Target/Dense_Layers/FC1/act1/Maximum:0")
        #Need dummy value for placeholders not in use
        zeros = np.zeros(shape=frames.shape).astype(np.uint8)
        rv = np.zeros((self.batch_size,1))
        #Inference
        a,error = self.sess.run([action,TD_error],{s1: [frames],s2: [zeros],r: rv,weights: weights_tmp})
        #self.write_label_to_tsv(a)
        #self.em_vec.append(em)
        return a,error[0]

    def perform_action(self,a):
        """
        Function performs an action in the game and returns
        args:
            a: int. Action to send to the controller.
        returns:
            m_dir: String. Defines direction in which movement happened
        """

        game = self.game
        #Telling game to perform an action based on the value of a
        # 0 left
        # 1 right
        # 2 do nothing
        move_dir = game.move_dir
        
        if a == 2:
            game.move(-1)
            m_dir = "stright"
            return m_dir
        
        if (move_dir == 0):
            key = game.left if a == 0 else game.right
            m_dir = "left" if a == 0 else "right"            
            game.move_dir = 2 if a == 0 else 3 
        elif (move_dir == 1):
            key = game.right if a == 0 else game.left
            m_dir = "right" if a == 0 else "left"
            game.move_dir = 3 if a == 0 else 2
        elif (move_dir == 2):
            key = game.down if a == 0 else game.up
            m_dir = "down" if a == 0 else "up"
            game.move_dir = 1 if a == 0 else 0 
        else:
            key = game.up if a == 0 else game.down 
            m_dir = "up" if a == 0 else "down" 
            game.move_dir =  0 if a == 0 else 1 
        game.move(key)

        return m_dir

    def get_reward_for_action(self):
        """
        Function determines the reward associated with the taken action by looking at game object

        """

        game = self.game
        
        #Grab play again div using xpath
        # play_again_elem = game.chrome.find_element_by_xpath("/html/body/div[2]/div[3]/div[1]/div/a[2]")
        
        # #If play again div is displayed stop play!
        # if (play_again_elem.is_displayed()):
        #     game.stop_play = True
        #     r = -1
        #     play_again_elem.click()
        # else:
        #     #r = game.get_reward2()
        #     r = game.get_reward3()
        #     if r == -1:
        #         game.stop_play = True
        
        return game.get_reward()


    def send_action_to_game_controller(self,phi1,a,reward):
        """
            Functions performs action in the game
            args:
                phi1: numpy array. First state in SARSA
                a: int. Action to send to the controller.
                reward: float. Reward for the current game iteration
            return:
                frame: numpy array: Second state in SARSA
                bval: Bool: States whether or not the game is currently being played
                r: int.Reward associated with the taken action.
                reward: float. Updated total reward for the current game iteration
        """ 
        global t1

        #Perform the action in the game
        m_dir = self.perform_action(a)
        #Get the reward for the action!
       
        chk_frm = phi1[:,:,self.seq_len-1]        
        # frame = self.get_frame()
        # while (np.array_equal(chk_frm,np.squeeze(frame))):
        #     print("IN WHILE")
        #     frame = self.get_frame()
        r = self.get_reward_for_action()
        #Determine the new state of the game
        frame = self.get_frame()
        
        
        #
        reward = 0 
        

        self.con_log("Action = {}\nMove dir = {}\nReward = {}".format(str(a),m_dir,r))
        self.con_log("Process Frames = {}\nTotal Frames {}".format(self.process_frames,self.total_frames))
        self.con_log("Avaiable Memory = {}".format(psutil.virtual_memory().available))
        #if frames are equal then invalid move..reinfer Process Frames
        #chk_frm = phi1[:,:,0]
        chk_frm = phi1[:,:,self.seq_len-1]
        return frame,r,reward,not np.array_equal(chk_frm,np.squeeze(frame))
        #return frame,r,reward,1



    def random_minibatch_sample(self,batchsize):
        """
            Function randomly samples batch from exp buffer
            args:
                batchsize: int. Size of the batch
            returns:
                tuple with batchsize number SARS pairs
        """
        
        exp = self.exp
        #Get random indexes
        line_N = np.random.randint(0,len(exp),size=batchsize)

        #Extract batch size number of elements from array
        img1s = np.array([exp[i][0] for i in line_N])
        a = np.array([exp[i][1] for i in line_N]).squeeze().reshape(batchsize,1)
        r = np.array([exp[i][2] for i in line_N]).squeeze().reshape(batchsize,1)
        img2s = np.array([exp[i][3] for i in line_N])
        return (img1s,a,r,img2s)

    def store_exp(self,error,seq):
        """
            Function stores new exp into experience buffer
            args:
                seq: tuple. new experience to be added to buffer
            returns:
        """

        self.exp.store(error,seq)

        return

        #Older experience is phased out by poping from exp buffer
        if (len(self.exp) > self.max_exp_len):
            self.exp.pop()
        #process_frames = process_frames+2

        #add new experience
        self.exp.appendleft(seq)    
        return

    def get_greed(self):
        """
            Function returns greed based on linear relationship
            
            return:
                float. Greed value that linearly ranges from 1->.1
        """

        
        frames = self.total_frames
        greed_frames = self.greed_frames
        if frames > greed_frames:
            return 0.1
        return (((.1-1)/greed_frames)*frames)+1


    def execute_train_operation(self,batch_size):
        """
            Function executes a training operation
            
            args:
                batch_size: Size of batch to train on
            return:
        
        """

        sess = self.sess
        ops_and_tens = self.ops_and_tens

        self.con_log("TRAINING OPERATION: {}".format(self.num_train_ops),"")
        train = ops_and_tens['train']
        loss = ops_and_tens['loss']
        s1 = ops_and_tens['s1']
        s2 = ops_and_tens['s2']
        r = ops_and_tens['r']
        prt1 = ops_and_tens['print1']
        prt2 = ops_and_tens['print2']
        TD_error = ops_and_tens['TD_error']
        weights = ops_and_tens['IS_weights']
        
        #Grab training batch
        #seq_n = self.random_minibatch_sample(batch_size)
        #Add to training queue
        
        leaf_idx,IS_weights,seq_n = self.exp.sample(batch_size)
        IS_weights = np.reshape(IS_weights,(batch_size,1))
        t,td,loss = sess.run([train,TD_error,loss],{s1: seq_n[0],r: seq_n[2],s2: seq_n[3],weights: IS_weights})
        print(loss)
        self.update_exp(leaf_idx,td)
        
        #Add to number of training operations
        self.num_train_ops +=1

    def update_exp(self,leaf_idx,td):
        """
            Function updates the experience tree priorities

        """
        self.exp.update(leaf_idx,td)

    def update_target_params(self,batch_size,n):
        """
        Function updates the target network params

        args:
            batch_size: Int. Number of seperate scenes to consider
            n: int. n^th update operatino
        returns:
            null
        """    

        sess = self.sess
        ops_and_tens = self.ops_and_tens

        s1 = ops_and_tens['s1']
        s2 = ops_and_tens['s2']
        r = ops_and_tens['r']
        action = ops_and_tens['action']
        weights = self.ops_and_tens['IS_weights']
        weights_tmp = np.zeros(shape=[self.batch_size,1])

        zeros = np.zeros(shape=(100,100,self.seq_len)).astype(np.uint8)
        rv = np.zeros((self.batch_size,1))
        self.con_log("UPDATING TARGET PARAMS: {}".format(n),"")
        sess.run([ops_and_tens['target_ops']],{s1: [zeros],s2: [zeros],r: rv,weights: weights_tmp})
        return


    def get_frame(self):
        """
            Function takes snapshot of the current state and returns the frames
        
            args:
                null
            returns:
                frame: numpy array. Screenshot of current state.
        """

        game = self.game
        frame = take_shot(game)
        #append the number of processed frames
        self.process_frames +=  1 
        self.total_frames += 1
        return frame

    def process_seq(self,seq):
        """
            Function process the last seq frames and stacks them for the network

            args:
                seq: List. Containg SARSA for game iteration
            returns:
                np_f: numpy array: Stacked Last four frames of the sequence
        """

        if (len(seq) < self.seq_len):
            frames = [np.zeros(shape=[100,100,1]) for i in range(0,self.seq_len-len(seq))]+ [i for i in seq]
        else:
            frames = seq
        
        np_f = frames[0]
        for i in range(1,len(frames)):
            np_f = np.concatenate((np_f,frames[i]),axis=2)
        return np_f

        #We need to grab every 3rd element in seq. These are the actual frames
        seq_len  = len(seq)     #Determine lenght of seq (it is always changing)
        lower_lim = self.seq_len*3    #We want a input frames of seq_len size. Therefore, we need the lower limit to be 3 times seq_len
        

        #idx contains the index of the last four frames in seq
        idx = [i  for i in range(seq_len-1,seq_len-1-lower_lim,-3) if i >= 0]

        #grab the frames into list
        frames = [seq[i] for i in reversed(idx)]
        
        #If length of frame is less than seq_len. -> Game just started. Add black images
        #This shouldn't effect the training(Everything is zero :))
        len_frames = len(frames)
        add_num = self.seq_len-len_frames
        for i in range(0,add_num):
            frames.insert(0,np.zeros(shape=[100,100,1]))

        #Concatenate the values in frames into a stack of seq_len frames
        np_f = frames[0]
        for i in range(1,len(frames)):
            np_f = np.concatenate((np_f,frames[i]),axis=2)
        return np_f


    def get_action(self,greed,phi1):
        """
        Function returns a greedy/infered action
        
        args:
            greed: float. Probability of chosing a random action
            phi1: Numpy array. State 1 of sarsa
        returns:
            a:  int. Action value
        """

        r_a = np.random.random_sample(1)
        if (r_a <= greed):
            a = np.asarray(np.random.randint(0,3))
            error = 1.0
            self.con_log("##########\tACTION RANDOM\t########## greed: {}".format(greed),"")
        else:
            a,error = self.infer_action(phi1)
            #self.write_label_to_tsv(a)
            self.con_log("##########\tACTION INFERED\t##########","")
        return a,error


    def train_target_update(self,len_exp,batch_size):
        """
        Function performs training and target network updates

        args:
            len_exp:    Int. Lenght of the experience vector
            batch_size: Int. Batch size
        
        returns:
            null:
        """

        if (len(self.exp) < batch_size):
            return
        else:     
            self.execute_train_operation(batch_size)

            if (self.num_train_ops % 10) == 0:
                self.update_target_params(self.seq_len,self.num_train_ops/10)
       
            return
        
        
        if (len_exp >= self.min_exp_len_train):
            self.execute_train_operation(batch_size)
            if (self.num_train_ops % 10) == 0:
                self.update_target_params(self.seq_len,self.num_train_ops/10)
    

    def Q_Algorithm(self):
        """
        Function implements the Q algorithm with experience replay.
        It is responsible for playing the game!

        args:
            null
        returns:
            null
        """

        global t1
        # t1 = time.perf_counter()
    
        #Grabbing variables used for training
        #process_frames = self.process_frames
        #frame_limit = self.frame_limit
        greed_frames = self.greed_frames
        batch_size = self.batch_size
        self.infer_action(np.zeros((100,100,self.seq_len)))
        self.game.click_play()        
        while (self.process_frames < self.frame_limit):       #While the number of processed frames is less than total training frame limit
                #wait_for(1)
                #Declare params for one iteration of the game
                iter_reward = 0
                reward_list = []    #populate with reward for each iteration
                seq = []

                #Get the first frame of the game and append to seq
                frame = self.get_frame()
                #Thread(target=save_img,args=(frame,0)).start()
                
                seq.append(frame)

                #Process the first frame of game for inference                
                phi1 = self.process_seq(seq)               
                #Now play the game!
                while not self.game.stop_play:   #While still playing the game
                    
                    # t1 = time.perf_counter()
                    #If esc key is pressed stop the game play!
                    if self.force_kill:
                        self.game.stop_play = True
                        break

                    #Get the greed
                    greed = self.get_greed()

                    #Determine the action to take
                    a,error = self.get_action(greed,phi1)

                    #send the action to the game controller to perform it
                    frame,r,iter_reward,unique = self.send_action_to_game_controller(phi1,a,iter_reward)
                    if not unique:
                        self.total_frames-=1
                        self.process_frames-=1
                        print("Not unqiue Frames. Don't add to experience")
                    #
                    if unique:
                        #Thread(target=save_img,args=(frame,a)).start()
                        #Increment reward for the game iteration
                        iter_reward +=r
                        
                        #Add the action, reward and new frame to the game play sequence
                        if (len(seq) >= self.seq_len):
                            seq.pop(0)
                        seq.append(frame)
                        self.con_log("SEQUENCE LENGTH = {}".format(len(seq)),"")
                        #Process the new frames for storing!
                        phi2 = self.process_seq(seq)

                        #Store the previous experience
                        self.store_exp(error,(phi1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),phi2))

                        #Now assign phi1 to be the new frame!....For inference!
                        phi1 = phi2

                        #Print length of experience vector
                        self.con_log("Len Exp Vector: {}".format(len(self.exp)),"")


                    #Run a training and update target params operation
                    self.train_target_update(len(self.exp),batch_size)
                
                
                #OUTSIDE WHILE LOOP
                #We have lost the game
                #Wait for a few milliseconds
                wait_for(.3)

                reward_list.append(iter_reward)
                self.create_reward_plot(reward_list)
                #Reset the internal game reward to zero
                self.game.reward = 0
                
                #If escape key is pressed break this while loop too! -> We are passing new params for the trainer :)
                if self.force_kill:
                    break

                #If game ended naturally...    
                self.game.stop_play = False

                #new_game_element = self.game.chrome.find_element_by_xpath('/html/body/div[2]/div[2]/a')
                #new_game_element.click()
                self.game.click_play()

                self.game_play_iteration = self.game_play_iteration+1
                self.print_progress(greed)

    def create_reward_plot(self,r_iter):
        """
        Funciton creates a plot of the reward vs game iteration.

        args:
            g_iter: int. NNumber of total games played
            r_iter: List of ints. List containing reward for each game iterstion
        """

        len_re = len(r_iter)
        if len_re == 0 or len_re == 1:
            return

        #Games vec
        g_vec = np.arange(0,len(r_iter))

        plt.plot(g_vec,r_iter)
        plt.title("Reward for game iteration")
        plt.savefig(r"E:\vishnu\SSPlayer\log2\rewardplt.png")

    def write_label_to_tsv(self,label):
        """
        Function constructs a tsv of labels
        args:
            label: Int. The label to write to file
        """
        with open(self.tsv_file,'w') as tsvf:
                tsvf.write(str(label)+'\n')


    def stop_training(self,key):
        """
        Function is used by key listener to stop training
        
        args:
            key:    Key Object.
        """
        self.con_log("",key)
        if key == keyboard.Key.esc:
            self.con_log("Kill loop","")
            self.force_kill = True
            return

    def esc_wrap(self,function):
        """
            Function wraps function in escape key to end training

            args:
                function: Function to wrap with keyboard listener

            returns:
                rtn_val: Return value of function
        """
        
        #Wrapping trainer inside a listener to stop training if esc key is pressed
        with keyboard.Listener(on_press=self.stop_training) as listener:
            #perform the Q algorithm with experience replay
            rtn_val = function()
            listener.stop()


        return rtn_val

    def play_train(self,x,n):
        """
        Function runs n play operations every x process frames
        
        args:
            x: int. Number of process frames to run before running play operations
            n: Number of times to play the game
        """

        #Function to wrap in esc_key
        def fun():        
            #Total iterations of play train
            total_run_times =  int(self.frame_limit/x)

            #avg reward for n play. Length is total run times
            reward_list = []

            for i in range(0,total_run_times):  
                #reassign process frames
                self.frame_limit = x
                #Run training operation for x frames
                self.Q_Algorithm()

                #Play game
                avg_reward = self.play(n)
                reward_list.append(avg_reward)

                #reset process_frames to 0
                self.process_frames = 0

                if self.force_kill:
                    break
            self.con_log("Reward List: ",reward_list)
            self.create_reward_plot(reward_list)

        self.esc_wrap(fun)

    def play(self,n):
        """
        Fucntion plays the game

        args:
            n: int. Number of times to play the game

        returns:
            rtn_val: double. Average reward for n iterations
        """
        #Declare return value
        rtn_val = 0

        #Game game session
        game = self.game
        game.stop_play = False
        #Start a new game
        #new_game_element = game.chrome.find_element_by_xpath('/html/body/div[2]/div[2]/a')
        #new_game_element.click()
        self.infer_action(np.zeros((100,100,self.seq_len)))
        game.click_play()

        #reward list
        reward_list = []
        #Play n times
        for i in range(0,n):
            #wait_for(1)
            self.con_log('Play iteration: ',i+1)
            
            #single game iteration variable
            reward = 0
            seq = []

            #Get the first frame of the game and append to seq
            frame = self.get_frame()
            seq.append(frame)

            #Process the first frame of game for inference                
            phi1 = self.process_seq(seq)  
            #While game oes not need to stop
            not_uniq_count = 0
            quit_game_play = False
            while not game.stop_play:
                #If esc key is pressed stop the game play!
                if self.force_kill:
                    self.game.stop_play = True
                    break
                
                a,error = self.infer_action(phi1)
                frame,r,d_reward,unique = self.send_action_to_game_controller(phi1,a,0)
                self.total_frames -= 1  #Total frames get appended every time new frames are grabbed. These are only used for greed

                #If network continues to breadic unique 
                if not unique :
                    not_uniq_count +=1
                    if (not_uniq_count == 25):
                        self.con_log("INFINITE LOOP: BREAKING PLAY ITERATION {}".format(i),"")
                        break
                else:
                    not_uniq_count = 0

                    
                #Increment reward for the game iteration
                reward +=r
                
                #Add the action, reward and new frame to the game play sequence
                if (len(seq) >= self.seq_len):
                    seq.pop(0)
                seq.append(frame)

                #Process the new frames for storing!
                phi2 = self.process_seq(seq)
                phi1 = phi2


            reward_list.append(reward)

            if self.force_kill:
                break

            #If game ended naturally...    
            self.game.stop_play = False

            #new_game_element = self.game.chrome.find_element_by_xpath('/html/body/div[2]/div[2]/a')
            #new_game_element.click()
            game.click_play()

            
        
        #Calc avg
        rtn_val = np.mean(reward_list)
        return rtn_val

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

    #FUNCTIONS BELOW NEED TO BE FIXED

    #TODO: Function needs to be modifed to train for game iterations
    def iter_train_reward(self,sess,game,frame_limit,greed_frames,batch_size,ops_and_tens,gsheets):
        """
            Function responsible for playing the game

            args:
                sess:   Tensorflow session.
                game:   GameController instance.
                frame_limit:    int. Number of frames to train for
                greed_frames:   int. Number of frames where greed is present.
                batch_size: int. Size of training batch.
                ops:    Dict. Relevent tensorflow operations.
                phs:    Dict. Relevent tensorflow placeholders.

            returns:

        """


        global process_frames
        gp = 0
        #cv2.imshow("frame",np.zeros(shape=(110,110,1)))
        #cv2.waitKey(1)
        #winname = "frame"
        #cv2.namedWindow(winname) 
        #cv2.moveWindow(winname, 2700,300)

        force_kill = False

        with keyboard.Listener(on_press=stop_training) as listener:
            while (process_frames < frame_limit):
                reward = 0
                wait_for(1)
                frame,bval = get_frame(game)
                fff =[]
                seq = []
                seq.append(frame)
                fff.append(frame)
                phi1 = process_seq(seq)        
                while not game.stop_play:
                    greed = get_greed(greed_frames,process_frames)
                    #cv2.imshow(winname,frame)
                    #cv2.waitKey(1)
                    r_a = np.random.random_sample(1)
                    if (r_a <= greed):
                        #self.con_log(greed,"greedy: ")
                        a = np.asarray(np.random.randint(0,4))
                    else:
                        a = infer_action(sess,phi1,ops_and_tens)
                        #self.con_log(a,q)
                    frame,stop_play,r,reward = send_action_to_game_controller(game,phi1,a,reward)
                    seq.append(a)
                    seq.append(r)
                    seq.append(frame)
                    fff.append(frame)
                    phi2 = process_seq(seq)
                    store_exp((phi1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),phi2))
                    if stop_play:
                        break
                    phi1 = phi2
                    if (len(exp) > batch_size):
                        #dist_add_to_queue(sess,batch_size,ops,phs) 
                        #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE execute train
                        x = 5
                    if force_kill:
                        game.stop_play = True
                wait_for(.3)
                game.reward = 0
                if force_kill:
                    break    
                game.stop_play = False
                gp = gp+1
                if (gp % 50 == 0):
                    self.con_log("Exp size: ", len(exp))
                    self.con_log("Number process Frames: ",process_frames)
                    self.con_log("greed: ",greed)
                save_seq_img(fff)
            listener.stop()
        return