import time
from timeit import timeit,Timer
import numpy as np
from PIL import Image
from t048_con import *
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
#import cv2


from pynput import keyboard


class Trainer:
    """
    Class is responsible for training
    """

    #Main Methods
    def __init__(self,sess,game,frame_limit,greed_frames,seq_len,batch_size,ops_and_tens,gsheets):
        """
        Constructor
        args:
            sess:   Tensorflow session object
            game:   Game object
            frame_limit:    int. Number of frames to play
            greed_frames:    int. Number of frames when greed is active
            seq_len:     int. Sequence size which determines state
            batch_size:     int. Determines batchsize of training operation
            ops_and_tens:   Dictionary. Contains the relevent tensorflow operations and tensors
            gsheets:        gsheets object. Used to post results to google sheets.
        returns:
            Null
        """

        #Declaring some variables
        self.exp = []                       #experience vector
        self.process_frames = process_frames                 #Number of frames to process
        self.force_kill = False                                 #Bool param determines if training should be forced to stop
        self.sess = sess                    #Tensorflow session
        self.game = game                    #Game object
        self.frame_limit = frame_limit      #Maximum number of frame sto play  
        self.greed_frames = greed_frames    #Maximum number of frames when greed is calculated
        self.seq_len = batch_size           #The number of frames which determines a state
        self.ops_and_tens = ops_and_tens    #Tensorflow operations and tensors
        self.gsheets = gsheets              #gsheets object ot post to sheets
        self.batch_size = batch_size        #Determines batchsize of training operation

        #Params used while trianing
        self.game_play_iteration = 0        #Number of iterations of game play
        self.num_train_ops = 0              #Number of trianing operations
        
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
        action = self.ops_and_tens['action']

        #Need dummy value for placeholders not in use
        zeros = np.zeros(shape=frames.shape).astype(np.uint8)

        #Inference
        a = self.sess.run([action],{s1: [frames],s2: [zeros],r: [[0]]})
        return a[0]

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
        if (a == 0):
            m_dir = "up"
            game.move(game.up)
        elif (a == 1):
            m_dir = "down"
            game.move(game.down)
        elif (a == 2):
            m_dir = "left"
            game.move(game.left)
        elif (a == 3):
            m_dir = "right"
            game.move(game.right)
        else:
            m_dir = "unknown"

        
        #Waiting for the graphics to catch up
        wait_for(.5)

        return m_dir

    def get_reward_for_action(self):
        """
        Function determines the reward associated with the taken action by looking at game object

        """

        game = self.game
        
        #Grab play again div using xpath
        play_again_elem = game.chrome.find_element_by_xpath("/html/body/div[2]/div[3]/div[1]/div/a[2]")
        
        #If play again div is displayed stop play!
        if (play_again_elem.is_displayed()):
            game.stop_play = True
            r = -1
            play_again_elem.click()
        else:
            r = game.get_reward2()
        
        return r


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

        #Perform the action in the game
        m_dir = self.perform_action(a)
        #Determine the new state of the game
        frame = self.get_frame()
        #Get the reward for the action!
        r = self.get_reward_for_action()
        
        #
        reward = 0 
        

        print("Action = {}\nMove dir = {}\nReward = {}".format(str(a),m_dir,r))
        #if frames are equal then invalid move..reinfer action
        chk_frm = phi1[:,:,0]
        if np.array_equal(chk_frm,np.squeeze(frame)):
            return 0
        else:
            return frame,r,reward



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

    def store_exp(self,seq):
        """
            Function stores new exp into experience buffer
            args:
                seq: tuple. new experience to be added to buffer
            returns:
        """

        #Older experience is phased out by poping from exp buffer
        if (self.process_frames > 10000):
            self.exp.pop(0)
        #process_frames = process_frames+2

        #add new experience
        self.exp.append(seq)    
        return

    def get_greed(self):
        """
            Function returns greed based on linear relationship
            
            return:
                float. Greed value that linearly ranges from 1->.1
        """

        
        frames = self.process_frames
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

        print("TRAINING OPERATION: {}".format(self.num_train_ops))
        train = ops_and_tens['train']
        s1 = ops_and_tens['s1']
        s2 = ops_and_tens['s2']
        r = ops_and_tens['r']
        prt = ops_and_tens['print']

        #Grab training batch
        seq_n = self.random_minibatch_sample(batch_size)
        #Add to training queue
        sess.run([train],{s1: seq_n[0],r: seq_n[2],s2: seq_n[3]})

        #Add to number of training operations
        self.num_train_ops +=1

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

        zeros = np.zeros(shape=(100,100,batch_size)).astype(np.uint8)
        
        print("UPDATING TARGET PARAMS: {}".format(n))
        sess.run([ops_and_tens['target_ops']],{s1: [zeros],s2: [zeros],r: [[0]]})
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
        self.process_frames = self.process_frames+1
        return frame

    def process_seq(self,seq):
        """
            Function process the last batchsize frames and stacks them for the network

            args:
                seq: List. Containg SARSA for game iteration
            returns:
                np_f: numpy array: Stacked Last four frames of the sequence
        """

        #We need to grab every 3rd element in seq. These are the actual frames
        seq_len  = len(seq)     #Determine lenght of seq (it is always changing)
        lower_lim = self.seq_len*3    #We want a input frames of seq_len size. Therefore, we need the lower limit to be 3 times seq_len
        

        #idx contains the index of the last four frames in seq
        idx = [i  for i in range(seq_len-1,seq_len-1-lower_lim,-3) if i >= 0]
        
        #grab the frames into list
        frames = [seq[i] for i in idx]
        
        #If lenght of frame is less than seq_len. -> Game just started. Add black images
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
            a = np.asarray(np.random.randint(0,4))
            print("##########\tACTION RANDOM\t########## greed: {}".format(greed))
        else:
            a = self.infer_action(phi1)
            print("##########\tACTION INFERED\t##########")
        return a


    def train_target_update(self,len_exp,batch_size):
        """
        Function performs training and target network updates

        args:
            len_exp:    Int. Lenght of the experience vector
            batch_size: Int. Batch size
        
        returns:
            null:
        """

        if (len_exp > batch_size):
            self.execute_train_operation(batch_size)
            if (self.num_train_ops % 10) == 0:
                self.update_target_params(batch_size,self.num_train_ops/10)
    

    def Q_Algorithm(self):
        """
        Function implements the Q algorithm with experience replay.
        It is responsible for playing the game!

        args:
            null
        returns:
            null
        """


        #Grabbing variables used for training
        process_frames = self.process_frames
        frame_limit = self.frame_limit
        greed_frames = self.greed_frames
        batch_size = self.batch_size

        while (process_frames < frame_limit):       #While the number of processed frames is less than total training frame limit
                wait_for(1)

                #Declare params for one iteration of the game
                iter_reward = 0
                seq = []

                #Get the first frame of the game and append to seq
                frame = self.get_frame()
                seq.append(frame)

                #Process the first frame of game for inference                
                phi1 = self.process_seq(seq)        
            
                
                #Now play the game!
                while not self.game.stop_play:   #While still playing the game
                    
                    #Get the greed
                    greed = self.get_greed()
                    
                    #Retry actions until no longer an invalid action. Sometimes the board state does not allow for 
                    #certain actions to be taken despite not having lost the game. Do not consider these as relevant in
                    #determining the policy function. Therefore, we loop until we take an action that is not invalid.
                    while True:
                        #Determine the action to take
                        a = self.get_action(greed,phi1)

                        #send the action to the game controller to perform it
                        rtn_val = self.send_action_to_game_controller(phi1,a,iter_reward)
                        
                        #if the rtn_val is not zero then the action is valid....proceed with computation!
                        if rtn_val != 0:
                            frame,r,iter_reward = rtn_val
                            break
                    
                    #Add the action, reward and new frame to the game play sequence
                    seq.extend((a,r,frame))

                    #Process the new frames for storing!
                    phi2 = self.process_seq(seq)

                    #Store the previous experience
                    self.store_exp((phi1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),phi2))

                    #Now assign phi1 to be the new frame!....For inference!
                    phi1 = phi2

                    #Print length of experience vector
                    print("Len Exp Vector: {}".format(len(self.exp)))

                    #Run a training and update target params operation
                    self.train_target_update(len(self.exp),batch_size)

                    #If esc key is pressed stop the game play!
                    if self.force_kill:
                        self.game.stop_play = True
                
                
                #OUTSIDE WHILE LOOP
                #We have lost the game
                #Wait for a few milliseconds
                wait_for(.3)

                #Reset the internal game reward to zero
                self.game.reward = 0
                
                #If escape key is pressed break this while loop too! -> We are passing new params for the trainer :)
                if self.force_kill:
                    break

                #If game ended naturally...    
                self.game.stop_play = False
                self.game_play_iteration = self.game_play_iteration+1
                self.print_progress(greed)

    def stop_training(self,key):
        """
        Function is used by key listener to stop training
        
        args:
            key:    Key Object.
        """
        print(key)
        if key == keyboard.Key.esc:
            print("Kill loop")
            self.force_kill = True
            return

    def train(self):
        """
            Function starts training

            args:

            returns:

        """
        
        #Wrapping trainer inside a listener to stop training if esc key is pressed
        with keyboard.Listener(on_press=self.stop_training) as listener:
            #perform the Q algorithm with experience replay
            self.Q_Algorithm()
            listener.stop()


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
            print("Exp size: ", len(self.exp))
            print("Number process Frames: ", self.process_frames)
            print("greed: ",greed)


    def save_seq_img(self,seq):
        for i in range(0,len(seq)):
            Image.fromarray(np.squeeze(seq[i])).save("imgs/test"+str(i)+".png")

    #FUNCTIONS BELOW NEED TO BE FIXED
    #TODO: This function needs to be fixed. Does not work propperly 
    def play(self,M,):

        game = self.game
        sess = self.sess
        for i in range(0,M):
            reward = 0
            wait_for(1)
            game.click_play()
            while not game.stop_play:
                frames1,test = get_4_frames(game)
                if  test:
                    break
                a = infer_action(sess,frames1,ops_and_tens)
                print(a)
                frames2,test,r,reward = send_action_to_game_controller(game,frames1,a,reward)
                if test:
                    break
            wait_for(.3)
            game.reward = 0
            game.stop_play = False
            print("Play Iteration: ",i)

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

        def stop_training(key):
            print(key)
            if key == keyboard.Key.esc:
                print("Kill loop")
                nonlocal force_kill
                force_kill = True
                return

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
                        #print(greed,"greedy: ")
                        a = np.asarray(np.random.randint(0,4))
                    else:
                        a = infer_action(sess,phi1,ops_and_tens)
                        #print(a,q)
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
                    print("Exp size: ", len(exp))
                    print("Number process Frames: ",process_frames)
                    print("greed: ",greed)
                save_seq_img(fff)
            listener.stop()
        return

exp = []
process_frames = 0