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

def infer_action(sess,frames,ops_and_tens):
    """
        Function infers action from the inference network
    
        args:
            sess: Tensorflow session
            frames: numpy array. Frames that are passed to the inference network
            ops: Dict. Contains relevant operations from the tensorflow gram
        return:
            a: int. action predicted by the inference network
            q: float. Q value of the action
    """

    s1 = ops_and_tens['s1']
    s2 = ops_and_tens['s2']
    r = ops_and_tens['r']
    action = ops_and_tens['action']

    zeros = np.zeros(shape=frames.shape).astype(np.uint8)

    a = sess.run([action],{s1: [frames],s2: [zeros],r: [[0]]})
    return a[0]

def send_action_to_game_controller(game,phi1,a,reward):
    """
        Functions performs action in the game
        args:
            game:  GameController instance.
            phi1: numpy array. First state in SARSA
            a: int. Action to send to the controller.
            reward: float. Reward for the current game iteration
        return:
            frame: numpy array: Second state in SARSA
            bval: Bool: States whether or not the game is currently being played
            r: int.Reward associated with the taken action.
            reward: float. Updated total reward for the current game iteration
    """ 
    
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
    
    print("a:{},dir:{}".format(str(a),m_dir))

    #Waiting for the graphics to catch up
    wait_for(.5)

    #Determine the new state of the game
    frame,bval = get_frame(game)



    """
    #Try catch checks if game has ended by looking for pop up window
    try:
        #If game has ended there will be no error
        game.chrome.find_element_by_xpath("/html/body/div[2]/div[3]/div[1]/div/a[2]").click()
        game.stop_play = True   #Stoping play
        r = -1  #reward associated with this action is -1
    except:
        #If game has not ended then there will be an error
        #IF the internal reward for the game iteration has changed 
        #then the reward has increased. Therefore the action taken
        #as an associated reward of 1. Else 0.
        
        
        #if  game.reward == reward:
        #    r = 0
        #elif game.reward > reward:
        #    r = 1
        #r = game.reward-reward
        r = game.get_reward2()
        reward = 0
        pass
    """

    play_again_elem = game.chrome.find_element_by_xpath("/html/body/div[2]/div[3]/div[1]/div/a[2]")

    if (play_again_elem.is_displayed()):
        game.stop_play = True
        r = -1
        play_again_elem.click()
    else:
        r = game.get_reward2()
        reward = 0
    


    #if frames are equal then invalid move..reinfer action
    chk_frm = phi1[:,:,0]
    if np.array_equal(chk_frm,np.squeeze(frame)):
        return 0
    else:
        return frame,bval,r,reward


def random_minibatch_sample(batchsize):
    """
        Function randomly samples batch from exp buffer
        args:
            batchsize: int. Size of the batch
        returns:
            tuple with batchsize number SARS pairs
    """
    
    global exp
    #Get random indexes
    line_N = np.random.randint(0,len(exp),size=batchsize)

    #Extract batch size number of elements from array
    img1s = np.array([exp[i][0] for i in line_N])
    a = np.array([exp[i][1] for i in line_N]).squeeze().reshape(batchsize,1)
    r = np.array([exp[i][2] for i in line_N]).squeeze().reshape(batchsize,1)
    img2s = np.array([exp[i][3] for i in line_N])

    return (img1s,a,r,img2s)
    
def store_exp(seq):
    """
        Function stores new exp into experience buffer
        args:
            seq: tuple. new experience to be added to buffer
        returns:
    """

    global exp
    global process_frames

    #Older experience is phased out by poping from exp buffer
    if (process_frames > 10000):
        exp.pop(0)
    #process_frames = process_frames+2

    #add new experience
    exp.append(seq)    
    return

def get_greed(greed_frames,frames):
    """
        Function returns greed based on linear relationship
        
        args:
            greed_frames: int. Number of frames for which greed changes
            frames: int. Total number of trainer frames
        
        return:
            float. Greed value that linearly ranges from 1->.1
    """
    if frames > greed_frames:
        return 0.1
    return (((.1-1)/greed_frames)*frames)+1

def execute_train_operation(sess,batch_size,ops_and_tens,n):
    """
        Function executes a training operation
        
        args:
            sess: Tensorflow Session.
            batch_size: int. Training batch size.
            ops_and_tens: Dictionary containing operations and tensors
            n: Int. n^th training operation
        return:
    
    """

    print("######\tTRAINING OPERATION: {}\t#####".format(n))
    train = ops_and_tens['train']
    s1 = ops_and_tens['s1']
    s2 = ops_and_tens['s2']
    r = ops_and_tens['r']
    prt = ops_and_tens['print']

    #Grab training batch
    seq_n = random_minibatch_sample(batch_size)
    #Add to training queue
    sess.run([train,prt],{s1: seq_n[0],r: seq_n[2],s2: seq_n[3]})

def update_target_params(sess,ops_and_tens,n):
    """
    Function updates the target network params

    args:
        sess:   Tensorflow session
        ops_and_tens:   Dictionary. Available operations and tensors
        n: int. n^th update operatino
    returns:
        null
    """    
    s1 = ops_and_tens['s1']
    s2 = ops_and_tens['s2']
    r = ops_and_tens['r']
    action = ops_and_tens['action']

    zeros = np.zeros(shape=(100,100,4)).astype(np.uint8)
    
    print("#####\tUPDATING TARGET PARAMS: {}\t######".format(n))
    sess.run([ops_and_tens['target_ops']],{s1: [zeros],s2: [zeros],r: [[0]]})
    return

def get_frame(game):
    """
        Function takes snapshot of the current state and returns the frames
    
        args:
            games: GameController instance.
        returns:
            frame: numpy array. Screenshot of current state.
            bval: bool. True if current game iter is still in play.
    """


    global process_frames
    frame = take_shot(game)
    #frame = game.take_shot()
    bval = game.stop_play

    #append the number of processed frames
    process_frames = process_frames+1
    return frame,bval

def process_seq(seq,batch_size):
    """
        Function process the last batchsize frames and stacks them for the network

        args:
            seq: List. Containg SARSA for game iteration
            batch_size: int. Batch_size(Group sequences by)
        returns:
            np_f: numpy array: Stacked Last four frames of the sequence
    """

    #Determine lenght of seq (it is always changing)
    seq_len = len(seq)
    lower_lim = batch_size*3
    #idx contains the index of the last four frames in seq
    #idx = [i  for i in range(seq_len-1,seq_len-1-12,-3) if i >= 0]
    idx = [i  for i in range(seq_len-1,seq_len-1-lower_lim,-3) if i >= 0]
    
    #grab the frames into list
    frames = [seq[i] for i in idx]
    
    #If lenght of frame is less than 4. -> Game just started. Add black images
    #This shouldn't effect the training(Everything is zero :))
    len_frames = len(frames)
    add_num = 4-len_frames
    for i in range(0,add_num):
        #frames.append(np.zeros(shape=[110,142,1]))
        frames.insert(0,np.zeros(shape=[100,100,1]))

    #Concatenate the values in frames into a stack of 4 frames
    np_f = frames[0]
    for i in range(1,len(frames)):
        np_f = np.concatenate((np_f,frames[i]),axis=2)
    return np_f


def get_action(greed,sess,phi1,ops_and_tens):
    """
    Function returns a greedy/infered action
    
    args:
        greed: float. Probability of chosing a random action
        sess: Tensorflow session
        phi1: Numpy array. State 1 of sarsa
        ops_and_tens:    Dictionary. Contains tensorflow operations
    """

    r_a = np.random.random_sample(1)
    if (r_a <= greed):
        a = np.asarray(np.random.randint(0,4))
        print("##########ACTION RANDOM######## greed: {}".format(greed))
    else:
        a = infer_action(sess,phi1,ops_and_tens)
        print("##########ACTION INFERED")
    return a

def train_target_update(sess,ops_and_tens,len_exp,batch_size,num_train_ops):
    """
    Function performs training and target network updates

    args:
        sess:       Tensorflow session
        len_exp:    Int. Lenght of the experience vector
        batch_size: Int. Batch size
        ops_and_tens: Tensorflow operations and tensors
        num_trian_ops:   Int. Nth training operation
    
    returns:
        num_train_ops
    """
    if (len_exp > batch_size):
        execute_train_operation(sess,batch_size,ops_and_tens,num_train_ops)
        num_train_ops = num_train_ops+1
        if (num_train_ops % 10) == 0:
            update_target_params(sess,ops_and_tens,num_train_ops/10)
    
    return num_train_ops

def frame_train_reward(sess,game,frame_limit,greed_frames,batch_size,ops_and_tens,gsheets):
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

    force_kill = False

    def stop_training(key):
        print(key)
        nonlocal game
        nonlocal force_kill
        if key == keyboard.Key.esc:
            print("Kill loop")
            force_kill = True
            game.chrome.set_window_position(50,50)
            return

    with keyboard.Listener(on_press=stop_training) as listener:
        num_train_ops = 0
        
        
        while (process_frames < frame_limit):
            reward = 0
            wait_for(1)
            frame,bval = get_frame(game)
            seq = []
            seq.append(frame)
            phi1 = process_seq(seq,batch_size)        
           
           #While still playing game
            while not game.stop_play:
                greed = get_greed(greed_frames,process_frames)
                
                #If actions are invalid...keep trying new actions
                while True:
                    a = get_action(greed,sess,phi1,ops_and_tens)
                    rtn_val = send_action_to_game_controller(game,phi1,a,reward,)
                    if rtn_val != 0:
                        frame,stop_play,r,reward = rtn_val
                        break
                
                seq.extend((a,r,frame))
                phi2 = process_seq(seq,batch_size)
                store_exp((phi1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),phi2))
                if stop_play:
                    break
                phi1 = phi2
                print("##### Len Exp Vector: {} #####".format(len(exp)))
                num_train_ops = train_target_update(sess,ops_and_tens,len(exp),batch_size,num_train_ops)
                if force_kill:
                    game.stop_play = True
            wait_for(.3)
            game.reward = 0
            if force_kill:
                break    
            game.stop_play = False
            gp = gp+1
            print_progress(gp,greed)
        listener.stop()
    return

def print_progress(gp,greed):
    """
    Function prints, relevent progress metrics

    args:
        gp: Int. Game Play iteration number
        greed: Float. Greed value
    """
    global process_frames, exp
    if (gp % 50 == 0):
        print("Exp size: ", len(exp))
        print("Number process Frames: ",process_frames)
        print("greed: ",greed)

def play(sess,game,M,ops_and_tens):
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


def save_seq_img(seq):
    for i in range(0,len(seq)):
        Image.fromarray(np.squeeze(seq[i])).save("imgs/test"+str(i)+".png")

def iter_train_reward(sess,game,frame_limit,greed_frames,batch_size,ops_and_tens,gsheets):
    #TODO: Function needs to be modifed to train for game iterations
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