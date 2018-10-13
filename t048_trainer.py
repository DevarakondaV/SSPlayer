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

def dist_infer_action(sess,frames,ops,phs):
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

    run_metadata = tf.RunMetadata()

    action = ops['action']
    x1 = phs['x1']
    q = ops['q_vals_pr']

    s_img1 = phs['s_img1']
    s_a = phs['s_a']
    s_r = phs['s_r']
    s_img2 = phs['s_img2']
    print("Before infer")
    
    #a,q = sess.run([action,q],{x1: [frames]})
    #a,q = sess.run([action,q],{x1: [frames]},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
    a,q = sess.run([action,q],{s_img1: np.random.rand(5,100,100,4),s_a: np.random.rand(5,1).astype(np.float16), s_r: np.random.rand(5,1).astype(np.float16),s_img2: np.random.rand(5,100,100,4),x1: [frames]})
    #a,q = sess.run([action],{x1: [np.zeros((100,100,4))]})
    print("QQQ",q)
    return a,q

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
        print(a,"up")
        game.move(game.up)
    elif (a == 1):
        print(a,"down")
        game.move(game.down)
    elif (a == 2):
        print(a,"left")
        game.move(game.left)
    elif (a == 3):
        print(a,"right")
        game.move(game.right)
    else:
        print(a,"what")
    
    #Waiting for the graphics to catch up
    wait_for(.5)

    #Determine the new state of the game
    frame,bval = get_frame(game)

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
        
        
        if  game.reward == reward:
            r = 0
        elif game.reward > reward:
            r = 1
        #r = game.reward-reward
        reward = game.reward
        pass
    
    #if frames are equal then reward needs to be hanged to -1
    chk_frm = phi1[:,:,0]
    if np.array_equal(chk_frm,np.squeeze(frame)):
        r = -.3
    print("state r: ",r)
    #print("reward: ", r)
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
    bval = game.stop_play

    #append the number of processed frames
    process_frames = process_frames+1
    return frame,bval

def process_seq(seq):
    """
        Function process the last 4 frames and stacks them for the network

        args:
            seq: List. Containg SARSA for game iteration
        returns:
            np_f: numpy array: Stacked Last four frames of the sequence
    """

    #Determine lenght of seq (it is always changing)
    seq_len = len(seq)
    #idx contains the index of the last four frames in seq
    idx = [i  for i in range(seq_len-1,seq_len-1-12,-3) if i >= 0]
    
    #grab the frames into list
    frames = [seq[i] for i in idx]
    
    #If lenght of frame is less than 4. -> Game just started. Add black images
    #This shouldn't effect the training
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



def dist_add_to_queue(sess,batch_size,ops,phs):
    """
        Function adds new training sequences to train queue
        
        args:
            sess: Tensorflow Session.
            batch_size: int. Training batch size.
            ops:   Dict. Contains relevent tensorflow opeartions
            phs: Dict. Contains relevent tensorflow placeholders
        return:
    
    """
    #Grab the enqueue operation and placeholders
    enqueue_op = ops['enqueue_op']
    s_img1 = phs['s_img1']
    s_a = phs['s_a']
    s_r = phs['s_r']
    s_img2 = phs['s_img2']
    x1 = phs['x1']

    #Grab training batch
    seq_n = random_minibatch_sample(batch_size)
    #Add to training queue
    sess.run([enqueue_op],{s_img1: seq_n[0],s_a: seq_n[1], s_r: seq_n[2],s_img2: seq_n[3],x1: np.random.rand(1,100,100,4)})


def frame_train_reward(sess,game,frame_limit,greed_frames,batch_size,ops,phs,gsheets):
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
                    #a,q = np.array(dist_infer_action(sess,phi1,ops,phs)).astype(np.float16)
                    a,q = dist_infer_action(sess,phi1,ops,phs)
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
                    dist_add_to_queue(sess,batch_size,ops,phs)
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


def dist_play(sess,game,M,ops,phs):
    for i in range(0,M):
        reward = 0
        wait_for(1)
        game.click_play()
        while not game.stop_play:
            frames1,test = get_4_frames(game)
            if  test:
                break
            #a,q= np.asarray(dist_infer_action(sess,frames,ops,phs)).astype(np.float16)
            a,q = dist_infer_action(sess,frames1,ops,phs)
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






exp = []
process_frames = 0