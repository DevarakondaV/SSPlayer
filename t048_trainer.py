import time
from timeit import timeit,Timer
import numpy as np
from PIL import Image
from t048_con import *
import sys
import os
import matplotlib.pyplot as plt
#import cv2

def dist_infer_action(sess,frames,ops,phs):
    action = ops['action']
    x1 = phs['x1']
    q = ops['q_vals_pr']
    a,q = sess.run([action,q],{x1: [frames]})
    return a,q

def send_action_to_game_controller(game,phi1,a,reward):
    
    if (a == 0):
        game.move(game.up)
    elif (a == 1):
        game.move(game.down)
    elif (a == 2):
        game.move(game.left)
    elif (a == 3):
        game.move(game.right)
    
    wait_for(.5)
    frame,bval = get_frame(game)

    #game.retry_button.click()
    try:
        game.chrome.find_element_by_xpath("/html/body/div[2]/div[3]/div[1]/div/a[2]").click()
        game.stop_play = True
        r = -1
    except:
        #print("never")
        if  game.reward == reward:
            r = 0
        elif game.reward > reward:
            r = 1
        reward = game.reward
        pass

    #print("reward: ", r)
    return frame,bval,r,reward

def random_minibatch_sample(batchsize):
    global exp
    line_N = np.random.randint(0,len(exp),size=batchsize)

    img1s = np.array([exp[i][0] for i in line_N])
    a = np.array([exp[i][1] for i in line_N]).squeeze().reshape(batchsize,1)
    r = np.array([exp[i][2] for i in line_N]).squeeze().reshape(batchsize,1)
    img2s = np.array([exp[i][3] for i in line_N])

    return (img1s,a,r,img2s)
    
def store_exp(seq):
    global exp
    global process_frames
    if (process_frames > 500000):
        exp.pop(0)
    #process_frames = process_frames+2
    exp.append(seq)    
    return

def get_greed(greed_frames,frames):
    if frames > greed_frames:
        return 0.1
    return (((.1-1)/greed_frames)*frames)+1


def get_frame(game):
    global process_frames
    frame = take_shot(game)
    bval = game.stop_play
    process_frames = process_frames+1
    return frame,bval

def process_seq(seq):
    #process last 4 frames and stacks for network

    seq_len = len(seq)
    idx = [i  for i in range(seq_len-1,seq_len-1-12,-3) if i >= 0]
    frames = [seq[i] for i in idx]
    len_frames = len(frames)
    add_num = 4-len_frames
    for i in range(0,add_num):
        #frames.append(np.zeros(shape=[110,142,1]))
        frames.insert(0,np.zeros(shape=[100,100,1]))

    np_f = frames[0]
    for i in range(1,len(frames)):
        np_f = np.concatenate((np_f,frames[i]),axis=2)
    return np_f



def dist_add_to_queue(sess,batch_size,ops,phs):
    enqueue_op = ops['enqueue_op']
    s_img1 = phs['s_img1']
    s_a = phs['s_a']
    s_r = phs['s_r']
    s_img2 = phs['s_img2']

    seq_n = random_minibatch_sample(batch_size)
    sess.run([enqueue_op],{s_img1: seq_n[0],s_a: seq_n[1], s_r: seq_n[2],s_img2: seq_n[3]})


def frame_train_reward(sess,game,frame_limit,greed_frames,batch_size,ops,phs,gsheets):
    global process_frames
    gp = 0
    #cv2.imshow("frame",np.zeros(shape=(110,110,1)))
    #cv2.waitKey(1)
    #winname = "frame"
    #cv2.namedWindow(winname) 
    #cv2.moveWindow(winname, 2700,300)
    while (process_frames < frame_limit):
        greed = get_greed(greed_frames,process_frames)
        reward = 0
        wait_for(1)
        
        frame,bval = get_frame(game)
        fff =[]
        seq = []
        seq.append(frame)
        fff.append(frame)
        phi1 = process_seq(seq)        
        while not game.stop_play:
            #cv2.imshow(winname,frame)
            #cv2.waitKey(1)
            r_a = np.random.random_sample(1)
            if (r_a <= greed):
                a = np.asarray(np.random.randint(0,5))
            else:
                a,q = np.array(dist_infer_action(sess,phi1,ops,phs)).astype(np.float16)
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
            print(len(exp),process_frames)
        wait_for(.3)
        game.reward = 0
        game.stop_play = False
        gp = gp+1
        #print(len(exp),process_frames)
        if (gp % 50 == 0):
            print("Exp size: ", len(exp))
            print("Number process Frames: ",process_frames)
            print("greed: ",greed)
        save_seq_img(fff)
        if (gp == 2):
            break
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