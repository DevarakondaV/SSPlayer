
# coding: utf-8

# In[1]:


#imports
import time
from timeit import timeit,Timer
import numpy as np
from GameController import *
import sys
import os
import psutil
#from scipy import stats
#import matplotlib.pyplot as plt
process = psutil.Process(os.getpid())

def dist_infer_action(sess,frames,ops,phs):
    action = ops['action']
    x1 = phs['x1']
    a = sess.run([action],{x1: [frames]})
    return a

def send_action_to_game_controller(game,a):
    if (a == 0):
        game.move_mouse_up()
    elif (a == 1):
        game.move_mouse_down()
    elif (a == 2):
        game.move_mouse_left()
    elif (a == 3):
        game.move_mouse_right()
    else:
        x = 1
    
    r = game.reward
    frames,bval = get_4_frames(game)
    return r,frames,bval

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
    exp.append(seq)
    return

def get_4_frames(game):
    imgs = np.concatenate((take_shot(game),
                         take_shot(game),
                         take_shot(game),
                         take_shot(game)),axis=2)
    bval = game.get_screen_number2(np.array(imgs))
    rtnbool = True if bval else False
    return imgs,rtnbool

def dist_add_to_queue(sess,batch_size,ops,phs):
    enqueue_op = ops['enqueue_op']
    s_img1 = phs['s_img1']
    s_a = phs['s_a']
    s_r = phs['s_r']
    s_img2 = phs['s_img2']

    seq_n = random_minibatch_sample(batch_size)
    sess.run([enqueue_op],{s_img1: seq_n[0],s_a: seq_n[1], s_r: seq_n[2],s_img2: seq_n[3]})


def dist_run(sess,game,greed,M,batch_size,ops,phs):
    global exp
    for i in range(0,M):
        if (i % 50 == 0):
            if (greed >= .2):
                greed = greed-.2
            else: 
                greed = .1
        wait_for(1)
        game.click_to_play()
        while game.get_screen_number(take_shot(game)):
            frames1,test = get_4_frames(game)
            if (not test):
                break  
            a = np.asarray(np.random.randint(0,5)).astype(np.uint8) if (np.random.random_sample(1) <= greed) else np.asarray(dist_infer_action(sess,frames1,ops,phs)).astype(np.float16)     
            r,frames2,test = send_action_to_game_controller(game,a)
            if (not test):
                break
            store_exp((frames1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),frames2))
            if (len(exp) > 10):
                dist_add_to_queue(sess,batch_size,ops,phs)
        game.release_click()
        wait_for(.3)
        sess.run([ops['uwb']])
        game.click_replay()
        print("Iteration: ",i)
    
    return

"""
def play_game(game,M):
    for i in range(0,M):
        wait_for(1)
        game.click_to_play()
        while game.get_screen_number2(take_shot(game)):
            frames,test = get_4_frames(game)
            if (not test):
                break
            a = infer_action([frames])
            r,frames,test = send_action_to_game_controller(game,a)
            if (not test):
                break
        game.release_click()
        wait_for(.3)
        game.click_replay()
        print("Play Iteration: ",i)
"""

exp = []
