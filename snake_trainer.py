import time
from timeit import timeit,Timer
import numpy as np
from snake_con import *
import sys
import os
import matplotlib.pyplot as plt
from threading import Thread,current_thread

def dist_infer_action(sess,frames,ops,phs):
    action = ops['action']
    x1 = phs['x1']
    q = ops['q_vals_pr']
    a,q = sess.run([action,q],{x1: [frames]})
    return a,q

def send_action_to_game_controller(game,frames1,a,reward):
    
    if (a == 0):
        game.move_up()
    elif (a == 1):
        game.move_down()
    elif (a == 2):
        game.move_left()
    elif (a == 3):
        game.move_right()
    else:
        xxxx = 1
    

    try:
        alert = game.chrome.switch_to.alert
        alert.accept()
        game.stop_play = True
        r = -1
    except:
        if reward is game.reward:
            r = 0
        elif game.reward > reward:
            r = 1
        reward = game.reward
        pass

    frames,bval = get_4_frames(game)
    print("r: ",r)
    return frames,bval,r,reward

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
    if (process_frames > 100000):
        exp.pop(0)
    process_frames = process_frames+8
    exp.append(seq)    
    return

def get_greed(greed_frames,frames):
    if frames > greed_frames:
        return 0.1
    return (((.1-1)/greed_frames)*frames)+1

def get_4_frames(game):
    imgs = np.concatenate((take_shot(game),
                         take_shot(game),
                         take_shot(game),
                         take_shot(game)),axis=2)
    bval = game.stop_play
    #rtnbool = True if bval else False
    return imgs,bval

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
        if (i % 100 == 0):
            if (greed >= .1 ):
                greed = greed-.1
        wait_for(1)
        game.click_to_play()
        Thread(target=game.kill_highscore_alert,args=(current_thread(),)).start()
        while game.get_screen_number2(take_shot(game)):
            frames1,test = get_4_frames(game)
            if (not test):
                break  
            a,q = [np.asarray(np.random.randint(0,5)).astype(np.uint8),0] if (np.random.random_sample(1) <= greed) else np.asarray(dist_infer_action(sess,frames1,ops,phs)).astype(np.float16)
            frames2,test,r = send_action_to_game_controller(game,frames1,a,0)
            r = game.reward_1(frames1,a)
            if (not test):
                break
            store_exp((frames1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),frames2))
            if (len(exp) > 10000):
                dist_add_to_queue(sess,batch_size,ops,phs)
        game.release_click()
        wait_for(.3)
        sess.run([ops['uwb']])
        game.click_replay()
        wait_for(.3)
        print("Iteration: ",i)
        if (not game.get_screen_number2(take_shot(game))):
            return
    return

def frame_train_reward(sess,game,frame_limit,greed_frames,batch_size,ops,phs,gsheets):
    global process_frames
    
    gp = 0
    while (process_frames < frame_limit):
        greed = get_greed(greed_frames,process_frames)
        reward = 0
        wait_for(1)
        game.click_play()
        frames1,test = get_4_frames(game)
        while not game.stop_play:
            a,q = [np.asarray(np.random.randint(0,5)).astype(np.uint8),0] if (np.random.random_sample(1) <= greed) else np.asarray(dist_infer_action(sess,frames1,ops,phs)).astype(np.float16)
            frames2,stop_play,r,reward = send_action_to_game_controller(game,frames1,a,reward)
            store_exp((frames1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),frames2))
            if stop_play:
                break
            frames1 = frames2
            if (len(exp) > 10000):
                dist_add_to_queue(sess,batch_size,ops,phs)
        wait_for(.3)
        sess.run([ops['uwb']])
        wait_for(.3)
        game.reward = 0
        game.stop_play = False
        gp = gp+1
        if (gp % 50 == 0):
            print("Exp size: ", len(exp))
            print("Number process Frames: ",process_frames)
            print("greed: ",greed)
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


exp = []
process_frames = 0