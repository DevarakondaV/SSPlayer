
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
import matplotlib.pyplot as plt
process = psutil.Process(os.getpid())

def dist_infer_action(sess,frames,ops,phs):
    action = ops['action']
    x1 = phs['x1']
    q = ops['q_vals_pr']
    a,q = sess.run([action,q],{x1: [frames]})
    return a,q

def send_action_to_game_controller(game,img,a):
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
    
    #game.calculate_reward(img,a)
    #r = game.reward
    frames,bval = get_4_frames(game)
    #return r,frames,bval
    return frames,bval

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
    if (process_frames > 1e6):
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
    bval = game.get_screen_number2(np.array(imgs))
    rtnbool = True if bval else False
    return imgs,rtnbool

def add_rewards_and_store(game,seq,survival_time):
    timestamps = np.linspace(0,survival_time,num=len(seq),endpoint=True).astype(np.float16)
    for i in range(0,len(seq)):
        seq[i][2] = timestamps[len(seq)-1-i]

    for i in seq:
        store_exp(game.reward_3(i))

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
        while game.get_screen_number2(take_shot(game)):
            frames1,test = get_4_frames(game)
            if (not test):
                break  
            a,q = [np.asarray(np.random.randint(0,5)).astype(np.uint8),0] if (np.random.random_sample(1) <= greed) else np.asarray(dist_infer_action(sess,frames1,ops,phs)).astype(np.float16)
            frames2,test = send_action_to_game_controller(game,frames1,a)
            r = game.reward_1(frames1,a)
            if (not test):
                break
            store_exp((frames1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),frames2))
            if (len(exp) > batch_size):
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

def frame_train_reward_1(sess,game,frame_limit,greed_frames,batch_size,ops,phs):
    global process_frames
    Qs = []
    while (process_frames < frame_limit):
        greed = get_greed(greed_frames,process_frames)
        q_max = 0
        wait_for(1)
        game.click_to_play()
        while game.get_screen_number2(take_shot(game)):
            frames1,test = get_4_frames(game)
            if not test:
                break
            a,q = [np.asarray(np.random.randint(0,5)).astype(np.uint8),0] if (np.random.random_sample(1) <= greed) else np.asarray(dist_infer_action(sess,frames1,ops,phs)).astype(np.float16)
            print(a)
            frames2,test = send_action_to_game_controller(game,frames1,a)
            r = game.reward_1(frames1,a)
            if (not test):
                break
            store_exp((frames1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),frames2))
            if (len(exp) > batch_size):
                dist_add_to_queue(sess,batch_size,ops,phs)
            if (process_frames > greed_frames):
                q_max = q_max+q
        game.release_click()
        wait_for(.3)
        sess.run([ops['uwb']])
        Qs.append(q_max)
        game.click_replay()
        wait_for(.3)
        if not game.get_screen_number2(take_shot(game)):
            break
    iters = np.arange(0,len(Qs))
    plt.plot(iters,Qs)
    plt.show()
    return

def frame_train_reward_2(sess,game,frame_limit,greed_frames,batch_size,ops,phs):
    global process_frames,exp
    Qs = []
    runs = 0
    while(process_frames < frame_limit):
        runs = runs+1
        greed = get_greed(greed_frames,process_frames)
        wait_for(1)
        game.click_to_play()
        seq = []
        q_max= 0
        t_init = time.time()
        while game.get_screen_number2(take_shot(game)):
            frames1,test = get_4_frames(game)
            if not test:
                break
            a,q = [np.asarray(np.random.randint(0,5)).astype(np.uint8),0] if (np.random.random_sample(1) <= greed) else np.asarray(dist_infer_action(sess,frames1,ops,phs)).astype(np.float16)
            frames2,test = send_action_to_game_controller(game,frames1,a)
            seq.append([frames1,np.array(a).astype(np.uint8),np.array(0).astype(np.float16),frames2])
            if (not test):
                break
            if (len(exp) > batch_size):
                dist_add_to_queue(sess,batch_size,ops,phs)
            if process_frames > greed_frames:
                q_max = q_max+q
        survival_time = time.time()-t_init
        game.release_click()
        wait_for(.3)
        add_rewards_and_store(game,seq,survival_time)
        Qs.append(q_max)
        sess.run([ops['uwb']])
        game.click_replay()
        wait_for(.3)
        if not game.get_screen_number2(take_shot(game)):
            break
        if (runs % 100) is 0:
            print("Exp size: ", len(exp))
            print("Number process Frames: ",process_frames)
            print("greed: ",greed)
    iters = np.arange(0,len(Qs))
    plt.plot(iters,Qs)
    plt.show()
    return


def dist_play(sess,game,M,ops,phs):
    for i in range(0,M):
        wait_for(1)
        game.click_to_play()
        while game.get_screen_number2(take_shot(game)):
            frames,test = get_4_frames(game)
            if (not test):
                break
            a,q= np.asarray(dist_infer_action(sess,frames,ops,phs)).astype(np.float16)
            #a,q = dist_infer_action(sess,frames,ops,phs)
            #a = np.argmax(a[0])
            print(a)
            frames,test = send_action_to_game_controller(game,frames,a)
            if (not test):
                break
        game.release_click()
        wait_for(.3)
        game.click_replay()
        print("Play Iteration: ",i)


exp = []
process_frames = 0
