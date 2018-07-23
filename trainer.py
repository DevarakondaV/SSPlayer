
# coding: utf-8

# In[1]:


#imports
from cnn import *
import time
from timeit import timeit,Timer
from GameController import *
#import itertools
import sys
#import objgraph
import os
import psutil
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import clear_output
process = psutil.Process(os.getpid())


# In[2]:


#Modified Functions
#infer_action
#get_seq_y
#train_network

#Relevant Functions
def infer_action(seq):
    action = tf.get_default_graph().get_tensor_by_name("action:0")
    q_compute = tf.get_default_graph().get_tensor_by_name("Qnext_val:0")
    pv = tf.get_default_graph().get_tensor_by_name("print:0")
    #outputs = tf.get_default_graph().get_tensor_by_name("dense_layers/FC2/act2/Maximum:0")
    x1,trai_bool,o= get_place_holders()
    #return sess.run([action],{x1: seq,x2: np.random.rand(1,110,84,4),next_state_bool: False})
    a,q,pr = sess.run([action,q_compute,pv],{x1: seq})
    #print(pr)
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
        # Do Nothing
    
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
    
    """
    print("shape img1: ",np.shape(img1s[0]))
    print("shape a: ",np.shape(a))
    print("shape r: ",np.shape(r))
    print("shape img2: ",np.shape(img2s[0]))
    """
    return (img1s,a,r,img2s)
    
def get_seq_y(seq,batch_size,gamma):
    q_compute = tf.get_default_graph().get_tensor_by_name("Qnext_val:0")
    x1,y,next_state_bool,Qnext = get_place_holders()
    #1.0/256
    
    imgs_2 = [((1.0)*np.array(seq[3,i])).astype(np.float16) for i in range(0,batch_size)]
    imgs_1 = [((1.0)*np.array(seq[0,i])).astype(np.float16) for i in range(0,batch_size)]
    
    q_vals = sess.run([q_compute],{x1: imgs_2})
    r_vals = [seq[2,i] for i in range(0,batch_size)]
    
    q_vals = np.array(q_vals)
    r_vals = np.array(r_vals)
    y = (r_vals+(gamma*q_vals)).reshape(len(r_vals),1)
    return y,np.squeeze(np.array(imgs_1))
    
def store_exp(seq):
    global exp
    exp.append(seq)
    return

def train_network(batch_size,q):
    #Required tensorflow variables and operations
    #global writer,summ,it
    #x1,y,next_state_bool,Qnext = get_place_holders()
    #train = tf.get_default_graph().get_operation_by_name("train/trainer")
    #y_vals,images = get_seq_y(seq,batch_size,gamma)
    #images = np.squeeze(images)
    #t,s = sess.run([train,summ],{x1: images, Qnext: y_vals})
    #writer.add_summary(s,it)
    #it = it+1
    
    #seq = tf.get_default_graph().get_tensor_by_name("train_place_holder/seq:0")
    #en_q = tf.get_default_graph().get_operation_by_name("TrainQueue/tq")
    seq_n = random_minibatch_sample(batch_size)
    q.enqueue(seq_n).run()
    #tf.Print(q.size(),[q.size()],"Q size: ").eval()
    return
    
def get_q():
    q = tf.FIFOQueue(capacity=25,
                    dtypes= (tf.uint8,tf.uint8,tf.float16,tf.uint8),
                    shapes= (tf.TensorShape([batch_size,110,84,4]),
                            tf.TensorShape([batch_size,1]),
                            tf.TensorShape([batch_size,1]),
                            tf.TensorShape([batch_size,110,84,4])),name="tq",shared_name="TrainQueue/train_queue")
    return q

def get_4_frames(game):
    imgs = np.concatenate((take_shot(game),
                         take_shot(game),
                         take_shot(game),
                         take_shot(game)),axis=2)
    bval = game.get_screen_number2(np.array(imgs))
    rtnbool = True if bval else False
    return imgs,rtnbool

def add_to_queue(frames):
    global writer,summ,i
    q = tf.get_default_graph().get_operation_by_name("FIFOQueue/Train_Queue")
    q.enqueue(frames)


# In[3]:


#Train/Play Fucnction
def run(game,greed,M,batch_size,gamma,q):
    global exp
    t_a = []
    for i in range(0,M):  #New play
        clear_output(wait=True)
        if (i % 50 == 0):
            if (greed >= .2):
                greed = greed-.2
            else:
                greed = .1
        wait_for(1)
        p = 0
        t = time.time()
        game.click_to_play()
        while game.get_screen_number(take_shot(game)): #for j in range(0,T): #While play active
            frames1,test = get_4_frames(game)
            if (not test):
                break
            a = np.asarray(np.random.randint(0,5)).astype(np.uint8) if (np.random.random_sample(1) <= greed) else np.asarray(infer_action([frames1])).astype(np.float16)
            r,frames2,test = send_action_to_game_controller(game,a)
            if (not test):
                break
            store_exp((frames1,np.array(a).astype(np.uint8),np.array(r).astype(np.float16),frames2))
            if (len(exp) > 10):
                #add_to_train_queue(batch_size)
                train_network(batch_size,q)
        game.release_click()
        t_a.append(time.time()-t)
        wait_for(.3)
        game.click_replay()
        print("Iteration: ",i)
        #print("Greed: ",greed)
        #print("size: ",len(exp))
        #print("size bytes: ",sys.getsizeof(exp))
        #print("mem: ",(process.memory_info().rss)/1e6)
    return t_a

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


# In[4]:


conv_k_size = [8,4]
conv_stride = [4,2]
conv = [0,16,32]
fclyr = [0,125,5]
conv_count = len(conv)
fc_count = len(fclyr)
learning_rate = 1e-4
gamma = np.array([.9]).astype(np.float16)
batch_size = 10
LOGDIR = r"c:\Users\devar\Documents\EngProj\SSPlayer\log"
writer,summ,q = create_model(learning_rate,gamma,batch_size,conv_count,fc_count,conv,fclyr,conv_k_size,conv_stride,LOGDIR)
wait_for(5)

# In[ ]:


it = 1
exp = []
app_dir = r"C:\Users\devar\Documents\EngProj\SSPlayer\Release.win32\ShapeScape.exe"
if it:
    sess_param = tf.ConfigProto()
    sess_param.gpu_options.allow_growth = True
    #if __name__ == "__main__":
    with tf.Session("grpc://localhost:52003",config=sess_param) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        game = SSPlayer(app_dir,2)
        wait_for(1)
        print("Game Byte Size: ",sys.getsizeof(game))
        game.click_play()
        t_a = run(game,.9,10,10,.9,q)
        #p_i = np.arange(0,2,1)
        #play_game(game,15)
        #sess.close()
        game.kill()
#[128,192,192,264,264]


# In[ ]:

"""
#Other Functions
def save_imgs(exp):
    img_1 = [(250*((1.0/256)*np.array(exp[i][0]))).astype(np.uint8) for i in range(0,len(exp))]
    img_2 = [(250*((1.0/256)*np.array(exp[i][3]))).astype(np.uint8) for i in range(0,len(exp))]
    
    print(np.shape(img_1))
    print(np.shape(img_2))
    
    imgs= []
    for i in img_1:
        imgs.append(i[:,:,0])
        imgs.append(i[:,:,1])
        imgs.append(i[:,:,2])
        imgs.append(i[:,:,3])
        
    
    for i in img_2:
        imgs.append(i[:,:,0])
        imgs.append(i[:,:,1])
        imgs.append(i[:,:,2])
        imgs.append(i[:,:,3])
        
    
    print(np.shape(imgs))
    print(sys.getsizeof(imgs)/1000)
    for i in range(0,len(imgs)):
        Image.fromarray(imgs[i]).save(r"test\frame"+str(i)+".png")
    return
        
def test_seq(game,batch_size):
    seq = []
    for i in range(0,batch_size):
        frames,t = get_4_frames(game,game.processing_crop)
        seq.append(frames)
        seq.append(np.random.randint(0,4))
        seq.append(np.random.random_sample(1))
        frames,t = get_4_frames(game,game.processing_crop)
        seq.append(frames)
    return seq

def test():
    x1,y,next_state_bool,Qnext = get_place_holders()
    train = tf.get_default_graph().get_operation_by_name("train/trainer")
    dummy = np.random.rand(32,110,84,4)
    y_vals = np.random.rand(32,1)
    sess.run([train],{x1: dummy,Qnext: y_vals})
    return

if False:
    print(Timer(lambda: test()).timeit(number=1))
    
def store_test_data(length):
    global exp
    for i in range(0,length):
        img1 = np.random.rand(110,84,4).astype(np.float16)
        a = np.random.randint(0,4,1)
        r = np.random.rand(1).astype(np.float16)
        img2 = np.random.rand(110,84,4).astype(np.float16)
        exp.append((img1,a,r,img2))
    return

    
def test_new():
    train = tf.get_default_graph().get_operation_by_name("train/trainer")
    q_comp = tf.get_default_graph().get_tensor_by_name("Qnext_val:0")
    act = tf.get_default_graph().get_tensor_by_name("action:0")
    p = 0
    while True:
        try:
            x,q,a = sess.run([train,q_comp,act])
            p = p+1
            print("shape: ",np.shape(a))
        except tf.errors.OutOfRangeError:
            print("done")
            break
    print(p)
    return


# In[ ]:


exp = []
store_test_data(500)


# In[ ]:


print(Timer(lambda: test_new()).timeit(number=1))


# In[ ]:


x = 0
if x:
    game = SSPlayer(app_dir,2)
    wait_for(1)
    game.click_play()
    wait_for(1)
    play_game(game,15)
    game.kill()


# In[ ]:


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
if it:
    plt.plot(p_i, t_a,'o')
    plt.plot(p_i, smooth(t_a,19), 'r-', lw=2)
    plt.show()

"""