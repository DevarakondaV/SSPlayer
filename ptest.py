import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly() 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


import sys
import numpy as np
from cnn import *
#from gsheets import *
tf.enable_eager_execution()
tf.executing_eagerly() 
from trainer import *
import json
import os

with open("meta.json","r") as params_file:
    data = json.load(params_file)

LOGDIR = data["logdir"] if data["pc"] == 1 else data["plogdir"]
save_steps = data["save_steps"]

#network params
batch_size = data["batchsize"]
seq_len = data["seq_len"]
conv_k_size = [i["l"+str(idx+1)] for i,idx in zip(data["conv_k_size"],range(0,len(data["conv_k_size"])))]
conv_stride = [i["l"+str(idx+1)] for i,idx in zip(data["conv_stride"],range(0,len(data["conv_stride"])))]
conv = [i["l"+str(idx+1)] for i,idx in zip(data["conv"],range(0,len(data["conv"])))]
fclyr = [i["l"+str(idx+1)] for i,idx in zip(data["fclyr"],range(0,len(data["fclyr"])))]
learning_rate = data["learing_rate"]
gamma = np.array([data["gamma"]]).astype(np.float16)
load_weights = True if data["load_weights"] == 1 else  False


img_dir = r'C:\Users\devar\Documents\EngProj\snake_imgs\\'
img = sorted([img_dir+i for i in os.listdir(img_dir)])


net = pdqn(seq_len,conv,fclyr,conv_k_size,conv_stride,LOGDIR,gamma=gamma,batch_size=batch_size,learning_rate=learning_rate)
if (load_weights):
    T1 = np.zeros(shape=(1,84,84,seq_len))
    infer_dummy = [T1]
    train_dummy = [np.vstack([T1,T1]),
                    np.asarray([[1],[0]]),
                    np.asarray([.5,-1.0]).reshape((2,1)),
                    np.vstack([T1,T1])]
    net.infer(infer_dummy)
    net.train(inputs=train_dummy,IS_weights=np.ones(shape=(2,1)),r=[0,0])
    net.set_model_weights(r"C:\\Users\\devar\\Documents\\EngProj\\SSPlayer\\sweights\\b5weights2.hdf5")

imgs = []
for i in img:
    im = np.expand_dims(np.asarray(Image.open(i)),0)
    imgs.append(im)

imgs = np.stack(imgs,axis=3)
T1 = imgs[:,:,:,0:5]
T2 = imgs[:,:,:,5:10]
T3 = imgs[:,:,:,10:15]
T4 = imgs[:,:,:,15:20]
T1 = np.vstack([T1,T2])
T2 = np.vstack([T3,T4])
print(T1.shape,T2.shape)

print("HERE")
#rts = net.infer(inputs = [T1])
y = net.train([T1,np.asarray([[1],[0]]),np.asarray([.5,-1]).reshape((2,1)),T2],IS_weights=np.ones(shape=(2,1)),r=[0,0])
#print(net.layer_dict["Tar_dense1"])
#print(rts)
exit()
#net.set_model_weights(r"C:\Users\vishnu\Documents\EngProj\test\weights1.hdf5")


import numpy as np
t = np.memmap("tsne/tar",mode="r",dtype=np.float16)
t.shape
t = t.reshape(5000,512)
t[1].shape
v = np.memmap("tsne/v",mode="r",dtype=np.float16)
v.shape
a = np.memmap("tsne/a",mode="r",dtype=np.float16)
a.shape
a = a.astype(np.uint8)
a.dtype

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D,proj3d
import matplotlib.pyplot as plt
from matplotlib import colors
import pylab


tsne = TSNE(n_components=3,learning_rate=150,perplexity=30)
print("Fitting TSNE")
imgs_ts = tsne.fit_transform(t)

xxts = imgs_ts[:,0]
yyts = imgs_ts[:,1]
zzts = imgs_ts[:,2]
#xxts = [xxts[i] for i in range(100,5000,100)]
#yyts = [yyts[i] for i in range(100,5000,100)]
#v = [v[i] for i in range(100,5000,100)]
#ac = [a[i] for i in range(100,5000,100)]

gp1 = [[],[],[]]
gp2 = [[],[],[]]
gp3 = [[],[],[]]
for i in range(0,len(a)):
    if a[i] == 0:
        gp1[0].append(xxts[i])
        gp1[1].append(yyts[i])
        gp1[2].append(zzts[i])
    elif a[i] == 1:
        gp2[0].append(xxts[i])
        gp2[1].append(yyts[i])
        gp2[2].append(zzts[i])
    else:
        gp3[0].append(xxts[i])
        gp3[1].append(yyts[i])
        gp3[2].append(zzts[i])

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
sctr = ax.scatter(xxts,yyts,zzts,c=v,marker='.')#,zzts,c=a,marker='.')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.colorbar(sctr)
ax2 = fig.add_subplot(122,projection='3d')
ax2.plot(gp1[0],gp1[1],gp1[2],marker='.',color='r',label='Left',linestyle='')
ax2.plot(gp2[0],gp2[1],gp2[2],marker='.',color='g',label='Right',linestyle='')
ax2.plot(gp3[0],gp3[1],gp3[2],marker='.',color='y',label='Stright',linestyle='')
ax2.legend()
plt.show()


del a
del t
del v