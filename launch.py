import subprocess
import os


def kill_all(process):
    for i in process:
        if (i.poll != None):
            i.kill()
    return


#relevent directories
py_dir = r"C:\Users\Vishnu\Envs\RL\Scripts\python.exe"
dist_tf_dir = r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\dist_tf.py"

#Popen arguments
ps_args = [py_dir,dist_tf_dir,"ps","0"]
worker0 = [py_dir,dist_tf_dir,"worker","0"]
worker1 = [py_dir,dist_tf_dir,"worker","1"]


#Start Process
ps_proc = subprocess.Popen(ps_args)
w0_proc = subprocess.Popen(worker0,creationflags=subprocess.CREATE_NEW_CONSOLE)
w1_proc = subprocess.Popen(worker1,creationflags=subprocess.CREATE_NEW_CONSOLE)

proc_arr = [ps_proc,w0_proc,w1_proc]

run = True
while(run):
    for i in proc_arr:
        if i.poll() is 0:
            run = False
kill_all(proc_arr)
exit()