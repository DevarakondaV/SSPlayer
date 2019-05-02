#from pywinauto import Application,Desktop
#import image
import time
#from timeit import Timer,timeit
from PIL import Image
import numpy as np
import mss
import mss.tools
from math import log
#import win32api,win32con
#from scipy import stats

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from threading import Thread,current_thread

import psutil
from base64 import b64decode,decodestring
import numpy as np
from io import StringIO,BytesIO

import uuid

def wait_for(sec):
	t = time.time()+sec
	while(True):
		if (time.time()>t):
			break

class snake:

    def __init__(self,id):
        #self.chrome_path = r'open -a /Applications/Google\ Chrome.app %s'
        self.url = "chrome-extension://gllcngkdngnfgilfmcbaanknakfgfepb/index.html"
        self.chrome_path = r'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
        if id==1:
            self.ext_path = r"C:\Users\devar\AppData\Local\Google\Chrome\User Data\Default\Extensions\gllcngkdngnfgilfmcbaanknakfgfepb\4.0_0"
            self.processing_crop = {'left':128,
                                    'top': 583,
                                    'width': 200,
                                    'height': 200}
        elif id==2:
            self.ext_path = r"C:\Users\vishnu\AppData\Local\Google\Chrome\User Data\Default\Extensions\gllcngkdngnfgilfmcbaanknakfgfepb\4.0_0"
            self.processing_crop = {'left': 80,
                                    'top': 362,
                                    'width': 127,
                                    'height': 123}
        self._launch_game()
        self.sct = mss.mss()
        self.stop_play = False
        self.reward = 0
        self.prv_score = 0
        self.prv_dist = 0
        self.s_len = 2
        self.p_steps = 0
        
        # 3 right
        # 2 left
        # 1 down
        # 0 up
        self.move_dir = 3

    def _launch_game(self):
        """
            Launches webbrowser with snake game
        """
        #Defining Options and launching browser
        self.chrome_options = Options()
        self.chrome_options.add_argument("--load-extension="+self.ext_path) 
        self.chrome = webdriver.Chrome(chrome_options=self.chrome_options)

        #Calling url, resizing, position
        chrome = self.chrome
        chrome.get(self.url)
        # chrome.set_window_size(550,725)
        chrome.set_window_size(300,575)
        chrome.set_window_position(50,50)
        #chrome.execute_script("window.scrollTo(0, 50)")

        #Key elements of game
        self.start_button = chrome.find_element_by_xpath("/html/body/div/div")
        self.score = chrome.find_element_by_xpath("/html/body/div/header/div/div[1]")

        #Snake and food position
        self.spx = chrome.find_element_by_id("spx")
        self.spy = chrome.find_element_by_id("spy")
        self.fx = chrome.find_element_by_id("fx")
        self.fy = chrome.find_element_by_id("fy")

        self.canvas = chrome.find_element_by_id("snake-game")
        self.game_container = chrome.find_element_by_class_name("container")
        self.chrome.execute_script("arguments[0].setAttribute('width','100')", self.canvas)
        self.chrome.execute_script("arguments[0].setAttribute('height','100')", self.canvas)
        self.chrome.execute_script("arguments[0].setAttribute('style','width: 100px;')", self.game_container)
        self.chrome.execute_script("arguments[0].setAttribute('class','')", self.game_container)
                
        self.up = Keys.ARROW_UP
        self.down = Keys.ARROW_DOWN
        self.left = Keys.ARROW_LEFT
        self.right = Keys.ARROW_RIGHT
    
    def move(self,ks):
        self.stop_play = True if self.start_button.get_attribute("style") == "display: block;" else False      
        
        try:
            if not self.stop_play :
                ActionChains(self.chrome).send_keys(ks).perform()
                #time.sleep(.0)
                self.stop_play = self.start_button.get_attribute("style") == "display: block;"
                if self.stop_play:
                    self.reward = -1
                    self.prv_score = 0
                    self.s_len = 2
                    self.p_steps = 0
                    self.prv_dist = 0
                    self.move_dir = 3
                else:
                    self.reward = self.get_score5()
                #self.reward = -1 if self.stop_play else self.get_score2()
            else :
                self.prv_score = 0
                self.s_len = 2
                self.p_steps = 0
        except:
            print("pass")
            pass

    def set_initial_dist(self):
        self.init_distance = self.get_dist()

    def get_reward(self):
        return self.reward
    
    def get_score(self):
        score = int(self.score.text)
        if (self.prv_score < score):
            self.prv_score = score
            self.set_initial_dist()
            return 1
        else :
            return 0

    def get_dist(self):
        dist = ((int(self.spx.text)-int(self.fx.text))**2+(int(self.spy.text)-int(self.fy.text))**2)**0.5
        self.prv_dist = dist
        return dist

    def get_score2(self):
        rtn_val = 0
        prv_dist = self.prv_dist
        score = int(self.score.text)
        if (self.prv_score < score):
            self.prv_score = score
            rtn_val = 1
        elif (prv_dist >= self.get_dist()) :
            rtn_val = .5
        else:
            rtn_val = -.5
        return rtn_val
    
    def get_score3(self):
        #gaus
        if (self.get_score()):
            return 10
        sx = int(self.spx.text)
        sy = int(self.spy.text)
        fx = int(self.fx.text)
        fy = int(self.fy.text)
        
        R = np.exp(-.5*((.45**2*np.power((sx-fx),2))+(.45**2*np.power((sy-fy),2))))
        #print("SX: {}\tSY: {}\tFX: {}\tFY: {}\tR: {}".format(sx,sy,fx,fy,R))
        if R < .4:
            return -1*R
        else:
            return R
        
    def get_score4(self):
        score = self.get_score()
        if score:
            return score
        
        diff = (self.init_distance - self.get_dist())/self.init_distance
        if (diff < 0 ):
            diff = 0
        return diff

    def get_score5(self):

            

        
        if (self.p_steps > self.get_steps()):
            delr = -0.5/self.s_len
            print(self.p_steps)
        else:
            num = self.s_len+self.prv_dist
            den = self.s_len+self.get_dist()
            delr = log(num/den)
            print("DELR",delr)

        if (self.get_score()):
            self.s_len +=2
            self.p_steps = 0
            delr = 1

        self.p_steps+=1
        return delr
        

    def click_play(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.SPACE).perform()
            self.get_dist()
        except:
            print("Cannot start game")
        #self.get_dist()

    def get_steps(self):
        return (.7*self.s_len)+10

        
    def take_shot(self):
        """
        Function takes a shot of the game screen
        args:
        returns:
        img: Numpy Image array
        """


        img = self.sct.grab(self.processing_crop)
        img = Image.fromarray(np.array(img)[:, :, 1]).resize((84, 84), resample=Image.LANCZOS)     
        img = np.expand_dims(np.array(img), axis=2)
        return img


def take_shot(game):
    """
        Takes single shot of play grid
    """
    img = game.sct.grab(game.processing_crop)
    img = Image.fromarray(np.array(img)[:, :, 1]).resize((84, 84), resample=Image.LANCZOS)#.resize((84, 110))
    #Thread(target=save_img, args=(img, 0)).start()
    # img.save("E:\\vishnu\\SSPlayer\\test.jpg")
    img = np.expand_dims(np.array(img), axis=2)
    return img

def save_img(img, a):
    img = Image.fromarray(img[:, :, 0])
    if a == 2:
        adir = "str"
    elif a == 0:
        adir = "left"
    else:
        adir = "right"
    fname = str(uuid.uuid4())+"__"+adir
    path = r"E:\vishnu\SSPlayer\imgs\\"+fname+".jpg"
    img.save(path, "JPEG")


