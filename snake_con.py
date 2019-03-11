from pywinauto import Application,Desktop
import image
import time
from timeit import Timer,timeit
from PIL import Image
import numpy as np
import mss
import mss.tools
import math
import win32api,win32con
from scipy import stats

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
        elif id==2:
            self.ext_path = r"C:\Users\Vishnu\AppData\Local\Google\Chrome\User Data\Default\Extensions\kcnffeedfpaijglkkpplkbdpchgjbako\1.2_0"
        self._launch_game()
        self.sct = mss.mss()
        self.stop_play = False
        self.reward = 0
        self.prv_score = 0
        
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
        self.spy = chrome.find_element_by_id("spx")
        self.fx = chrome.find_element_by_id("fx")
        self.fy = chrome.find_element_by_id("fy")

        self.canvas = chrome.find_element_by_id("snake-game")
        self.game_container = chrome.find_element_by_class_name("container")
        self.chrome.execute_script("arguments[0].setAttribute('width','300')", self.canvas)
        self.chrome.execute_script("arguments[0].setAttribute('height','300')", self.canvas)
        self.chrome.execute_script("arguments[0].setAttribute('style','width: 300px;')", self.game_container)
        self.chrome.execute_script("arguments[0].setAttribute('class','')", self.game_container)
        

        # #crop screen
        # self.processing_crop = {'left':127,
        #                         'top': 465,
        #                         'width': 1025,
        #                         'height': 915}

        self.processing_crop = {'left':128,
                                'top': 475,
                                'width': 610,
                                'height': 610}
        
        self.up = Keys.ARROW_UP
        self.down = Keys.ARROW_DOWN
        self.left = Keys.ARROW_LEFT
        self.right = Keys.ARROW_RIGHT
    
    def move(self,ks):
        self.stop_play = True if self.start_button.get_attribute("style") == "display: block;" else False            
        if ks == -1:
            self.reward = self.get_score()
            return        
        
        try:
            if not self.stop_play :
                ActionChains(self.chrome).send_keys(ks).perform()
                time.sleep(.1)
                self.stop_play = True if self.start_button.get_attribute("style") == "display: block;" else False                    
                self.reward = -1 if self.stop_play == True else self.get_score()
            else :
                self.prv_score = 0
        except:
            print("pass")
            pass

    def get_reward(self):
        return self.reward
    
    def get_score(self):
        score = int(self.score.text)
        if (self.prv_score < score):
            self.prv_score = score
            return 1
        else :
            return 0

    def get_current_dist(self):
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
        elif (prv_dist >= self.get_current_dist()) :
            rtn_val = .5
        else: 
            rtn_val = -.5
        return rtn_val   

    def click_play(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.SPACE).perform()
        except:
            print("Cannot start game")
        self.get_current_dist()


def take_shot(game):
    """
        Takes single shot of play grid
    """
    img = game.sct.grab(game.processing_crop)
    img = Image.fromarray(np.array(img)[:,:,1]).resize((100,100),resample=Image.LANCZOS)#.resize((84,110))
    #Thread(target=save_img,args=(img,0)).start()
    img = np.expand_dims(np.array(img),axis=2)
    return img

def save_img(img,a):
    img = Image.fromarray(img[:,:,0])
    if a == 2:
        adir = "str"
    elif a == 0:
        adir = "left"
    else:
        adir = "right"
    fname = str(uuid.uuid4())+"__"+adir
    path = r"E:\vishnu\SSPlayer\imgs\\"+fname+".jpg"
    img.save(path,"JPEG")


