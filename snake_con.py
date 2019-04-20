from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import time
from PIL import Image
import numpy as np
import mss
import mss.tools
from threading import Thread,current_thread
import psutil
from base64 import b64decode,decodestring
import numpy as np
from io import StringIO,BytesIO
import uuid
from math import log



class snake:

    def __init__(self,id):
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
        self.move_dir = 3
        self.snake_length = 5
        self.iter_frame = 0

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
        self.start_button = chrome.find_element_by_xpath("/html/body/div/div[2]")
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
    
    def kill(self):
        self.chrome.quit()
    
    def move(self,ks):
        try:
            ActionChains(self.chrome).send_keys(ks).perform()
            #time.sleep(.05)
        except:
            print("pass")
            pass
        
        self.stop_play = True if self.start_button.get_attribute("style") == "display: block;" else False
        self.iter_frame += 1
        del_r = self.calc_delr()
        self.reward = self.reward+del_r
        print("REWARD BEFORE CLIP: {} + {}".format(self.reward,del_r))
        self.reward = np.clip(self.reward,a_min=-1,a_max=1)
        return

    def perform_action(self,a):
        """
        Function performs an action and returns
        args:
            a: int. Action to send to the controller.
        returns:
            m_dir: String. Defines direction in which movement happened
        """

        #Telling self to perform an action based on the value of a
        # 0 left
        # 1 right
        # 2 do nothing
        move_dir = self.move_dir
        

        if a == 2:
            self.move(-1)
            m_dir = "Straight"
            return self.stop_play
        
        if (move_dir == 0):
            key = self.left if a == 0 else self.right
            m_dir = "left" if a == 0 else "right"            
            self.move_dir = 2 if a == 0 else 3 
        elif (move_dir == 1):
            key = self.right if a == 0 else self.left
            m_dir = "right" if a == 0 else "left"
            self.move_dir = 3 if a == 0 else 2
        elif (move_dir == 2):
            key = self.down if a == 0 else self.up
            m_dir = "down" if a == 0 else "up"
            self.move_dir = 1 if a == 0 else 0 
        else:
            key = self.up if a == 0 else self.down 
            m_dir = "up" if a == 0 else "down" 
            self.move_dir =  0 if a == 0 else 1 
        
        self.move(key)
        return self.stop_play


    def get_reward(self):
        return self.reward
    
    def get_score(self):
        score = int(self.score.text)
        if (self.prv_score < score):
            self.prv_score = score
            self.snake_length+=1
            self.iter_frame = 0
            return 1
        else :
            return 0

    def get_current_dist(self):
        dist = ((int(self.spx.text)-int(self.fx.text))**2+(int(self.spy.text)-int(self.fy.text))**2)**0.5
        self.prv_dist = dist
        return dist

    def calc_delr(self):
        #If scored reward is 1
        if (self.get_score()):
            return 1

        #if hit wall reward is -1
        if (self.stop_play):
            return -1
        
        #Else use distance
        if (self.iter_frame > self.calc_iter_timeout()):
            print("ITER FRAME {}: val: {}".format(self.iter_frame,self.snake_length))
            return -0.5/self.snake_length
        else:
            print("DISTANCE")
            num = self.snake_length+self.prv_dist
            den = self.snake_length+self.get_current_dist()
            return log(num/den,self.snake_length)
        
    def calc_iter_timeout(self):
        return (.7*self.snake_length)+10


    def click_play(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.SPACE).perform()

        except:
            print("Cannot start game")
        #self.get_dist()

        #Set Defaults
        self.stop_play = False
        self.reward = 0
        self.snake_length = 5
        self.get_current_dist()
        self.prv_score = 0
        self.move_dir = 3
        self.iter_frame = 0
    
    #Must be implemented
    def get_frame(self):
        return self.take_shot()

        
    def take_shot(self):
        """
        Function takes a shot of the game screen
        args:
        returns:
        img: Numpy Image array
        """


        img = self.sct.grab(self.processing_crop)
        img = Image.fromarray(np.array(img)[:, :, 1]).resize((84, 84))     
        img = np.expand_dims(np.array(img), axis=2)
        return img


def save_img(img):
    img = Image.fromarray(img[:, :, 0])
    
    #fname = str(uuid.uuid4())+"__"+adir
    path = r"C:\Users\Vishnu\Documents\EngProj\tflog\test.jpeg"
    img.save(path, "JPEG")


