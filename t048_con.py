


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import win32api,win32con
import numpy as np 
import mss
import mss.tools
import time
from PIL import Image



def wait_for(sec):
    t = time.time()+sec
    while(True):
        if (time.time() > t):
            break



class t048:

    def __init__(self,id):
        #self.url = r"https://www.google.com/?gws_rd=ssl"
        self.url = r"chrome-extension://jfnbjbahocpfkbbadndnocljpjpccggf/index.html"
        self.chrome_path = r'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
        if id==1:
            self.ext_path = r"C:\Users\devar\AppData\Local\Google\Chrome\User Data\Default\Extensions\jfnbjbahocpfkbbadndnocljpjpccggf\1.5_0"
            self.processing_crop = {'left':310,
                                    'top': 500,
                                    'width': 546,
                                    'height': 546}
        elif id==2:
            #self.ext_path = r"C:\Users\Vishnu\AppData\Local\Google\Chrome\User Data\Default\Extensions\kcnffeedfpaijglkkpplkbdpchgjbako\1.2_0"
            self.ext_path = r"C:\Users\Vishnu\AppData\Local\Google\Chrome\User Data\Default\Extensions\jfnbjbahocpfkbbadndnocljpjpccggf\1.5_0"
            self.processing_crop = {'left':155,
                                    'top': 250,
                                    'width': 273,
                                    'height': 273}
        
        self._launch_game()
        self.sct = mss.mss()
        self.stop_play = False
        self.reward = 0

        self.up = Keys.ARROW_UP
        self.down = Keys.ARROW_DOWN
        self.left = Keys.ARROW_LEFT
        self.right = Keys.ARROW_RIGHT

    def _launch_game(self):
        """
            Launches webbrowser with snake game
        """
        
        self.chrome_options = Options()
        self.chrome_options.add_argument("--load-extension="+self.ext_path)
        self.chrome_options.add_argument("--ignore-certificate-errors")
        self.chrome_options.add_argument("--ignore-ssl-errors")
        self.chrome = webdriver.Chrome(chrome_options=self.chrome_options)

        chrome = self.chrome
        chrome.get(self.url)
        chrome.set_window_size(500,600)
        chrome.set_window_position(50,50)

        self.new_game_button = chrome.find_element_by_xpath("/html/body/div[2]/div[2]/a")
        #Check if retry button visible
        self.retry_buttom = chrome.find_element_by_xpath("/html/body/div[2]/div[3]/div[1]/div/a[2]")
        self.score = chrome.find_element_by_xpath("/html/body/div[2]/div[1]/div/div[1]")
        #self.tile_inner_elems = chrome.find_elements_by_class_name("tile-inner")

        chrome.execute_script("window.scrollTo(0,225)")

    def click_at_location(self,cord):
        x = cord[0]
        y = cord[1]

        win32api.SetCursorPos(cord)
        wait_for(.2)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    
    def send_key(self,key):
        win32api.keybd_event(key,0,1,0)
        wait_for(.1)
        win32api.keybd_event(key,0,2,0)

    def move(self,ks):
        try:
            ActionChains(self.chrome).send_keys(ks).perform()
            #print("reward: ",self.reward)
        except:
            print("pass")
            pass
        self.reward = self.get_reward()
        
    def get_reward(self):
        txt = self.score.text
        tile_inner_elems = self.chrome.find_elements_by_class_name("tile-inner")
        r = 0
        for i in tile_inner_elems:
            if i.text.isdigit():
                if int(i.text) == 8:
                    r = 1
        return r
        #while "+" in txt:
        #    txt = self.score.text
        
        #return int(txt)

    

def take_shot(game):
    img = game.sct.grab(game.processing_crop)
    img = Image.fromarray(np.array(img)[:,:,1]).resize((100,100))
    #print("shape: ",np.array(img).shape)
    #Image.fromarray(np.array(img)).save("imgs/test.png")
    img = np.expand_dims(np.array(img),axis=2)
    return img

