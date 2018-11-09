


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import io
import win32api,win32con
import numpy as np 
import mss
import mss.tools
import time
from PIL import Image

import psutil
from pywinauto import Application
from pywinauto.win32functions import SetForegroundWindow

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


        #vars for reward2
        self.count_64 = 0
        self.count_128 = 0
        self.count_256 = 0
        self.count_512 = 0
        self.count_1024 = 0
        self.count_2048 = 0

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
        
        self.chrome_pid = self.chrome.service.process.pid
        print("###################",self.chrome_pid)
        self.app = Application().connect(process = self.chrome_pid)


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

        except:
            print("pass")
            pass
        
    def get_reward1(self):
        """
        Function returns reward. Reward is 1 when 512 tile shows up
        """
        txt = self.score.text
        tile_inner_elems = self.chrome.find_elements_by_class_name("tile-inner")
        r = 0
        for i in tile_inner_elems:
            if i.text.isdigit():
                if int(i.text) == 512:
                    r = 1
        return r

    def get_reward2(self):
        """
        Function incrementally assigns reward in the following fashion.

        First 64-> reward = 1
        Second 64-> reward = 1
        ....
        First 128-> reward = 2
        Second 128-> reward = 2

        """
        tile_inner_elems = self.chrome.find_elements_by_class_name("tile-inner")
        
        count_64 = 0
        count_128 = 0
        count_256 = 0
        count_512 = 0
        count_1024 = 0
        count_2048 = 0
        for i in tile_inner_elems:
            txt = i.text
            if txt.isdigit():
                i_txt = int(txt)
                if i_txt == 64:
                    count_64 += 1
                elif i_txt == 128:
                    count_128 += 1
                elif i_txt == 256:
                    count_256 += 1
                elif i_txt == 512:
                    count_521 += 1
                elif i_txt == 1024:
                    count_1024 += 1
                elif i_txt == 2048:
                    count_2048 += 1
        
        r = 0
        if (count_64 > self.count_64):
            r += count_64-self.count_64
        self.count_64 = count_64
        if (count_128 > self.count_128):
            r += 2*(count_128-self.count_128)
        self.count_128 = count_128
        if (count_256 > self.count_256):
            r += 3*(count_256-self.count_256)
        self.count_256 = count_256
        if (count_512 > self.count_512):
            r += 4*(count_512-self.count_512)
        self.count_512 = count_512
        if (count_1024 > self.count_1024):
            r += 5*(count_1024-self.count_1024)
        self.count_1024 = count_1024
        if (count_2048 > self.count_2048):
            r += 6*(count_2048-self.count_2048)
        self.count_2048 = count_2048
        return r

    def take_shot(self):
        """
        Takes screenshot using selenium

        """

        chrome = self.chrome
        game_div = chrome.find_elements_by_class_name("grid-container")
        game_div = game_div[0]
        #if div exists
        if game_div is not None:
            #Fild the location and size
            game_div_loc = game_div.location_once_scrolled_into_view
            game_div_size = game_div.size 

            crop_points = [game_div_loc['x'],game_div_loc['y'],
                            game_div_loc['x']+game_div_size['width'],
                            game_div_loc['y']+game_div_size['height']]

            #Take shot of entier screen
            
            png = chrome.get_screenshot_as_png()
            img_bin = io.BytesIO(png)
            with Image.open(img_bin).convert('L') as img:
                crp_img =  img.crop(crop_points).resize((100,100))
            img = np.expand_dims(np.array(crp_img),axis=2)
        return img

    def move_win_pos(self):
        if self.chrome.get_window_position() == {'x': -500, 'y':50}:
           self.chrome.set_window_position(50,50)
        else:
           self.chrome.set_window_position(-500,50)

          

    

def take_shodt(game):
    img = game.sct.grab(game.processing_crop)
    img = Image.fromarray(np.array(img)[:,:,1]).resize((100,100))
    #print("shape: ",np.array(img).shape)
    #Image.fromarray(np.array(img)).save("imgs/test.png")
    img = np.expand_dims(np.array(img),axis=2)
    return img

    