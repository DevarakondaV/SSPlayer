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

def wait_for(sec):
	t = time.time()+sec
	while(True):
		if (time.time()>t):
			break

class snake:

    def __init__(self,id):
        #self.chrome_path = r'open -a /Applications/Google\ Chrome.app %s'
        self.url = "chrome-extension://kcnffeedfpaijglkkpplkbdpchgjbako/snake.html"
        self.chrome_path = r'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
        if id==1:
            self.ext_path = r"C:\Users\devar\AppData\Local\Google\Chrome\User Data\Default\Extensions\kcnffeedfpaijglkkpplkbdpchgjbako\1.2_0"
        elif id==2:
            self.ext_path = r"C:\Users\Vishnu\AppData\Local\Google\Chrome\User Data\Default\Extensions\kcnffeedfpaijglkkpplkbdpchgjbako\1.2_0"
        self._launch_game()
        self.sct = mss.mss()
        self.stop_play = False
        self.reward = 0

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
        chrome.set_window_size(500,500)
        chrome.set_window_position(50,50)


        #Key elements of game
        self.start_button = chrome.find_element_by_id("iniciar")
        self.score = chrome.find_element_by_id("placar")
        self.layout = chrome.find_element_by_id("layout")
        self.play_grid = chrome.find_element_by_id('jogo')

        #crop screen
        self.processing_crop = {'left':133,
                                'top': 432,
                                'width': 728,
                                'height': 447}
    
    def move_up(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.ARROW_UP).perform()
            self.reward = self.get_score()
        except:
            pass


    def move_down(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.ARROW_DOWN).perform()
            self.reward = self.get_score()
        except:
            pass

    def move_left(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.ARROW_LEFT).perform()
            self.reward = self.get_score()
        except:
            pass

    def move_right(self):
        try:
            ActionChains(self.chrome).send_keys(Keys.ARROW_RIGHT).perform()
            self.reward = self.get_score()
        except:
            pass
    
    def get_score(self):
        if (self.score.text is ''):
            return 0
        else:
            return int(self.score.text)

    def click_play(self):
        self.start_button.click()

    def kill_alert(self):
        try:
            WebDriverWait(self.chrome,15).until(EC.alert_is_present(),
                                                'Timed out waiting for PA creation ' +
                                                'confirmation popup to appear.')
            alert = self.chrome.switch_to.alert
            alert.accept()
            print("alert accepted")
        finally:
            print("no alert")

    def kill_highscore_alert(self,main_thread):
        while main_thread.is_alive(): #not self.stop_play:
            try:
                alert = self.chrome.switch_to.alert
                alert.accept()
                self.stop_play = True
            except:
                continue
        #self.reward = 0


def take_shot(game):
    """
        Takes single shot of play grid
    """
    img = game.sct.grab(game.processing_crop)
    img = Image.fromarray(np.array(img)[:,:,1]).resize((142,110))#.resize((84,110))
    img = np.expand_dims(np.array(img),axis=2)
    return img






