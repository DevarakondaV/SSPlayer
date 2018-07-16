from pywinauto import *
import image
import time
from PIL import Image
import numpy as np
import mss
import mss.tools
import win32api, win32con
from scipy import stats

def wait_for(sec):
	t = time.time()+sec
	while(True):
		if (time.time()>t):
			break

class SSPlayer:
	
	__slots__ = ['l_o_d',
				'app',
				'reward',
				'window_width',
				'window_height',
				'window_screen_loc',
				'mainscene',
				'playscene',
				'processing_crop',
				'play_click_loc',
				'playing_click_loc',
				'replay_click_loc',
				'sct']

	def __init__(self,dir,l_or_d):
		self.l_o_d = l_or_d
		self.app = self.launch_app(dir)
		self.reward = 1
		self.sct = mss.mss()
	
		
	def launch_app(self,dir):
		app = application.Application().start(dir)
		app.ShapeScape.Wait('visible',timeout=20)
		app.ShapeScape.SetFocus()
		app.ShapeScape.MoveWindow(x=50,y=50)
		
		self.window_width = app.ShapeScape.client_rect().width()
		self.window_height = app.ShapeScape.client_rect().height()
		self.window_screen_loc = app.ShapeScape.Rectangle()

		
		if (self.l_o_d == 1):
			self.mainscene = np.array(Image.open(r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\testimg\laptop\mainscene.png"))
			self.playscene = np.array(Image.open(r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\testimg\laptop\playscene.png"))
			self.processing_crop = {'top': self.window_screen_loc.top+31,
									'left': self.window_screen_loc.left+8,
									'width': self.window_width,
									'height': self.window_height}
			self.play_click_loc = (233,294)
			self.playing_click_loc = (209,529)
			self.replay_click_loc = (148,498)

		
		else:
			self.mainscene = np.array(Image.open(r"C:\Users\devar\Documents\EngProj\SSPlayer\testimg\desktop\mainscene.png"))
			self.playscene = np.array(Image.open(r"C:\Users\devar\Documents\EngProj\SSPlayer\testimg\desktop\playscene.png"))
			self.processing_crop = {'top': self.window_screen_loc.top+58,
									'left': self.window_screen_loc.left+13,
									'width': self.window_width,
									'height': self.window_height}
			self.play_click_loc = (243,319)
			self.playing_click_loc = (209,529)
			self.replay_click_loc = (148,518)
		
		return app
	
	def crop_image_for_test(self,img):
		return img[:,0:8,10:27]
	
	def kill(self):
		self.app.kill()
		
	def click_play(self):
		win32api.SetCursorPos(self.play_click_loc)
		wait_for(.2)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.play_click_loc[0],self.play_click_loc[1],0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,self.play_click_loc[0],self.play_click_loc[1],0,0)

	def click_to_play(self):
		win32api.SetCursorPos(self.playing_click_loc)
		wait_for(.2)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.playing_click_loc[0],self.playing_click_loc[1],0,0)

	def click_replay(self):
		win32api.SetCursorPos(self.replay_click_loc)
		wait_for(.2)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.replay_click_loc[0],self.replay_click_loc[1],0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,self.replay_click_loc[0],self.replay_click_loc[1],0,0)
	
	def release_click(self):
		x,y = win32api.GetCursorPos()
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

	def move_mouse(self,cord):
		x,y = win32api.GetCursorPos()
		dx = cord[0]-x
		dy = cord[1]-y
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx,dy,0,0)
		
	def move_mouse_right(self):
		x,y = win32api.GetCursorPos()
		if (x < (self.processing_crop['left']+self.processing_crop['width'])):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,4,0,0,0)
		else:
			self.reward = self.reward-.5
	
	def move_mouse_left(self):
		x,y = win32api.GetCursorPos()
		if (x > self.processing_crop['left']):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,-4,0,0,0)
		else:
			self.reward = self.reward-.5

	def move_mouse_up(self):
		x,y = win32api.GetCursorPos()
		if (y > self.processing_crop['top']):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,-4,0,0)
		else:
			self.reward = self.reward-.5
			
	def move_mouse_down(self):
		x,y = win32api.GetCursorPos()
		if (y < (self.processing_crop['top']+self.processing_crop['height']-5)):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,4,0,0)
		else:
			self.reward = self.reward-.5
	
	#returns the screen number
	#1 for main, 2 for play, 3 for end
	def get_screen_number(self,img):
		img = self.crop_image_for_test(img)
		if (np.array_equal(img,self.mainscene)):
			return 1
		elif (np.array_equal(img,self.playscene)):
			self.reward = self.reward
			return 2
		else:
			self.reward = 0
			return 3
			
	
	def get_screen_number2(self,img):
		img = self.crop_image_for_test(img)
		check_m = np.array_equiv(self.playscene,img)
		return check_m
			
	
	

def img_normalize(img):
	print("Img Shape: ",img.shape)
	i_max2 = np.amax(img,axis=0)
	print("Img Max: ",i_max2)
	return img
	
def img_standardize(img):
	return stats.zscore(img,axis=0)

def take_shot(game):
		img = game.sct.grab(game.processing_crop)
		#img = misc.imresize(np.array(img)[:,:,1],(110,84))
		img = Image.fromarray(np.array(img)[:,:,1]).resize((84,110))
		return np.expand_dims(np.array(img),axis=0)
		

