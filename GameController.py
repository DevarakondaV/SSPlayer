from pywinauto import *
import image
import time
from PIL import Image
import numpy as np
import mss
import mss.tools
import math
import win32api, win32con
from scipy import stats


def wait_for(sec):
	t = time.time()+sec
	while(True):
		if (time.time()>t):
			break


class SSPlayer:
	
	__slots__ = ['l_o_d','app','reward',
				'window_width','window_height',
				'window_screen_loc','mainscene',
				'playscene','processing_crop',
				'play_click_loc','playing_click_loc',
				'replay_click_loc','sct',
				'move_pixels','x_scale_factor',
				'y_scale_factor']

	def __init__(self,dir,l_or_d):
		self.l_o_d = l_or_d
		self.app = self.launch_app(dir)
		self.reward = 0
		self.move_pixels = 4
		self.x_scale_factor = 84/320
		self.y_scale_factor = 110/480
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
		#return img[:,0:8,10:27]
		return img[0:8,10:27,:]
	
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
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,self.move_pixels,0,0,0)
		else:
			self.reward = self.reward-.5
	
	def move_mouse_left(self):
		x,y = win32api.GetCursorPos()
		if (x > self.processing_crop['left']):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,-self.move_pixels,0,0,0)
		else:
			self.reward = self.reward-.5

	def move_mouse_up(self):
		x,y = win32api.GetCursorPos()
		if (y > self.processing_crop['top']):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,-self.move_pixels,0,0)
		else:
			self.reward = self.reward-.5
			
	def move_mouse_down(self):
		x,y = win32api.GetCursorPos()
		if (y < (self.processing_crop['top']+self.processing_crop['height']-5)):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,self.move_pixels,0,0)
		else:
			self.reward = self.reward-.5
	
	#returns the screen number
	#1 for main, 2 for play, 3 for end
	def get_screen_number(self,img):
		img = self.crop_image_for_test(img)
		if (np.array_equal(img,self.mainscene)):
			return 1
		elif (np.array_equal(img,self.playscene)):
			#self.reward = self.reward-1
			return 2
		else:
			#self.reward = -1
			return 3
			
	
	def get_screen_number2(self,img):
		img = self.crop_image_for_test(img)
		check_m = np.array_equiv(self.playscene,np.rollaxis(img,2,0))
		return check_m
			
	#Reward Functions
	#Has to be called before shifting mouse position

	#Spatial Reward
	def reward_1(self,img,a):
		frame = img[:,:,3]
		x,y = win32api.GetCursorPos()
		
		if (self.l_o_d == 2):
			x = x-13
			y = y-58
		else :
			x = x-8
			y = y-31

		new_x1 = int(x*self.x_scale_factor)-12
		new_y1 = int(y*self.y_scale_factor)-12

		if a is 0:
			y = y-self.move_pixels
		elif a is 1:
			y = y+self.move_pixels
		elif a is 2:
			x = x-self.move_pixels
		elif a is 3:
			x = x+self.move_pixels

		new_x2 = int(x*self.x_scale_factor)-12
		new_y2 = int(y*self.y_scale_factor)-12

		#Enforcing Conditions to keep frame inside
		#First frame 
		o_x1 = new_x1 if (new_x1 > 0) else 0
		o_y1 = new_y1 if (new_y1 > 0) else 0
		a1 = o_y1-15 if ((o_y1-15) > 0) else 0
		b1 = o_y1+4 if ((o_y1+4) < 110) else 110
		c1 = o_x1-5 if ((o_x1-5) > 0) else 0
		d1 = o_x1+5 if ((o_x1+5) < 84) else 84
		#Second Frame
		o_x2 = new_x2 if (new_x2 > 0) else 0
		o_y2 = new_y2 if (new_y2 > 0) else 0
		a2 = o_y2-15 if ((o_y2-15) > 0) else 0
		b2 = o_y2+4 if ((o_y2+4) < 110) else 110
		c2 = o_x2-5 if ((o_x2-5) > 0) else 0
		d2 = o_x2+5 if ((o_x2+5) < 84) else 84
		
		crop1 = frame[a1:b1,c1:d1]
		crop2 = frame[a2:b2,c2:d2]
		crop1_empty = True if (np.mean(crop1) < 10) else False
		crop2_empty = True if (np.mean(crop2) < 10) else False	

		if (crop1_empty and not crop2_empty):
			self.reward = -1
		elif (crop1_empty and crop2_empty):
			self.reward = 1
		elif (not crop1_empty and crop2_empty):
			self.reward = 2
		else:
			self.reward = 0
		return self.reward

	#Temporal Scaled-Sigmoid Reward
	def reward_2(self,seq):
		survival_time = seq[2]
		#reward = 1.0/(25*(1+math.exp(-survival_time)))
		reward = math.sqrt(survival_time)/5.0
		seq[2] = reward
		#print(survival_time,reward)
		return seq

	def reward_3(self,seq):
		#survival_time = seq[2]
		#reward = (2.0/(1+(25*math.exp(-3*survival_time))))-1
		#seq[2] = reward
		#print(survival_time,reward)
		return seq



def img_normalize(img):
	print("Img Shape: ",img.shape)
	i_max2 = np.amax(img,axis=0)
	print("pImg Max: ",i_max2)
	
	return img
	
def img_standardize(img):
	return stats.zscore(img,axis=0)

def take_shot(game):
		img = game.sct.grab(game.processing_crop)
		#img = misc.imresize(np.array(img)[:,:,1],(110,84))
		img = Image.fromarray(np.array(img)[:,:,1]).resize((84,110))
		return np.expand_dims(np.array(img),axis=2)