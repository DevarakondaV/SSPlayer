from pywinauto import *
import time
import image
from timeit import timeit,Timer
from PIL import Image
import numpy as np
import mss
import mss.tools
import win32api, win32con
import threading
import multiprocessing
from scipy import misc
#multiprocessing.set_start_method('spawn')

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
		self.reward = 0
		self.sct = mss.mss()
	
		
	def launch_app(self,dir):
		app = application.Application().start(dir)
		app.ShapeScape.Wait('visible',timeout=20)
		app.ShapeScape.SetFocus()
		app.ShapeScape.MoveWindow(x=50,y=50)
		self.window_width = app.windows()[0].client_rect().width()
		self.window_height = app.windows()[0].client_rect().height()
		self.window_screen_loc = app.windows()[0].Rectangle()

		
		if (self.l_o_d == 1):
			self.mainscene = np.array(Image.open(r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\testimg\laptop\mainscene.png"))
			self.playscene = np.array(Image.open(r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\testimg\laptop\playscene.png"))
			#self.mainscene = img_standardize(self.mainscene)
			#self.playscene = img_standardize(self.playscene)
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
			#self.mainscene = img_standardize(self.mainscene)
			#self.playscene = img_standardize(self.playscene)
			self.processing_crop = {'top': self.window_screen_loc.top+58,
									'left': self.window_screen_loc.left+13,
									'width': self.window_width,
									'height': self.window_height}
			self.play_click_loc = (243,319)
			self.playing_click_loc = (209,529)
			self.replay_click_loc = (148,518)
		
		return app
	
	def crop_image_for_test(self,img):
		return img[0:8,10:27]
	
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
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,3,0,0,0)
		else:
			self.reward = self.reward-.5
	
	def move_mouse_left(self):
		x,y = win32api.GetCursorPos()
		if (x > self.processing_crop['left']):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,-3,0,0,0)
		else:
			self.reward = self.reward-.5

	def move_mouse_up(self):
		x,y = win32api.GetCursorPos()
		if (y > self.processing_crop['top']):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,-3,0,0)
		else:
			self.reward = self.reward-.5
			
	def move_mouse_down(self):
		x,y = win32api.GetCursorPos()
		if (y < (self.processing_crop['top']+self.processing_crop['height'])):
			win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,3,0,0)
		else:
			self.reward = self.reward-.5
	
	#returns the screen number
	#1 for main, 2 for play, 3 for end
	def get_screen_number(self,img):
		img = self.crop_image_for_test(img)
		if (np.array_equal(img,self.mainscene)):
			return 1
		elif (np.array_equal(img,self.playscene)):
			self.reward = self.reward+1e-5
			return 2
		else:
			self.reward = 0
			return 3
			
	def get_reward(self):
		return self.reward

	def get_window_shape(self):
		return self.app.windows()[0].Rectangle() #self.rect
	
	

def img_normalize(img):
	i_max = [np.amax(img[i]) for i in range(0,4)]
	i_min = [np.amin(img[i]) for i in range(0,4)]
	#i_max = np.amax(img)
	#i_min = np.amin(img)
	diff = [i_max[i]-i_min[i] for i in range(0,4)]
	#img = (1.0/diff)*(img-i_min)
	img = [(1.0/diff[i])*(img[i]-i_min[i]) for i in range(0,4)]
	return img
	
def img_standardize(img):
	img = img_normalize(img)
	#m = [np.mean(img[i]).astype(np.float16) for i in range(0,4)]
	#sdv = [np.std(img[i]).astype(np.float16) for i in range(0,4)]
	#adj_sdv = [max(sdv[i],1.0/np.power(110*84,.5)) for i in range(0,len(sdv))]
	#adj_stdev = max(sdv,1.0/np.power(110*84,.5))
	#img = (1.0/adj_stdev)*(img-m)
	#img = np.array([(1.0/adj_sdv[i])*(img[i]-m[i]) for i in range(0,4)]).astype(np.float16)
	return np.array(img)

def take_shot(game):
		img = game.sct.grab(game.processing_crop)
		img = misc.imresize(np.array(img)[:,:,1],(110,84))
		#img = misc.imresize(img[:,:,1],(110,84))
		return img
		
def multi_add_training_images(q,e,p):
	while not e.is_set():
		q.put(take_shot(p))
		
def get_four(player_instance,process_queue):
	rtn_array = []
	for i in range(0,4):
		rtn_array.append(process_queue.get())
	
	rm_idx = []
	for i in range(0,4):
		if (player_instance.get_screen_number(rtn_array[i]) != 2):
			rm_idx.append(i)
	
	for i in rm_idx:
		rtn_array.pop(i)
	return rtn_array
		
"""
def Run(player_instance,shot_process,process_event,process_queue):
	pp = player_instance.processing_crop
	frames = []
	
	if (player_instance.get_screen_number(take_shot(pp)) == 1):
		player_instance.click_play()
		wait_for(.5)
		click_play()
		wait_for(.5)
		shot_process.start()
		
	while not process_queue().qsize() > 4:
		continue
	
	player_instance.click_to_play()
		
	img_array = []
	while(player_instance.get_screen_number(take_shot(pp)) == 2):
		if process_queue.qsize() > 4:
			img_array = get_four(player_instance,process_queue)
		
		action = get_from_tensorflow(img_array)
		player_instance.perform(action)
		
		
	process_event.set()
	shot_process.terminate()
	return
"""

