from pywinauto import *
import time
import image
from timeit import timeit,Timer
from PIL import Image
import numpy as np
import mss
import mss.tools
import win32api, win32con
app_dir = r"C:\Users\devar\Documents\EngProj\SSPlayer\Release.win32\ShapeScape.exe"

def wait_for(sec):
	t = time.time()+sec
	while(True):
		if (time.time()>t):
			break

class SSPlayer:
	def __init__(self,dir,l_or_d):
		self.l_o_d = l_or_d
		self.app = self.launch_app(dir)
		
	def launch_app(self,dir):
		app = application.Application().start(dir)
		app.ShapeScape.Wait('visible',timeout=20)
		app.ShapeScape.SetFocus()
		app.ShapeScape.MoveWindow(x=50,y=50)
		self.window_width = app.windows()[0].client_rect().width()
		self.window_height = app.windows()[0].client_rect().height()
		self.window_screen_loc = app.windows()[0].Rectangle()
		
		if (self.l_o_d == 1):
			#self.test_crop = (8,31,100,85)
			#self.processing_crop = (8,31,328,511)
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
			#333
			#self.test_crop = (13,58,100,113)
			#self.processing_crop = (13,58,333,538)
			self.mainscene = Image.open(r"C:\Users\devar\Documents\EngProj\SSPlayer\testimg\desktop\mainscene.png")
			self.playscene = Image.open(r"C:\Users\devar\Documents\EngProj\SSPlayer\testimg\desktop\playscene.png")
			self.processing_crop = {'top': self.window_screen_loc.top+58,
									'left': self.window_screen_loc.left+13,
									'width': self.window_width,
									'height': self.window_height}
			self.play_click_loc = (243,319)
			self.playing_click_loc = (209,529)
			self.replay_click_loc = (148,518)

		return app
		
		
	def take_screenshot(self):
		img = mss.mss().grab(self.processing_crop)
		self.current_shot = np.array(img)[:,:,1]
	
	def crop_image_for_test(self):
		self.take_screenshot()
		return self.current_shot[0:50,0:100]
	
	def kill(self):
		self.app.kill()
		
	def click_play(self):
		win32api.SetCursorPos(self.play_click_loc)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.play_click_loc[0],self.play_click_loc[1],0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,self.play_click_loc[0],self.play_click_loc[1],0,0)
	
	def click_to_play(self):
		win32api.SetCursorPos(self.playing_click_loc)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.playing_click_loc[0],self.playing_click_loc[1],0,0)
		
	def click_replay(self):
		win32api.SetCursorPos(self.replay_click_loc)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.replay_click_loc[0],self.replay_click_loc[1],0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,self.replay_click_loc[0],self.replay_click_loc[1],0,0)
	
	def release_click(self):
		mouse.release(button='left',coords=self.replay_click_loc)

	def move_mouse(self,cord):
		#win32api.SetCursorPos(cord)
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, cord[0],cord[1],0,0)

	#returns the screen number
	#1 for main, 2 for play, 3 for end
	def get_screen_number(self):
		img = self.crop_image_for_test()
		if (np.array_equal(img,self.mainscene)):
			return 1
		elif (np.array_equal(img,self.playscene)):
			return 2
		else:
			return 3

	def get_window_shape(self):
		return self.app.windows()[0].Rectangle() #self.rect

con = SSPlayer(app_dir,2)
wait_for(1)
con.click_play()
wait_for(1)
con.click_to_play()
wait_for(1)
for i in range(0,30):
	print(Timer(lambda: con.move_mouse((-1,-1))).timeit(number=1))
#wait_for(10)
#con.release_click()
#wait_for(1)
#con.click_replay()
wait_for(3)
con.kill()