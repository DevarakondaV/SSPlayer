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
from scipy import misc


app_dir = r"C:\Users\Vishnu\Documents\EngProj\SSPlayer\Release.win32\ShapeScape.exe"

def wait_for(sec):
	t = time.time()+sec
	while(True):
		if (time.time()>t):
			break

class SSPlayer:
	def __init__(self,dir,l_or_d):
		self.l_o_d = l_or_d
		self.app = self.launch_app(dir)
		#self.tr_images = threading.Lock()
		self.tr_img_ev = threading.Event()
		self.training_images = []
		self.counter = 0
		self.current_screen = 1
		
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
		x,y = win32api.GetCursorPos()
		dx = cord[0]-x
		dy = cord[1]-y
		#win32api.SetCursorPos(cord)
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx,dy,0,0)
		
	def move_mouse_right(self):
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,1,0,0,0)
	
	def move_mouse_left(self):
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,-1,0,0,0)

	def move_mouse_up(self):
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,-1,0,0)
		
	def move_mouse_down(self):
		win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0,1,0,0)
		
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

	def add_training_images(self,e):
		if not e.is_set():
			self.current_screen = self.get_screen_number()
			self.counter = self.counter+1
			self.training_images.append(misc.imresize(self.current_shot,(110,84)))
			threading.Timer(.03,self.add_training_images,[self.tr_img_ev]).start()
	
	
	def get_training_images(self):
		if (len(self.training_images) < 4):
			return []
		else:
			if (len(self.training_images) >= 4):
				im1 = np.array(self.training_images.pop(0))
				im2 = np.array(self.training_images.pop(0))
				im3 = np.array(self.training_images.pop(0))
				im4 = np.array(self.training_images.pop(0))
			
			return [im1,im2,im3,im4]
	
def save_frames(frames):
	p = 1
	for i in range(0,np.shape(frames)[0]):
		for x in range(0,np.shape(frames)[1]):
			Image.fromarray(frames[i][x][:,:]).save("test/img_f"+str(p)+".png")
			p = p+1
	
	return
		

		

gm = SSPlayer(app_dir,1)
wait_for(1)
#gm.add_training_images(gm.tr_img_ev)
#print(Timer(lambda: gm.take_screenshot()).timeit(number=1))
wait_for(2)
n_frames = 0
p = 0
t = time.time()+1
#frames = []
while(True): #gm.current_screen == 2):
	#sp = gm.get_training_images()
	#if (np.shape(sp)[0] != 0):
	#	n_frames = n_frames+4
	#	frames.append(sp)
	gm.take_screenshot()
	Image.fromarray(gm.current_shot).save("test/img_f"+str(p)+".png")
	p = p+1
	#print(Timer(lambda: gm.take_screenshot()).timeit(number=1))
	if (time.time()>t):
		break


gm.tr_img_ev.set()
gm.kill()
print("counter: ",gm.counter)
#print("n_frames: ",n_frames)
#print(np.shape(frames))
#save_frames(frames)