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
	
	def crop_image_for_test(self,img):
		return img[0:8,10:27]
	
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
		x,y = win32api.GetCursorPos()
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

	def move_mouse(self,cord):
		x,y = win32api.GetCursorPos()
		dx = cord[0]-x
		dy = cord[1]-y
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
	def get_screen_number(self,img):
		img = self.crop_image_for_test(img)
		if (np.array_equal(img,self.mainscene)):
			return 1
		elif (np.array_equal(img,self.playscene)):
			return 2
		else:
			return 3

	def get_window_shape(self):
		return self.app.windows()[0].Rectangle() #self.rect
	
	

def save_frames(frames):
	p = 1
	for i in range(0,np.shape(frames)[0]):
		for x in range(0,np.shape(frames)[1]):
			Image.fromarray(frames[i][x][:,:]).save("test/img_f"+str(p)+".png")
			p = p+1
	
	return
		
def save_frames2(frames):
	p = 0
	for i in frames:
		Image.fromarray(i).save("test/img_f"+str(p)+".png")
		p = p+1
	return
	

def take_shot(processing_crop):
		img = mss.mss().grab(processing_crop)
		return misc.imresize(np.array(img)[:,:,1],(110,84))		
		
def multi_add_training_images(q,e,p):
	while not e.is_set():
		q.put(take_shot(p))
		
		
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
	
	test_img = []
	for i in range(0,4):
		test_img.append(process_queue.get())
	while(player_instance.get_screen_number(test_img) == 2):
			
	return 

			
if __name__ == "__main__":
	gm = SSPlayer(app_dir,1)
	t_img = multiprocessing.Queue()
	ev = multiprocessing.Event()
	pp = gm.processing_crop
	p = multiprocessing.Process(target=multi_add_training_images,args=[t_img,ev,pp])

	print(Timer(lambda: Run(gm,p,ev,t_img)).timeit(number=1))
	
