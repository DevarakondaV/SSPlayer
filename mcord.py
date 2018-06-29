import pywinauto
import win32api
import time



t = time.time()+60
while(True):
    x,y = win32api.GetCursorPos()
    print(x,y)
    if (time.time() > t):
        break