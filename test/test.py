import importlib.util
modules_path = r'c:\Users\Vishnu\Documents\EngProj\SSplayer\\'
spec = importlib.util.spec_from_file_location("snake", modules_path+"snake_con.py")
snakemod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snakemod)
from PIL import Image
from hlp import cv_disp

img = Image.open("test.png")
img.save("test2.png")
exit()


game = snakemod.snake(2)
game.click_play()
snakemod.wait_for(.2)
game.take_shot()

#cv_disp(game, game.take_shot)
