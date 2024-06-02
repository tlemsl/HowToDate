import pyautogui
import time
from PIL import ImageGrab
from PIL import Image
import numpy as np
import cv2
from game import Game
import utill
pyautogui.FAILSAFE = False
"""
resolution
720x405
960x540
1024x576
1280x720
1600x900
1920x1080

action
(x, y, type)            
    type:
        c: click
        d: drag
        w: wheel

"""
class DateSimulation:
    def __init__(self, game) -> None:
        self.game=game

    def step(self, state, action, answer, reward):
        next_state, new_answer, new_reward, done=self.game(state, action, answer, reward)
        return  next_state, new_answer, new_reward, done

    def reset(self):
        return 1
    