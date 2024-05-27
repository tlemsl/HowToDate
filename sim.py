import pyautogui
import time
from PIL import ImageGrab
from PIL import Image
import numpy as np
import cv2

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
    def __init__(self, resolution = "720x405", start_pixel = (0,30), scale=(18,10)) -> None:
        
        self._good_ending   = cv2.imread("images/good_ending.png", cv2.IMREAD_COLOR)
        self._bad_ending    = cv2.imread("images/bad_ending.png", cv2.IMREAD_COLOR)
        self._normal_ending = cv2.imread("images/normal_ending.png", cv2.IMREAD_COLOR)
        self._start_x, self._start_y = start_pixel
        self._width, self._height = map(int, resolution.split('x'))
        self._scale = scale
        self._threshold = 0.01
    
    def _command_to_pixel(self, x, y):
        return x//self._scale[0]*self._width + self._start_x, y//self._scale[1]*self._height + self._start_y

    def get_image(self):
        return ImageGrab.grab((self._start_x, self._start_y, self._start_x+self._width, self._start_y + self._height))
    
    def get_state(self):
        image=ImageGrab.grab((0, 30, 720, 435))
        image_array=np.array(image)
        return image #self._wait_for_stabilized()
    
    def step(self, state, action):
        x, y, type = action
        x, y = self._command_to_pixel(x,y)
        if type=="c":
            pyautogui.moveTo(x,y)
            pyautogui.click()
        elif type == "d":
            pyautogui.dragTo(x,y)
        time.sleep(1)
        next_state=self.get_state()
        reward=0
        print(utill.similiaity(state, next_state))
        if utill.similiaity(state, next_state)>self._threshold: #유사도 측정 방법 변경
            reward=1
            for i in range(30):
                pyautogui.press('enter')
                time.sleep(0.2)
            next_state=self.get_state()
        done=False
        if utill.similiaity(next_state, self._bad_ending) < self._threshold: #유사도
            done=True
            reward=10
        elif  utill.similiaity(next_state, self._normal_ending)  < self._threshold: #유사도
            done=True
            reward=20
        return  next_state, reward, done

    def _wait_for_stabilized(self):
        prev_state = self.get_image()
        while True:
            time.sleep(0.5)
            self._try_to_skip()
            current_state = self.get_image()
            print(utill.similiaity(current_state, prev_state))
            if utill.similiaity(current_state, prev_state) < self._threshold:
                return current_state
            self._try_to_skip()
            prev_state = current_state
    
    def _try_to_skip(self):
        pyautogui.press("space")

    def get_reward(self):
        return 0.0, False

    def reset(self):
        if self.is_ending()[0]:
            pyautogui.moveTo(360 + self._start_x, 350 + self._start_y)
            pyautogui.click()
            time.sleep(0.1)
        else:
            pyautogui.moveTo(690 + self._start_x, 40 + self._start_y)
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.moveTo(360 + self._start_x, 230 + self._start_y)
            pyautogui.click()
            time.sleep(0.1)
        

        pyautogui.moveTo(550 + self._start_x, 210 + self._start_y)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(150 + self._start_x, 200 + self._start_y)
        pyautogui.click()
        time.sleep(2)   

    def is_ending(self):
        img = self.get_image()
        if utill.similiaity(img, self._good_ending) < self._threshold:
            return True, "Good ending"
        elif utill.similiaity(img, self._bad_ending)  < self._threshold:
            return True, "Bad ending"
        elif utill.similiaity(img, self._normal_ending)  < self._threshold:
            return True, "Normal ending"
        else:
            return False, "Not ending scene"


