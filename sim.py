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
    def __init__(self, resolution = "720x405", start_pixel = (0,30), scale=(20, 15), roi = ((100,100), (500, 400))) -> None:
        
        self._good_ending   = cv2.imread("images/good_ending.png", cv2.IMREAD_COLOR)
        self._bad_ending    = cv2.imread("images/bad_ending.png", cv2.IMREAD_COLOR)
        self._normal_ending = cv2.imread("images/normal_ending.png", cv2.IMREAD_COLOR)
        self._start_x, self._start_y = start_pixel
        self._width, self._height = map(int, resolution.split('x'))
        
        
        
        self._roi = roi
        self._roi_start_x, self._roi_start_y = self._roi[0]
        self._roi_width, self._roi_height = roi[1][0] - roi[0][0], roi[1][1] - roi[0][1] #map(int, resolution.split('x'))
        
        self._scale = scale
        self._threshold = 0.001
        self._sim_checker = utill.SimilarityChecker(self._normal_ending, self._bad_ending, self._threshold)
    
    def _command_to_pixel(self, x, y):
        return x/self._scale[0]*self._roi_width + self._roi_start_x, y/self._scale[1]*self._roi_height + self._roi_start_y

    def get_image(self):
        return utill.PIL2OpenCV(ImageGrab.grab((self._start_x, self._start_y, self._start_x+self._width, self._start_y + self._height)))
    
    def get_state(self):
        image=self.get_image()
        image_array=np.array(image)
        return utill.numpy2tuple(image_array) #self._wait_for_stabilized()
    
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
        b, v = self._sim_checker.check(state, next_state)
        print(f"Action similarity result: {v}")
        if not b: #유사도 측정 방법 변경
            reward=1
            next_state=self._wait_for_stabilized()
        done=False
        
        type = self._sim_checker.ending_check(next_state)
        if type == 1: #유사도
            done=True
            reward=10
        elif type == 2: #유사도
            done=True
            reward=20
        
        return  next_state, reward, done

    def _wait_for_stabilized(self):
        prev_state = self.get_state()
        while True:
            time.sleep(0.5)
            self._try_to_skip()
            current_state = self.get_state()
            b, v = self._sim_checker.check(prev_state, current_state)
            print(f"Stabilizing similarity result: {v}")
            if b:
                return current_state
            self._try_to_skip()
            prev_state = current_state
    
    def _try_to_skip(self, cnt = 10):
        for i in range(cnt):
            pyautogui.press("space")

    def get_reward(self):
        return 0.0, False

    def reset(self):
        if self._sim_checker.ending_check(self.get_image()):
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
        return self._wait_for_stabilized()
    