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
    def __init__(self, place, start_img, scale=(20, 15), roi = ((120,50), (500, 420))) -> None:
        self._start_x, self._start_y = place[0], place[1]
        self._width, self._height = place[2]-place[0], place[3]-place[1]
        self._start_img=start_img
        
        
        self._roi = roi
        self._roi_start_x, self._roi_start_y = self._roi[0]
        self._roi_width, self._roi_height = roi[1][0] - roi[0][0], roi[1][1] - roi[0][1] #map(int, resolution.split('x'))
        
        self._scale = scale
        self._threshold = 0.001
        #self._sim_checker = utill.SimilarityChecker(self._normal_ending, self._bad_ending, self._threshold)
    
    def _command_to_pixel(self, x, y):
        return x/self._scale[0]*self._roi_width + self._roi_start_x, y/self._scale[1]*self._roi_height + self._roi_start_y

    def get_image(self):
        return ImageGrab.grab((self._start_x, self._start_y, self._start_x+self._width, self._start_y + self._height))
    
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
        if state!=next_state: #유사도 측정 방법 변경
            reward=1
            next_state=self._wait_for_stabilized()
        done=False
        #type = self._sim_checker.ending_check(next_state)
        if next_state == self._start_img: #유사도
            print(1)
            done=True
            reward=utill.cal_ending(state)
        return  next_state, reward, done

    def _wait_for_stabilized(self):
        prev_state = self.get_state()
        while True:
            time.sleep(0.5)
            self._try_to_skip()
            current_state = self.get_state()
            #b, v = self._sim_checker.check(prev_state, current_state)
            # print(f"Stabilizing similarity result: {v}")
            if prev_state==current_state:
                return current_state
            #self._try_to_skip()
            prev_state = current_state
    
    def _try_to_skip(self, cnt = 5):
        for i in range(cnt):
            pyautogui.press("space")

    def get_reward(self):
        return 0.0, False

    def reset(self):
        sleep_time = 0.4
        # if self._sim_checker.ending_check(self.get_image()):
        #     pyautogui.moveTo(360 + self._start_x, 350 + self._start_y)
        #     pyautogui.click()
        #     time.sleep(sleep_time)
        # else:
        pyautogui.moveTo(680 + self._start_x, 30 + self._start_y)
        pyautogui.click()
        
        time.sleep(sleep_time)
        pyautogui.moveTo(360 + self._start_x, 230 + self._start_y)
        pyautogui.click()
        time.sleep(sleep_time)
    

        pyautogui.moveTo(550 + self._start_x, 210 + self._start_y)
        pyautogui.click()
        time.sleep(sleep_time)
        pyautogui.moveTo(150 + self._start_x, 200 + self._start_y)
        pyautogui.click()
        time.sleep(2)
        return self._wait_for_stabilized()
    